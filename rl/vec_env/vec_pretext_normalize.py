from . import VecEnvWrapper
import numpy as np
from .running_mean_std import RunningMeanStd
import torch
import os
from collections import deque

import copy
import pickle

from gst_updated.src.gumbel_social_transformer.temperature_scheduler import Temp_Scheduler
from gst_updated.scripts.wrapper.crowd_nav_interface_parallel import CrowdNavPredInterfaceMultiEnv

class VecPretextNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that processes the observations and rewards, used for GST predictors
    and returns from an environment.
    config: a Config object
    test: whether we are training or testing
    """

    def __init__(self, venv, ob=False, ret=False, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, config=None, test=False):
        VecEnvWrapper.__init__(self, venv)

        self.config = config
        self.device=torch.device(self.config.training.device)
        if test:
            self.num_envs = 1
        else:
            self.num_envs = self.config.env.num_processes

        self.max_human_num = config.sim.human_num + config.sim.human_num_range

        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = torch.zeros(self.num_envs).to(self.device)
        self.gamma = gamma
        self.epsilon = epsilon

        # load and configure the prediction model
        load_path = os.path.join(os.getcwd(), self.config.pred.model_dir)
        if not os.path.isdir(load_path):
            raise RuntimeError('The result directory was not found.')
        checkpoint_dir = os.path.join(load_path, 'checkpoint')
        with open(os.path.join(checkpoint_dir, 'args.pickle'), 'rb') as f:
            self.args = pickle.load(f)

        self.predictor = CrowdNavPredInterfaceMultiEnv(load_path=load_path, device=self.device, config = self.args, num_env = self.num_envs)

        temperature_scheduler = Temp_Scheduler(self.args.num_epochs, self.args.init_temp, self.args.init_temp, temp_min=0.03)
        self.tau = temperature_scheduler.decay_whole_process(epoch=100)

        # handle different prediction and control frequency
        self.pred_interval = int(self.config.data.pred_timestep//self.config.env.time_step)
        self.buffer_len = (self.args.obs_seq_len - 1) * self.pred_interval + 1



    def talk2Env_async(self, data):
        self.venv.talk2Env_async(data)


    def talk2Env_wait(self):
        outs=self.venv.talk2Env_wait()
        return outs

    def step_wait(self):
        obs, rews, done, infos = self.venv.step_wait()

        # process the observations and reward
        obs, rews = self.process_obs_rew(obs, done, rews=rews)

        return obs, rews, done, infos

    def _obfilt(self, obs):
        if self.ob_rms and self.config.RLTrain:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        # queue for inputs to the pred model
        # fill the queue with dummy values
        self.traj_buffer = deque(list(-torch.ones((self.buffer_len, self.num_envs, self.max_human_num, 2), device=self.device)*999),
                                 maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 2)
        self.mask_buffer = deque(list(torch.zeros((self.buffer_len, self.num_envs, self.max_human_num, 1), dtype=torch.bool, device=self.device)),
                                 maxlen=self.buffer_len) # (n_env, num_peds, obs_seq_len, 1)

        self.step_counter = 0

        # for calculating the displacement of human positions
        self.last_pos = torch.zeros(self.num_envs, self.max_human_num, 2).to(self.device)

        obs = self.venv.reset()
        obs, _ = self.process_obs_rew(obs, np.zeros(self.num_envs))

        return obs


    '''
    1. Process observations: 
    Run inference on pred model with past obs as inputs, fill in the predicted trajectory in O['spatial_edges']
    
    2. Process rewards: 
    Calculate reward for colliding with predicted future traj and add to the original reward, 
    same as calc_reward() function in crowd_sim_pred.py except the data are torch tensors
    '''
    def process_obs_rew(self, O, done, rews=0.):
        # O: robot_node: [nenv, 1, 7], spatial_edges: [nenv, observed_human_num, 2*(1+predict_steps)],temporal_edges: [nenv, 1, 2],
        # pos_mask: [nenv, max_human_num], pos_disp_mask: [nenv, max_human_num]
        # prepare inputs for pred_model
        # find humans' absolute positions
        human_pos = O['robot_node'][:, :, :2] + O['spatial_edges'][:, :, :2]

        # insert the new ob to deque
        self.traj_buffer.append(human_pos)
        self.mask_buffer.append(O['visible_masks'].unsqueeze(-1))
        # [obs_seq_len, nenv, max_human_num, 2] -> [nenv, max_human_num, obs_seq_len, 2]
        in_traj = torch.stack(list(self.traj_buffer)).permute(1, 2, 0, 3)
        in_mask = torch.stack(list(self.mask_buffer)).permute(1, 2, 0, 3).float()

        # index select the input traj and input mask for GST
        in_traj = in_traj[:, :, ::self.pred_interval]
        in_mask = in_mask[:, :, ::self.pred_interval]

        # forward predictor model
        out_traj, out_mask = self.predictor.forward(input_traj=in_traj, input_binary_mask=in_mask)
        out_mask = out_mask.bool()

        # add penalties if the robot collides with predicted future pos of humans
        # deterministic reward, only uses mu_x, mu_y and a predefined radius
        # constant radius of each personal zone circle
        # [nenv, human_num, predict_steps]
        hr_dist_future = out_traj[:, :, :, :2] - O['robot_node'][:, :, :2].unsqueeze(1)
        # [nenv, human_num, predict_steps]
        collision_idx = torch.norm(hr_dist_future, dim=-1) < self.config.robot.radius + self.config.humans.radius

        # [1,1, predict_steps]
        # mask out invalid predictions
        # [nenv, human_num, predict_steps] AND [nenv, human_num, 1]
        collision_idx = torch.logical_and(collision_idx, out_mask)
        coefficients = 2. ** torch.arange(2, self.config.sim.predict_steps + 2, device=self.device).reshape(
            (1, 1, self.config.sim.predict_steps))  # 4, 8, 16, 32, 64

        # [1, 1, predict_steps]
        collision_penalties = self.config.reward.collision_penalty / coefficients

        # [nenv, human_num, predict_steps]
        reward_future = collision_idx.to(torch.float)*collision_penalties
        # [nenv, human_num, predict_steps] -> [nenv, human_num*predict_steps] -> [nenv,]
        # keep the values & discard indices
        reward_future, _ = torch.min(reward_future.reshape(self.num_envs, -1), dim=1)
        # print(reward_future)
        # seems that rews is on cpu
        rews = rews + reward_future.reshape(self.num_envs, 1).cpu().numpy()

        # get observation back to env
        robot_pos = O['robot_node'][:, :, :2].unsqueeze(1)

        # convert from positions in world frame to robot frame
        out_traj[:, :, :, :2] = out_traj[:, :, :, :2] - robot_pos

        # only take mu_x and mu_y
        out_mask = out_mask.repeat(1, 1, self.config.sim.predict_steps * 2)
        new_spatial_edges = out_traj[:, :, :, :2].reshape(self.num_envs, self.max_human_num, -1)
        O['spatial_edges'][:, :, 2:][out_mask] = new_spatial_edges[out_mask]

        # sort all humans by distance to robot
        # [nenv, human_num]
        hr_dist_cur = torch.linalg.norm(O['spatial_edges'][:, :, :2], dim=-1)
        sorted_idx = torch.argsort(hr_dist_cur, dim=1)
        # sorted_idx = sorted_idx.unsqueeze(-1).repeat(1, 1, 2*(self.config.sim.predict_steps+1))
        for i in range(self.num_envs):
            O['spatial_edges'][i] = O['spatial_edges'][i][sorted_idx[i]]

        obs={'robot_node':O['robot_node'],
            'spatial_edges':O['spatial_edges'],
            'temporal_edges':O['temporal_edges'],
            'visible_masks':O['visible_masks'],
             'detected_human_num': O['detected_human_num'],

        }

        self.last_pos = copy.deepcopy(human_pos)
        self.step_counter = self.step_counter + 1

        return obs, rews
