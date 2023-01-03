import matplotlib.pyplot as plt
import torch
import copy
import os
import numpy as np
from tqdm import trange

from crowd_nav.configs.config import Config
from rl.networks.envs import make_vec_envs
from crowd_sim.envs import *

# train_data: True if collect training data, False if collect testing data
def collectData(device, train_data, config):
    # set robot policy to orca
    config.robot.policy = 'orca'

    env_name = 'CrowdSimVarNumCollect-v0'
    # for render
    env_num = 1 if config.data.render else config.data.num_processes

    if config.data.render:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        plt.ion()
        plt.show()
    else:
        ax = None

    # create parallel envs
    seed = np.random.randint(0, np.iinfo(np.uint32).max)
    envs = make_vec_envs(env_name, seed, env_num,
                         config.reward.gamma, None, device, allow_early_resets=True, config=config,
                         ax=ax, wrap_pytorch=False)


    # collect data for pretext training
    data = [] # list for all data collected
    for i in range(env_num):
        data.append([])

    obs = envs.reset()

    # 1 epoch -> 1 file
    # todo: add pretext arguments to config.py
    # make one prediction every "pred_interval" simulation steps
    pred_interval = config.data.pred_timestep // config.env.time_step
    tot_steps = int(config.data.tot_steps * pred_interval)
    for step in trange(tot_steps):

        if config.data.render:
            envs.render()
        if step % pred_interval == 0:
            # append a single data one by one
            for i in range(env_num):
                non_empty_obs = obs['pred_info'][i][np.logical_not(np.isinf(obs['pred_info'][i, :, -1]))]
                non_empty_obs = non_empty_obs.reshape((-1, 4)).tolist()
                if non_empty_obs:
                    data[i].extend(copy.deepcopy(non_empty_obs))

        action = np.zeros((env_num, 2))
        # action is is dummy action!
        obs, rew, done, info = envs.step(action)

    # save observations as pickle files
    filePath = os.path.join(config.data.data_save_dir, 'train') if train_data \
        else os.path.join(config.data.data_save_dir, 'test')
    if not os.path.isdir(filePath):
        os.makedirs(filePath)

    for i in range(env_num):
        with open(os.path.join(filePath, str(i)+'.txt'), 'w') as f:
            for item in data[i]:
                item = str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\t' + str(item[3])
                f.write("%s\n" % item)


    envs.close()

if __name__ == '__main__':
    config = Config()
    device = torch.device("cuda")
    collectData(device, config.data.collect_train_data, config)
