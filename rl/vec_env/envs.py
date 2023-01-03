import os

import gym
import numpy as np

from gym.spaces.box import Box

from .vec_env import VecEnvWrapper
import torch

from .dummy_vec_env import DummyVecEnv
from .shmem_vec_env import ShmemVecEnv


from .vec_pretext_normalize import VecPretextNormalize


def make_env(env_id, seed, rank):
    def _thunk():

        env = gym.make(env_id)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  device,
                  randomCollect,
                  config=None):
    envs = [
        make_env(env_name, seed, i)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs) if os.name == 'nt' else ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    if not randomCollect: # if it is not for random data collection phase
        if gamma is None:
            envs = VecPretextNormalize(envs, ob=False, ret=False, config=config)
        else:
            envs = VecPretextNormalize(envs, ob=False, gamma=gamma, config=config)

        envs = VecPyTorch(envs, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, dict):
            for key in obs:
                obs[key]=torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if isinstance(obs, dict):
            for key in obs:
                obs[key] = torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info





