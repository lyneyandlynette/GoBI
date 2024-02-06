import os
import sys
import threading
import time
import timeit
import json

from pathlib import Path
import numpy as np
import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from src.core import file_writer
from src.core import prof
from src.core import vtrace

import src.models as models
import src.losses as losses
from src.seen_utils import *
import wandb 
from pdb import set_trace as st

from src.env_utils import FrameStack, Environment, Minigrid2Image
from src.utils import get_batch, log, create_env, create_buffers, act, create_heatmap_buffers
from src.env_utils import _format_observation

MinigridPolicyNet = models.MinigridPolicyNet

def collect_data(flags):  
    flags.device = torch.device('cuda')
    gym_env = create_env(flags)
    model = MinigridPolicyNet(gym_env.observation_space.shape, gym_env.action_space.n)    
    
    env = Environment(gym_env, fix_seed=False, env_seed=flags.env_seed)
    env_output = env.initial()
    agent_state = model.initial_state(batch_size=1) 
    data_size, max_size = flags.num_data, 50000
    saving_cycle = data_size // max_size
    panos = torch.zeros((data_size, 28, 7, 3))
    observations = torch.zeros((data_size, 7, 7, 3))
    next_observations = torch.zeros((data_size, 7, 7, 3))
    actions = torch.zeros((data_size, 1))
    step_count = 0 
    done = False 
    file_name = f'./data/{flags.env}'
    Path(file_name).mkdir(parents=True, exist_ok=True)
    while step_count < data_size:
        # take a random action
        agent_output, agent_state = model(env_output, agent_state)
        pano, action = env_output['pano'][0][0], agent_output['action']
        obs = _format_observation(env.get_partial_obs())[0][0]
        env_output = env.step(action)
        done = env_output['done'][0][0]
        if not done:
            panos[step_count] = pano
            actions[step_count] = action
            observations[step_count] = obs
            next_observations[step_count] = _format_observation(env.get_partial_obs())[0][0]
            step_count += 1
        for i in range(1, saving_cycle+1):
            prev, now = max_size * (i - 1), max_size * i
            if step_count == now:
                pano_save, obs_save, next_obs_save, action_save = panos[prev:now,:,:,:].data.numpy(), observations[prev:now].data.numpy(), next_observations[prev:now].data.numpy(), actions[prev:now]
                pano_save, obs_save, next_obs_save, action_save = pano_save.reshape((max_size, -1)), obs_save.reshape((max_size, -1)), next_obs_save.reshape((max_size, -1)), action_save.reshape((max_size, -1))
                to_save = np.concatenate((pano_save, obs_save, next_obs_save, action_save), axis = -1)
                with open(file_name + '/' + str(now) + '.npy', 'wb') as f:
                    np.save(f, to_save)
        if step_count % 1000 == 0:
            print(step_count)