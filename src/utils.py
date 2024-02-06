# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import division
import torch.nn as nn
import torch 
import typing
import gym 
import threading
from torch import multiprocessing as mp
import logging
import traceback
import os 
import numpy as np
import copy

from src.core import prof
from src.env_utils import FrameStack, Environment, Minigrid2Image
from src import atari_wrappers as atari_wrappers

from gym_minigrid import wrappers as wrappers

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from src.seen_utils import *

# from nes_py.wrappers import JoypadSpace
# from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# import vizdoomgym
OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
}

# This augmentation is based on random walk of agents
def augmentation(frames):
    # agent_loc = agent_loc(frames)
    return frames

# Entropy loss on categorical distribution
def catentropy(logits):
    a = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    e = torch.exp(a)
    z = torch.sum(e, dim=-1, keepdim=True)
    p = e / z
    entropy = torch.sum(p * (torch.log(z) - a), dim=-1)
    return torch.mean(entropy)

# Here is computing how many objects
def num_objects(frames):
    T, B, H, W, *_ = frames.shape
    num_objects = frames[:, :, :, :, 0]
    num_objects = (num_objects == 4).long() + (num_objects == 5).long() + \
        (num_objects == 6).long() + (num_objects == 7).long() + (num_objects == 8).long()
    return num_objects

# EMA of the 2 networks
def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def agent_loc(frames):
    T, B, H, W, *_ = frames.shape
    agent_location = torch.flatten(frames, 2, 3)
    agent_location = agent_location[:,:,:,0] 
    agent_location = (agent_location == 10).nonzero() #select object id
    agent_location = agent_location[:,2]
    agent_location = torch.cat(((agent_location//W).unsqueeze(-1), (agent_location%W).unsqueeze(-1)), dim=-1) 
    agent_location = agent_location.view(-1).tolist()
    return agent_location

COMPLETE_MOVEMENT = [
    ['NOOP'],
    ['up'],
    ['down'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['B'],
    ['A', 'B'],
] 

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('torchbeast')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_env(flags):
    if 'MiniGrid' in flags.env:
        return Minigrid2Image(wrappers.FullyObsWrapper(gym.make(flags.env)))
    elif 'Mario' in flags.env:
        env = atari_wrappers.wrap_pytorch(
            atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(flags.env, noop=True),
                clip_rewards=False,
                frame_stack=True,
                scale=False,
                fire=True)) 
        env = JoypadSpace(env, COMPLETE_MOVEMENT)
        return env
    else:
        env = atari_wrappers.wrap_pytorch(
            atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(flags.env, noop=False),
                clip_rewards=False,
                frame_stack=True,
                scale=False,
                fire=False)) 
        return env


def get_batch(free_queue: mp.Queue,
              full_queue: mp.Queue,
              buffers: Buffers,
              initial_agent_state_buffers,
              initial_encoder_state_buffers,
              flags,
              timings,
              lock=threading.Lock()):
    with lock:
        timings.time('lock')
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time('dequeue')
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    if initial_encoder_state_buffers is not None:
        initial_encoder_state = (
            torch.cat(ts, dim=1)
            for ts in zip(*[initial_encoder_state_buffers[m] for m in indices])
        )
    timings.time('batch')
    for m in indices:
        free_queue.put(m)
    timings.time('enqueue')
    batch = {
        k: t.to(device=flags.device, non_blocking=True)
        for k, t in batch.items()
    }
    initial_agent_state = tuple(t.to(device=flags.device, non_blocking=True)
                                for t in initial_agent_state)
    if initial_encoder_state_buffers is not None:
        initial_encoder_state = tuple(t.to(device=flags.device, non_blocking=True)
                                    for t in initial_encoder_state)
    else:
        initial_encoder_state = None
    timings.time('device')
    return batch, initial_agent_state, initial_encoder_state

def create_buffers(obs_shape, num_actions, l, flags) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.uint8),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
        episode_win=dict(size=(T + 1,), dtype=torch.int32),
        carried_obj=dict(size=(T + 1,), dtype=torch.int32),
        carried_col=dict(size=(T + 1,), dtype=torch.int32),
        partial_obs=dict(size=(T + 1, 7, 7, 3), dtype=torch.uint8),
        pano=dict(size=(T + 1, 28, 7, 3), dtype=torch.uint8),
        episode_state_count=dict(size=(T + 1, ), dtype=torch.float32),
        train_state_count=dict(size=(T + 1, ), dtype=torch.float32),
        partial_state_count=dict(size=(T + 1, ), dtype=torch.float32),
        encoded_state_count=dict(size=(T + 1, ), dtype=torch.float32),
        seen_change=dict(size=(T + 1, ), dtype=torch.float32),
        seen=dict(size=(T + 1, ), dtype=torch.float32),
        r_epi=dict(size=(T + 1, ), dtype=torch.float32),
        r_pred=dict(size=(T + 1, ), dtype=torch.float32),
        pano_count=dict(size=(T + 1, ), dtype=torch.float32),
        pred_pano_count=dict(size=(T + 1, ), dtype=torch.float32),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers

def create_heatmap_buffers(obs_shape):
    specs = []
    for r in range(obs_shape[0]):
        for c in range(obs_shape[1]):
            specs.append(tuple([r, c]))
    buffers: Buffers = {key: torch.zeros(1).share_memory_() for key in specs}
    return buffers

def act(i: int, free_queue: mp.Queue, full_queue: mp.Queue,
        model: torch.nn.Module, 
        encoder: torch.nn.Module,
        buffers: Buffers, 
        episode_state_count_dict: dict, train_state_count_dict: dict,
        partial_state_count_dict: dict, encoded_state_count_dict: dict,
        heatmap_dict: dict, 
        heatmap_buffers: Buffers,
        initial_agent_state_buffers, 
        initial_encoder_state_buffers,
        flags,
        forward_model=None):
    try:
        log.info('Actor %i started.', i)
        timings = prof.Timings()  

        gym_env = create_env(flags)
        seed = i ^ int.from_bytes(os.urandom(4), byteorder='little')
        gym_env.seed(seed)
        
        if flags.num_input_frames > 1:
            gym_env = FrameStack(gym_env, flags.num_input_frames)  

        env = Environment(gym_env, fix_seed=flags.fix_seed, env_seed=flags.env_seed, simu_step=flags.simu_step,
                          disable_movable=flags.disable_movable)
        env.forward_model = forward_model

        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        if encoder is not None:
            encoder_state = encoder.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        prev_env_output = None

        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor
            if encoder is not None:
                for i, tensor in enumerate(encoder_state):
                    initial_encoder_state_buffers[index][i][...] = tensor


            # Update the episodic state counts
            if episode_state_count_dict is not None:
                episode_state_key = tuple(env_output['frame'].view(-1).tolist())
                if episode_state_key in episode_state_count_dict:
                    episode_state_count_dict[episode_state_key] += 1
                else:
                    episode_state_count_dict.update({episode_state_key: 1})
                buffers['episode_state_count'][index][0, ...] = \
                    torch.tensor(1 / np.sqrt(episode_state_count_dict.get(episode_state_key)))
            
                # Reset the episode state counts when the episode is over
                if env_output['done'][0][0]:
                    for episode_state_key in episode_state_count_dict:
                        episode_state_count_dict = dict()
            
            if train_state_count_dict is not None:
                # Update the training state counts
                train_state_key = tuple(env_output['frame'].view(-1).tolist())
                if train_state_key in train_state_count_dict:
                    train_state_count_dict[train_state_key] += 1
                else:
                    train_state_count_dict.update({train_state_key: 1})
                buffers['train_state_count'][index][0, ...] = \
                    torch.tensor(1 / np.sqrt(train_state_count_dict.get(train_state_key)))
                    
            if partial_state_count_dict is not None:
                partial_state_key = tuple(env_output['partial_obs'].view(-1).tolist())
                if partial_state_key in partial_state_count_dict:
                    partial_state_count_dict[partial_state_key] += 1
                else:
                    partial_state_count_dict.update({partial_state_key: 1})
                buffers['partial_state_count'][index][0, ...] = \
                    torch.tensor(1 / np.sqrt(partial_state_count_dict.get(partial_state_key)))

            # Update the agent position counts
            if heatmap_buffers is not None:
                heatmap_key = tuple(agent_loc(env_output['frame']))
                heatmap_buffers[heatmap_key] += 1

            # Do new rollout
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)
                    if encoder is not None and flags.model == 'bebold':
                        _, encoder_state = encoder(env_output['partial_obs'], encoder_state, env_output['done'])

                timings.time('model')

                prev_env_output = copy.deepcopy(env_output)
                env_output = env.step(agent_output['action'])

                timings.time('step')

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
    
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                # Update the episodic state counts
                if episode_state_count_dict is not None:
                    episode_state_key = tuple(env_output['frame'].view(-1).tolist())
                    if episode_state_key in episode_state_count_dict:
                        episode_state_count_dict[episode_state_key] += 1
                    else:
                        episode_state_count_dict.update({episode_state_key: 1})
                    buffers['episode_state_count'][index][t + 1, ...] = \
                        torch.tensor(1 / np.sqrt(episode_state_count_dict.get(episode_state_key)))

                    # Reset the episode state counts when the episode is over
                    if env_output['done'][0][0]:
                        episode_state_count_dict = dict()

                if train_state_count_dict is not None:
                    # Update the training state counts
                    train_state_key = tuple(env_output['frame'].view(-1).tolist())
                    if train_state_key in train_state_count_dict:
                        train_state_count_dict[train_state_key] += 1
                    else:
                        train_state_count_dict.update({train_state_key: 1})
                    buffers['train_state_count'][index][t + 1, ...] = \
                        torch.tensor(1 / np.sqrt(train_state_count_dict.get(train_state_key)))

                if partial_state_count_dict is not None:
                    partial_state_key = tuple(env_output['partial_obs'].view(-1).tolist())
                    if partial_state_key in partial_state_count_dict:
                        partial_state_count_dict[partial_state_key] += 1
                    else:
                        partial_state_count_dict.update({partial_state_key: 1})
                    buffers['partial_state_count'][index][t + 1, ...] = \
                        torch.tensor(1 / np.sqrt(partial_state_count_dict.get(partial_state_key)))

                # Update the agent position counts
                if heatmap_buffers is not None:
                    heatmap_key = tuple(agent_loc(env_output['frame']))
                    heatmap_buffers[heatmap_key] += 1

                timings.time('write')
            full_queue.put(index)

        if i == 0:
            log.info('Actor %i: %s', i, timings.summary())

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        logging.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e