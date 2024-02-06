# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gym 
import torch 
from collections import deque, defaultdict
from gym import spaces
import numpy as np
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
from src.seen_utils import *
import random
from copy import deepcopy
from collections import defaultdict
import torch.nn.functional as F

class HashTensorWrapper():
    def __init__(self, tensor):
        self.tensor = tensor

    def __hash__(self):
        return hash(self.tensor.numpy().tobytes())

    def __eq__(self, other):
        return torch.all(self.tensor == other.tensor)


gym.envs.register(
    id='MiniGrid-MultiRoom-N12-S10-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnv',
    kwargs={'minNumRooms' : 12, \
            'maxNumRooms' : 12, \
            'maxRoomSize' : 10},
)

gym.envs.register(
    id='MiniGrid-MultiRoom-N7-S8-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnv',
    kwargs={'minNumRooms' : 7, \
            'maxNumRooms' : 7, \
            'maxRoomSize' : 8},
)

gym.envs.register(
    id='MiniGrid-MultiRoom-N7-S4-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnv',
    kwargs={'minNumRooms' : 7, \
            'maxNumRooms' : 7, \
            'maxRoomSize' : 4},
)


def _format_observation(obs):
    obs = torch.tensor(obs)
    return obs.view((1, 1) + obs.shape) 


class Minigrid2Image(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, observation):
        return observation['image']


class Environment:
    def __init__(self, gym_env, fix_seed=False, env_seed=1, 
                       simu_step=0, disable_movable=False, forward_model=None):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self.episode_win = None
        self.fix_seed = fix_seed
        self.env_seed = env_seed
        self.simu_step = simu_step
        self.disable_movable = disable_movable
        self.forward_model = forward_model

    def get_partial_obs(self, env=None):
        if not env:
            return self.gym_env.env.env.gen_obs()['image']
        return env.env.env.gen_obs()['image']

    def fill_memory(self, obs):
        y, x = self.agent_pos
        rotation = self.agent_rot
        if rotation == 0:
            x_start,y_start=x-3,y-6
            rotated = obs
        elif rotation == 1:
            x_start, y_start = x,y-3
            rotated = np.rot90(obs, 1)
        elif rotation == 2:
            x_start, y_start = x-3,y
            rotated = np.rot90(obs, 2)
        elif rotation == 3:
            x_start, y_start = x-6,y-3
            rotated = np.rot90(obs, 3)
        h, w = rotated.shape[0], rotated.shape[1]
        x_end, y_end = x_start+h, y_start+w

        padded_grid = pad_grid(self.gym_env.unwrapped.grid.encode().copy(), self.pos_offset, self.gym_env.env.env.agent_pos, self.l)
        record = self.seen[x_start:x_end, y_start:y_end].clone()
        self.seen[x_start:x_end, y_start:y_end] = ((record + torch.from_numpy(rotated[:,:,0]!=0).float())>0).float()
        padded_grid = torch.from_numpy(padded_grid).double()*self.seen.unsqueeze(-1).double()
        self.memory = padded_grid
    
    def forward_pos(self, action):
        if action == 0:
            self.agent_rot -= 1
        elif action == 1:
            self.agent_rot += 1
        elif action == 2:
            grid = self.memory
            position = self.agent_pos
            rotation = self.agent_rot
            obj, state, x, y = front_obj(grid, position, rotation)
            if (obj.item()==1) or (obj.item()==4 and state.item()==0):
                self.agent_pos = (x, y)
        self.agent_rot = self.agent_rot%4

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        self.episode_win = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        if self.fix_seed:
            self.gym_env.seed(seed=self.env_seed)
        initial_frame = _format_observation(self.gym_env.reset())
        partial_obs = _format_observation(self.get_partial_obs())
        pano = _format_observation(self.get_panorama().reshape(-1, 7, 3))

        if self.gym_env.env.env.carrying:
            carried_col, carried_obj = torch.LongTensor([[COLOR_TO_IDX[self.gym_env.env.env.carrying.color]]]), torch.LongTensor([[OBJECT_TO_IDX[self.gym_env.env.env.carrying.type]]])
        else:
            carried_col, carried_obj = torch.LongTensor([[5]]), torch.LongTensor([[1]])   

        self.h, self.w = self.gym_env.env.env.height, self.gym_env.env.env.width
        self.l = max(self.h, self.w)
        gt_agent_pos = self.gym_env.env.env.agent_pos
        gt_agent_dir = self.gym_env.env.env.agent_dir
        self.agent_pos, self.agent_rot, self.pos_offset, self.memory, self.seen, self.last_memory = reset_memory(self.l, self.h, self.w, gt_agent_pos, gt_agent_dir)
        self.fill_memory(self.get_partial_obs().copy())
        self.last_seen = 0
        seen_area = self.seen.sum().item()
        self.reset_pano_rec()
        r_epi = self.simulate_reachable_panos()
        r_pred = self.pred_reachable_panos()

        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            episode_win=self.episode_win,
            carried_col = carried_col,
            carried_obj = carried_obj, 
            partial_obs=partial_obs,
            seen_change=self.seen.sum().item()-self.last_seen,
            seen=seen_area,
            r_epi=r_epi,
            r_pred=r_pred,
            pano_count=1,
            pred_pano_count=1,
            pano=pano,
            )

    def reset_pano_rec(self):
        if self.forward_model is None:
            self.episodic_panos, self.pano_idx = {}, 0
        else:
            self.pred_panos, self.pred_idx, self.pred_rec = {}, 0, {}

        
    def step(self, action):
        self.last_seen = self.seen.sum().item()
        self.last_memory = self.memory.clone()
        last_obs = self.process_image(self.get_partial_obs())
        frame, reward, done, _ = self.gym_env.step(action.item())
        if self.forward_model is not None:
            self.update_pred_rec(last_obs, self.process_image(self.get_partial_obs()), action.item())

        self.episode_step += 1
        episode_step = self.episode_step

        self.episode_return += reward
        episode_return = self.episode_return 

        if done and reward > 0:
            self.episode_win[0][0] = 1 
        else:
            self.episode_win[0][0] = 0 
        episode_win = self.episode_win 
        
        self.forward_pos(action.item())
        self.fill_memory(self.get_partial_obs().copy())
        seen_change = self.seen.sum().item()-self.last_seen
        seen_area = self.seen.sum().item()

        r_epi = self.simulate_reachable_panos()
        r_pred = self.pred_reachable_panos()
        pano_count = self.pano_idx if self.forward_model is None else 0 
        pred_pano_count = self.pred_idx if self.forward_model is not None else 0 

        if done:
            if self.fix_seed:
                self.gym_env.seed(seed=self.env_seed)
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
            self.episode_win = torch.zeros(1, 1, dtype=torch.int32)
            gt_agent_pos = self.gym_env.env.env.agent_pos
            gt_agent_dir = self.gym_env.env.env.agent_dir
            self.agent_pos, self.agent_rot, self.pos_offset, self.memory, self.seen, self.last_memory = reset_memory(self.l, self.h, self.w, gt_agent_pos, gt_agent_dir)
            self.fill_memory(self.get_partial_obs().copy())
            self.reset_pano_rec()
            self.simulate_reachable_panos()
            self.pred_reachable_panos()

        frame = _format_observation(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        partial_obs = _format_observation(self.get_partial_obs())
        pano = _format_observation(self.get_panorama().reshape(-1, 7, 3))
        
        if self.gym_env.env.env.carrying:
            carried_col, carried_obj = torch.LongTensor([[COLOR_TO_IDX[self.gym_env.env.env.carrying.color]]]), torch.LongTensor([[OBJECT_TO_IDX[self.gym_env.env.env.carrying.type]]])
        else:
            carried_col, carried_obj = torch.LongTensor([[5]]), torch.LongTensor([[1]])  

        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step = episode_step,
            episode_win = episode_win,
            carried_col = carried_col,
            carried_obj = carried_obj, 
            partial_obs=partial_obs,
            seen_change=seen_change,
            seen=seen_area,
            r_epi=r_epi,
            r_pred=r_pred,
            pano_count=pano_count,
            pred_pano_count=pred_pano_count,
            pano=pano,
            )

    def update_buffer(self, obs, depth):
        hashable_obs = hash(str(obs))
        if hashable_obs not in self.episodic_panos:
            self.pano_idx += 1
            self.episodic_panos[hashable_obs] = depth
        else:
            if self.episodic_panos[hashable_obs] > depth:
                self.episodic_panos[hashable_obs] = depth
            else:
                return False 
        return True 

    def update_pred_rec(self, obs, next_obs, action):
        hashable_next_obs = hash(str(next_obs))
        hashable_obs = hash(str(obs))
        seen_key = str(hashable_obs) + '_' + str(action)
        self.pred_rec[seen_key] = hashable_next_obs

    def update_pred_buffer_single(self, pred, depth):
        hashable_pred_obs = hash(str(pred))
        if hashable_pred_obs not in self.pred_panos: 
            self.pred_idx += 1
            self.pred_panos[hashable_pred_obs] = depth
        else:
            if self.pred_panos[hashable_pred_obs] > depth:
                self.pred_panos[hashable_pred_obs] = depth
            else:
                return False 
        return True 

    def update_pred_buffer(self, obs, pred_obs, depth):
        hashable_obs = hash(str(obs))
        for i in range(pred_obs.shape[0]):
            seen_key = str(hashable_obs) + '_' + str(i)
            if seen_key in self.pred_rec:
                return 
            self.update_pred_buffer_single(self.process_image(pred_obs[i]), depth)

    def pred_reachable_panos(self):
        if self.forward_model is None:
            return 0 
        record = self.pred_idx
        self.update_pred_buffer_single(self.process_image(self.get_partial_obs()), 0)
        actions = torch.Tensor(np.array(list(range(6))))
        pano = np.expand_dims(np.concatenate(self.get_panorama(), axis=0), axis=0)
        panos = torch.Tensor(np.repeat(pano, 6, axis=0))
        actions = F.one_hot(actions.to(torch.int64), num_classes=7)
        [pred_objs, pred_colors, pred_conds], _, _, _ = self.forward_model(panos, actions)
        pred_objs = np.expand_dims(torch.argmax(pred_objs, dim=-1).cpu().data.numpy(), axis=-1)
        pred_colors = np.expand_dims(torch.argmax(pred_colors, dim=-1).cpu().data.numpy(), axis=-1)
        pred_conds = np.expand_dims(torch.argmax(pred_conds, dim=-1).cpu().data.numpy(), axis=-1)
        preds = np.concatenate((pred_objs, pred_colors, pred_conds), axis=-1)
        self.update_pred_buffer(self.process_image(self.get_partial_obs()), preds, 1)
        r_epi = self.get_epi_rew(record, mode='pred')
        return r_epi

    def simulate_reachable_panos(self):
        return 0 
        # this is for gt gobi
        # if self.forward_model is not None:
        #     return 0 
        # record = self.pano_idx
        # self.update_buffer(self.get_partial_obs(process=True), 0)
        # stack = [(self.gym_env, 0)]
        # while stack:
        #     env, depth = stack.pop()
        #     if depth >= self.simu_step:
        #         continue 
        #     for action in range(6): #all possible actions 
        #         forward_env = deepcopy(env)
        #         forward_env.step_count = 0
        #         _, _, done, _ = forward_env.step(action)
        #         if self.update_buffer(self.get_partial_obs(forward_env, process=True), depth+1) and not done:
        #             stack.append((forward_env, depth+1))
        # r_epi = self.get_epi_rew(record)
        # return r_epi

    def get_epi_rew(self, prev_idx, mode='gt'):
        if mode == 'gt':
            return self.pano_idx - prev_idx
        elif mode == 'pred':
            return self.pred_idx - prev_idx

    def process_image(self, frame):
        # can pick up key, ball, box: 5, 6, 7
        if not self.disable_movable:
            return frame 
        objects, color, condition = frame[:,:,0], frame[:,:,1], frame[:,:,2]
        for obj in [5, 6, 7]:
            rows, cols = np.where(objects == obj)
            objects[rows, cols] = 1
            color[rows, cols] = 0
            condition[rows, cols] = 0
        objects = np.expand_dims(objects, axis=-1)
        color = np.expand_dims(color, axis=-1)
        condition = np.expand_dims(condition, axis=-1)
        return np.concatenate((objects, color, condition), axis=-1)

    def get_panorama(self, env=None):
        if env is None:
            env = deepcopy(self.gym_env)
        else:
            env = deepcopy(env)
        pano = []
        for _ in range(4):
            frame, *_ = env.step(1)
            pano.append(env.env.env.gen_obs()['image'])
        return np.array(pano)

    def close(self):
        self.gym_env.close()


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]