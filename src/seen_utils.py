import numpy as np 
import pdb 
import sys 
import wandb 
import torch 
from pdb import set_trace as st
import copy 

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def front_obj(grid, position, rotation):
    x, y = position
    if rotation == 0:
        x, y = x-1, y
    elif rotation == 1:
        x, y = x, y+1
    elif rotation == 2:
        x, y = x+1, y
    else:
        x, y = x, y-1
    obj = grid[:,:,0][y,x]
    state = grid[:,:,2][y,x]
    return obj, state, x, y

def decode_single(obs, channel):
    if channel==0:
        idx_to_color = {0: np.array([0, 0, 0]),
                        1: np.array([50, 50, 50]),
                        2: np.array([100, 100, 100]), #gray empty
                        3: np.array([255, 255, 0]),
                        4: np.array([50, 155, 155]),
                        5: np.array([255, 0, 255]), #purple key 
                        6: np.array([0, 0, 255]),
                        7: np.array([0, 255, 55]),
                        8: np.array([0, 255, 0]), #green goal
                        10: np.array([255, 0, 0])} #red agent
    elif channel == 2:
        idx_to_color = {0: np.array([0, 0, 0]), #open
                        1: np.array([0, 0, 255]), #closed
                        2: np.array([255, 0, 0])} #locked
    elif channel == 1:
        idx_to_color = {0: np.array([0, 0, 0]),
                        1: np.array([0, 255, 0]),
                        2: np.array([0, 0, 255]),
                        3: np.array([112, 39, 195]),
                        4: np.array([255, 255, 0]),
                        5: np.array([255, 0, 0])} #seen
    elif channel == 3:
        idx_to_color = {0: np.array([255, 0, 0]), #red, turn left 
                        1: np.array([0, 255, 0]), #green, turn right 
                        2: np.array([0, 0, 255]), #blue, forward
                        3: np.array([112, 39, 195]), #purple, pickup 
                        4: np.array([255, 255, 0]), #yellow, drop
                        5: np.array([100, 100, 100]), #dark gray, toggle 
                        6: np.array([200, 200, 200])} #light gray, done 

    H,W = obs.shape
    image = np.zeros((H,W,3))
    for idx in idx_to_color.keys():
        rows, cols = np.where(obs[:,:]==idx)
        image[rows, cols] = idx_to_color[idx]
    return image

def get_delta(start_dir, x, y, h, w, position):
    if start_dir == 1:
        x, y = h-1-x, w-1-y
    elif start_dir == 2:
        x, y = y, h-1-x
    elif start_dir == 0:
        x, y = w-1-y, x
    return position[0]-x, position[1]-y
    
    
def reset_memory(l, h, w, agent_pos, agent_dir):
    record_pos = (l+6, l+6)
    memory = torch.zeros(l*2+12, l*2+12, 3)
    last_memory = torch.zeros(l*2+12, l*2+12, 3)
    seen = torch.zeros(l*2+12, l*2+12)
    agent_rotation = 0
    y, x = agent_pos
    start_dir = agent_dir
    del_x, del_y = get_delta(start_dir, x, y, h, w, record_pos)
    pos_offset = [del_x, del_y, start_dir]
    return record_pos, agent_rotation, pos_offset, memory, seen, last_memory


def pad_grid(full_state, pos_offset, gt_agent_pos, l):
    del_x, del_y, start_dir = pos_offset
    y, x = gt_agent_pos
    if start_dir == 1:
        full_state = np.rot90(full_state, 2)
    elif start_dir == 2:
        full_state = np.rot90(full_state, 1)
    elif start_dir == 0:
        full_state = np.rot90(full_state, 3)
    h, w, _ = full_state.shape
    empty = np.zeros((l*2+12, l*2+12, 3))
    empty[del_y:del_y+h, del_x:del_x+w] = full_state
    return empty

def decode_visualizable(grid, positions, panos):
    original = copy.deepcopy(grid)
    img = decode_single(original[:,:,0])
    for x, y in positions:
        grid[x][y][0] = 10
    reachable = decode_single(grid[:,:,0])
    pano_imgs = []
    for pano in panos:
        curr = []
        for dim in range(4):
            curr.append(decode_single((pano.tensor).numpy()[dim][:,:,0]))
        pano_imgs.append(np.array(curr).reshape((-1, 7, 3)))
    return np.concatenate((img, reachable), axis=0), np.concatenate(pano_imgs, axis=1)


