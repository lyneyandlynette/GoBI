import time
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import torch.nn.functional as F
from pdb import set_trace as st
from src.core.prediction import PredModel
import numpy as np 
from src.utils import get_batch, log, create_env, create_buffers, act, create_heatmap_buffers
import os
from torch.utils.data import Dataset
from pathlib import Path
import pickle 

class MiniGridDataset(Dataset):
    def __init__(self, pano_buffer, obs_buffer, action_buffer, next_obs_buffer, mode='train'):
        self.mode = mode
        self.pano_buffer = pano_buffer
        self.obs_buffer = obs_buffer
        self.next_obs_buffer = next_obs_buffer
        self.action_buffer = action_buffer
        self.data_num = self.pano_buffer.shape[0]
        print("The total number of data is", self.data_num)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        pano = self.pano_buffer[idx]
        next_obs = self.next_obs_buffer[idx]
        obs = self.obs_buffer[idx]
        action = self.action_buffer[idx]
        return pano, action, next_obs, obs



def train_pred(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:0')
    kwargs = {'decoder_layer_sizes': [256, 512],
              'latent_size': 8,
              'noise_dim': 4, 
              'num_classes': 11,
              'noise_layer_sizes': [16, 16],
              'noise_lambda': 0.5,
              }

    path = f'./data/{args.env}/'
    file_names = [path+'50000.npy', path+'100000.npy']
    buffer_size = 100000
    pano_buffer = np.zeros((buffer_size, 28, 7, 3))
    obs_buffer = np.zeros((buffer_size, 7, 7, 3))
    action_buffer = np.zeros((buffer_size, 1))
    next_obs_buffer = np.zeros((buffer_size, 7, 7, 3))
    start_idx = 0
    # load data 
    for file_name in file_names:
        p = Path(file_name)
        with p.open('rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            while f.tell() < fsz:
                data = np.load(f)
                loaded_num = data.shape[0]
                panos = data[:, :588].astype(int).reshape((-1, 28, 7, 3))
                obs = data[:, 588:735].astype(int).reshape((-1, 7, 7, 3))
                next_obs = data[:, 735:882].astype(int).reshape((-1, 7, 7, 3))
                actions = data[:, -1].astype(int).reshape((-1, 1))
                pano_buffer[start_idx:start_idx + loaded_num] = panos
                obs_buffer[start_idx:start_idx + loaded_num] = obs
                action_buffer[start_idx:start_idx + loaded_num] = actions
                next_obs_buffer[start_idx:start_idx + loaded_num] = next_obs
                start_idx += loaded_num

    _, indices = np.unique(np.concatenate((pano_buffer.reshape((buffer_size, -1)), action_buffer), axis=-1), axis=0, return_index=True)
    obs_buffer, action_buffer, next_obs_buffer = obs_buffer[indices], action_buffer[indices], next_obs_buffer[indices]
    pano_buffer = pano_buffer[indices]
    if args.disable_movable:
        next_obs_buffer = process_image(next_obs_buffer)
    dataset = MiniGridDataset(pano_buffer, obs_buffer, action_buffer, next_obs_buffer)
    
    split = int(0.8*dataset.data_num)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [split, dataset.data_num-split])
    data_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)
    num_actions = 7
    num_labels = [11, 6, 3]

    model = PredModel(decoder_layer_sizes=kwargs['decoder_layer_sizes'],
              noise_layer_sizes=kwargs['noise_layer_sizes'],
              num_actions=num_actions,
              no_noise=False,
              noise_dim=kwargs['noise_dim'],
              num_labels=num_labels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(80):
        print("epoch", epoch)
        losses = []

        sample_pano, sample_action, sample_next_obs, sample_obs = next(iter(test_data_loader))
        sample_preds, _, _, _ = model(sample_pano.to(device), F.one_hot(sample_action.to(torch.int64), num_classes=num_actions).to(device))
        for iteration, (panos, actions, next_obs, obs) in enumerate(data_loader):
            actions = F.one_hot(actions.to(torch.int64), num_classes=num_actions).to(device)

            preds, mean, log_var, z = model(panos.to(device), actions)
            next_objs, next_colors, next_conditions = next_obs[:,:,:,0], next_obs[:,:,:,1], next_obs[:,:,:,2]
            next_objs = F.one_hot(next_objs.to(torch.int64), num_classes=num_labels[0]).to(device)
            next_colors = F.one_hot(next_colors.to(torch.int64), num_classes=num_labels[1]).to(device)
            next_conditions = F.one_hot(next_conditions.to(torch.int64), num_classes=num_labels[2]).to(device)
            
            loss = model.loss(preds, [next_objs, next_colors, next_conditions], mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(np.mean(losses))
    return model.cpu()
