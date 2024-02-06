# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import threading
import time
import timeit
import pprint
import json

import numpy as np
from collections import deque

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from src.core import file_writer
from src.core import prof
from src.core import vtrace

from src.seen_utils import *
import src.models as models
import src.losses as losses
from src.core.prediction_train import train_pred
import wandb 

from src.env_utils import FrameStack
from src.utils import get_batch, log, create_env, create_buffers, act, create_heatmap_buffers

MinigridPolicyNet = models.MinigridPolicyNet
MinigridStateEmbeddingNet = models.MinigridStateEmbeddingNet
MinigridMLPEmbeddingNet = models.MinigridMLPEmbeddingNet
MinigridMLPTargetEmbeddingNet = models.MinigridMLPTargetEmbeddingNet

def learn(actor_model,
          model,
          random_target_network,
          predictor_network,
          actor_encoder,
          encoder,
          batch,
          initial_agent_state, 
          initial_encoder_state,
          optimizer,
          predictor_optimizer,
          scheduler,
          flags,
          frames=None,
          lock=threading.Lock()):
    """Performs a learning (optimization) step."""
    with lock:
        count_rewards = torch.ones((flags.unroll_length, flags.batch_size), 
            dtype=torch.float32).to(device=flags.device)
        # Use the scale of square root N
        count_rewards = batch['partial_state_count'][1:].float().to(device=flags.device)
        intrinsic_rewards = batch['r_pred'][1:].float().to(device=flags.device)
        intrinsic_rewards = intrinsic_rewards*count_rewards 

        count_reward_coef = flags.count_reward_coef
        weighted_count_rewards = count_reward_coef*count_rewards

        intrinsic_rewards = intrinsic_rewards + weighted_count_rewards

        intrinsic_reward_coef = flags.intrinsic_reward_coef
        intrinsic_rewards *= intrinsic_reward_coef
        
        learner_outputs, unused_state = model(batch, initial_agent_state)

        bootstrap_value = learner_outputs['baseline'][-1]

        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }
        
        rewards = batch['reward']
        intrinsic_rewards = intrinsic_rewards*((1 - flags.intrinsic_decay)**(frames))
            
        if flags.no_reward: #ignore this for this codebase as it doesn't deal with the environment termination signal the same as in C-BET. 
            total_rewards = intrinsic_rewards
        else:            
            total_rewards = rewards + intrinsic_rewards
        clipped_rewards = torch.clamp(total_rewards, -1, 1)
        
        discounts = (~batch['done']).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch['policy_logits'],
            target_policy_logits=learner_outputs['policy_logits'],
            actions=batch['action'],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value)

        pg_loss = losses.compute_policy_gradient_loss(learner_outputs['policy_logits'],
                                               batch['action'],
                                               vtrace_returns.pg_advantages)
        baseline_loss = flags.baseline_cost * losses.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs['baseline'])
        entropy_loss = flags.entropy_cost * losses.compute_entropy_loss(
            learner_outputs['policy_logits'])

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch['episode_return'][batch['done']]
        episode_seens = batch['seen'][batch['done']]
        pred_pano_counts = batch['pred_pano_count'][batch['done']]
        stats = {
            'train/mean_episode_return': torch.mean(episode_returns).item(),
            'train/mean_episode_seen': torch.mean(episode_seens).item(),
            'train/mean_pred_pano_count': torch.mean(pred_pano_counts).item(),
            'loss/total_loss': total_loss.item(),
            'loss/pg_loss': pg_loss.item(),
            'loss/baseline_loss': baseline_loss.item(),
            'loss/entropy_loss': entropy_loss.item(),
            'rew/mean_rewards': torch.mean(rewards).item(),
            'rew/mean_intrinsic_rewards': torch.mean(intrinsic_rewards).item(),
            'rew/mean_total_rewards': torch.mean(total_rewards).item(),
            'rew/mean_count_rewards': torch.mean(count_rewards).item() * intrinsic_reward_coef,
        }
        
        scheduler.step()
        optimizer.zero_grad()
        predictor_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        nn.utils.clip_grad_norm_(predictor_network.parameters(), flags.max_grad_norm)
        optimizer.step()
        predictor_optimizer.step()

        actor_model.load_state_dict(model.state_dict())
        return stats

def train(flags):
    if flags.xpid is None:
        flags.xpid = flags.env + '-predx-%s' % time.strftime('%Y%m%d-%H%M%S')
    plogger = file_writer.FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )

    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                         'model.tar')))

    flags.disable_movable = True if 'ObstructedMaze' in flags.env else False
    pred_model = train_pred(flags)

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        log.info('Using CUDA.')
        flags.device = torch.device('cuda')
    else:
        log.info('Not using CUDA.')
        flags.device = torch.device('cpu')

    env = create_env(flags)
    if flags.num_input_frames > 1:
        env = FrameStack(env, flags.num_input_frames)  

    if 'MiniGrid' in flags.env: 
        if flags.use_fullobs_policy:
            raise Exception('We have not implemented full ob policy!')
        else:
            model = MinigridPolicyNet(env.observation_space.shape, env.action_space.n)    
        random_target_network = MinigridMLPTargetEmbeddingNet().to(device=flags.device) 
        predictor_network = MinigridMLPEmbeddingNet().to(device=flags.device) 
    else:
        raise Exception('Only MiniGrid is suppported Now!')

    expand_l = max(env.env.env.width, env.env.env.height)
    buffers = create_buffers(env.observation_space.shape, model.num_actions, 2*expand_l+12, flags)
    model.share_memory()
    
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context('fork')
    free_queue = ctx.Queue()
    full_queue = ctx.Queue()

    episode_state_count_dict = dict()
    train_state_count_dict = dict()
    partial_state_count_dict = dict()
    encoded_state_count_dict = dict()
    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, model, None, buffers, 
                None, None, partial_state_count_dict, None,
                None, None, initial_agent_state_buffers, None, flags, pred_model))
            # free_queue, full_queue, model, encoder, buffers, 
            # episode_state_count_dict, train_state_count_dict,
            # partial_state_count_dict, encoded_state_count_dict,
            # heatmap_dict, heatmap_buffers, initial_agent_state_buffers, initial_encoder_state_buffers,
        actor.start()
        actor_processes.append(actor)

    if 'MiniGrid' in flags.env: 
        if flags.use_fullobs_policy:
            raise Exception('We have not implemented full ob policy!')
        else:
            learner_model = MinigridPolicyNet(env.observation_space.shape, env.action_space.n)\
                .to(device=flags.device)
    else:
        raise Exception('Only MiniGrid is suppported Now!')

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    predictor_optimizer = torch.optim.Adam(
        predictor_network.parameters(), 
        lr=flags.predictor_learning_rate)

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger('logfile')
    stat_keys = [
        'loss/total_loss',
        'train/mean_episode_return',
        'train/mean_episode_seen',
        'train/mean_pred_pano_count',
        'loss/pg_loss',
        'loss/baseline_loss',
        'loss/entropy_loss',
        'rew/mean_rewards',
        'rew/mean_intrinsic_rewards',
        'rew/mean_total_rewards',
        'rew/mean_count_rewards',
    ]

    logger.info('# Step\t%s', '\t'.join(stat_keys))

    frames, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats
        timings = prof.Timings()
        while frames < flags.total_frames:
            timings.reset()
            batch, agent_state, _ = get_batch(free_queue, full_queue, buffers, 
                initial_agent_state_buffers, None, flags, timings)
            stats = learn(model, learner_model, random_target_network, predictor_network,
                          None, None, batch, agent_state, None, optimizer, 
                          predictor_optimizer, scheduler, flags, frames=frames)
            timings.time('learn')
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                wandb.log(to_log)
                frames += T * B

        if i == 0:
            log.info('Batch and learn: %s', timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []    
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,))
        thread.start()
        threads.append(thread)


    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        checkpointpath = os.path.expandvars(os.path.expanduser(
            '%s/%s/%s' % (flags.savedir, flags.xpid,'model_'+str(frames)+'.tar')))
        log.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'model_state_dict': model.state_dict(),
            # 'random_target_network_state_dict': random_target_network.state_dict(),
            # 'predictor_network_state_dict': predictor_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'predictor_optimizer_state_dict': predictor_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'flags': vars(flags),
        }, checkpointpath)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()

            fps = (frames - start_frames) / (timer() - start_time)
            
            if stats.get('episode_returns', None):
                mean_return = 'Return per episode: %.1f. ' % stats[
                    'train/mean_episode_return']
            else:
                mean_return = ''

            total_loss = stats.get('total_loss', float('inf'))
            if stats:
                log.info('After %i frames: loss %f @ %.1f fps. Mean Return %.1f. \n Stats \n %s', \
                        frames, total_loss, fps, stats['train/mean_episode_return'], pprint.pformat(stats))

    except KeyboardInterrupt:
        return  
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint(frames)
    plogger.close()