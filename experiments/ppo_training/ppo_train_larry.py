"""Some testing with pytorch"""

import torch
from tqdm import tqdm
from collections import OrderedDict 

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time

from treescan.policies import DiscreteReinforce, DiscretePPO, DiscretePPOGAE, DiscretePPOGAEMB
from treescan.networks.treeworld import SimpleConvNetwork, BetterConvNetwork, BetterConvNetwork2
from treescan.agents import Agent

from treescan.environments import TreeWorld




# Agent
agent_name = "Larry"
agents_folderpath = "C:/workspace/cs5180-project/experiments/ppo_training/agents"

# PPO
epsilon = 0.2
beta = 0.05
lambda_gae = 0.95
gamma = 0.99
alpha_logit = 0.001
alpha_value = 0.001
do_normalize_advantage = True

# Resume
resume_epoch = 1625

# Training
checkpoint_interval = 100
batch_size = 32
optimizer_epochs = 4
minibatch_size = 512

# World
step_limit = 499
use_fixed_map = False
enable_extra_channels = True
enable_extra_dist_channel = True
do_smooth_complete_reward = True
do_smooth_end_dist = True
do_gate_ending = True




# Environment
train_env = TreeWorld(render_mode=None, step_limit=step_limit, 
                      obs_as_tensor=True, use_fixed_map=use_fixed_map, 
                      enable_extra_channels=enable_extra_channels, enable_extra_dist_channel=enable_extra_dist_channel,
                      do_smooth_complete_reward=do_smooth_complete_reward, do_smooth_end_dist=do_smooth_end_dist,
                      do_gate_ending=do_gate_ending)

obs,_ = train_env.reset(seed=2025)
obs_channels = obs.shape[0]
action_list = [a for a in train_env.ACTIONS]
action_dim = len(action_list)

folderpath = f"{agents_folderpath}/{agent_name}"

# Networks
logit_network = SimpleConvNetwork(input_channels=obs_channels, output_width=action_dim, 
                 hidden_channels1=32, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=64, kernel2=3, stride2=1, padding2=1,
                 poolwidth = 4, poolheight = 4,
                 fc1_width = 128)
value_network = SimpleConvNetwork(input_channels=obs_channels, output_width=1, 
                 hidden_channels1=32, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=64, kernel2=3, stride2=1, padding2=1,
                 poolwidth = 4, poolheight = 4,
                 fc1_width = 128)

# Agent
policy = DiscretePPOGAEMB(logit_network, value_network, actions=action_list, 
                     logit_lr=alpha_logit, value_lr = alpha_value, entropy_bonus=beta, 
                     do_normalize_advantage=do_normalize_advantage)
friend = Agent(policy)


# Main
start = time.time()
try:
    friend.train(train_env,epochs=100_000, batch_size=batch_size, minibatch_size=minibatch_size, optimizer_epochs=optimizer_epochs, 
                 clip_epsilon=epsilon, gamma=gamma, lambda_gae=lambda_gae,
                 folderpath = folderpath, checkpoint_interval=checkpoint_interval, resume_epoch=resume_epoch,
                 start_seed=2025)

except KeyboardInterrupt:
    print("Interrupting...")

except Exception as e:
    print("Exception occurred")
    raise

finally:
    friend.save(f"{agents_folderpath}/{agent_name}")

print(f"Finished training after {time.time()-start:4.1f}s")