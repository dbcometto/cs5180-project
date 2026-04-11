"""Some testing with pytorch"""

import torch
from tqdm import tqdm
from collections import OrderedDict 

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time

from treescan.policies import DiscreteReinforce, DiscretePPO, DiscretePPOGAE
from treescan.networks.treeworld import SimpleConvNetwork, BetterConvNetwork
from treescan.agents import Agent

from treescan.environments import TreeWorld



        


train_env = TreeWorld(render_mode=None, step_limit=999, obs_as_tensor=True, use_fixed_map=False, enable_extra_channels=True)
obs,_ = train_env.reset(seed=2025)
# print(obs)

obs_channels = obs.shape[0]
# print(obs_channels)
action_list = [a for a in train_env.ACTIONS]
action_dim = len(action_list)

start = time.time()







# Agent
agent_name = "Marv"
agents_folderpath = "C:/workspace/cs5180-project/experiments/ppo_training/agents"


# PPO
epsilon = 0.2
beta = 0.01
lambda_gae = 0.95
gamma = 0.99
alpha_logit = 0.0005
alpha_value = 0.0005

# Training
resume_epoch = None
checkpoint_interval = 250
batch_size = 64
optimizer_epochs = 8



# Setup
folderpath = f"{agents_folderpath}/{agent_name}"

logit_network = BetterConvNetwork(input_channels=obs_channels, output_width=action_dim, 
                 hidden_channels1=64, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=96, kernel2=3, stride2=1, padding2=1,
                 hidden_channels3=128, kernel3=3, stride3=1, padding3=1,
                 poolwidth = 8, poolheight = 8,
                 fc1_width = 128)
value_network = BetterConvNetwork(input_channels=obs_channels, output_width=1, 
                 hidden_channels1=64, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=96, kernel2=3, stride2=1, padding2=1,
                 hidden_channels3=128, kernel3=3, stride3=1, padding3=1,
                 poolwidth = 8, poolheight = 8,
                 fc1_width = 128)

policy = DiscretePPOGAE(logit_network, value_network, actions=action_list, 
                     logit_lr=alpha_logit, value_lr = alpha_value, entropy_bonus=beta, 
                     do_normalize_advantage=False)
friend = Agent(policy)


# Main
try:
    friend.train(train_env,epochs=100_000, batch_size=batch_size, optimizer_epochs=optimizer_epochs, 
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