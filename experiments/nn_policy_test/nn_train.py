"""Some testing with pytorch"""

import torch
from tqdm import tqdm
from collections import OrderedDict 

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time

from treescan.policies import DiscreteReinforce, DiscretePPO
from treescan.networks.gridworld import SimplePolicyNetwork, SimpleValueNetwork
from treescan.agents import Agent

from treescan.environments import GridWorld



        


train_env = GridWorld(render_mode=None,step_limit=999,fixed_goal=True,flatten_obs=True,one_hot_obs=True)
obs,_ = train_env.reset(seed=2025)
# print(obs)

obs_dim = len(obs)
action_list = [a for a in train_env.ACTIONS]
action_dim = len(action_list)

# network = SimplePolicyNetwork(input_width=obs_dim,embedded_width=16,hidden_width=32,output_width=action_dim)
# policy = DiscreteReinforce(network,actions=action_list,obs_dim=obs_dim,lr=0.01,weight_decay=0)

policy_network = SimplePolicyNetwork(input_width=obs_dim,embedded_width=16,hidden_width=32,output_width=action_dim)
value_network = SimpleValueNetwork(input_width=obs_dim,embedded_width=16,hidden_width=32)
policy = DiscretePPO(policy_network,value_network,actions=action_list,obs_dim=obs_dim,policy_lr=0.001,value_lr = 0.001)


friend = Agent(policy)

start = time.time()
# friend.train(train_env,episodes=10)
friend.train(train_env, ppo_epochs=10, epochs=10, episodes=10)

agents_folderpath = "C:/workspace/cs5180rl-main/cs5180-project/experiments/nn_policy_test/agents"
friend.save(f"{agents_folderpath}/marv")

print(f"Finished training after {time.time()-start:4.1f}s")