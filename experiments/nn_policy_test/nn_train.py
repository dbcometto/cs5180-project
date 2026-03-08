"""Some testing with pytorch"""

import torch
from tqdm import tqdm
from collections import OrderedDict 

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time

from treescan.policies import DiscreteReinforce, DiscretePPO, DiscreteBatchReinforce, DiscreteBatchReinforceBaseline, DiscreteAdvantageActorCritic
from treescan.networks.gridworld import SimplePolicyNetwork, SimpleValueNetwork, SuperSimpleLogitNetwork, SuperSimpleValueNetwork
from treescan.agents import Agent

from treescan.environments import GridWorld



        


train_env = GridWorld(render_mode=None,step_limit=999,fixed_goal=True,flatten_obs=True,one_hot_obs=True)
obs,_ = train_env.reset(seed=2025)
# print(obs)

obs_dim = len(obs)
# print(obs_dim)
action_list = [a for a in train_env.ACTIONS]
action_dim = len(action_list)



agent_name = "perry3"

start = time.time()

# REINFORCE (single)
# network = SimplePolicyNetwork(input_width=obs_dim,embedded_width=16,hidden_width=32,output_width=action_dim)
# policy = DiscreteReinforce(network,actions=action_list,obs_dim=obs_dim,lr=0.01,weight_decay=0)
# friend = Agent(policy)
# friend.train(train_env,episodes=10)

# REINFORCE (batch)
# network = SuperSimpleLogitNetwork(input_width=obs_dim,hidden_width=8,output_width=action_dim)
# policy = DiscreteBatchReinforceBaseline(network,actions=action_list,obs_dim=obs_dim,lr=0.01,weight_decay=0)
# friend = Agent(policy)
# friend.train(train_env,epochs=1000,batch_size=1,entropy=0.0)

# A2C
# logit_network = SuperSimpleLogitNetwork(input_width=obs_dim,hidden_width=8,output_width=action_dim)
# value_network = SuperSimpleValueNetwork(input_width=obs_dim,hidden_width=8,output_width=1)
# policy = DiscreteAdvantageActorCritic(logit_network,value_network,actions=action_list,obs_dim=obs_dim,logit_lr=0.001,value_lr = 0.001)
# friend = Agent(policy)
# friend.train(train_env,epochs=400,batch_size=1,start_seed=2025)


# PPO
logit_network = SuperSimpleLogitNetwork(input_width=obs_dim,hidden_width=8,output_width=action_dim)
value_network = SuperSimpleValueNetwork(input_width=obs_dim,hidden_width=8,output_width=1)
policy = DiscretePPO(logit_network,value_network,actions=action_list,obs_dim=obs_dim,logit_lr=0.001,value_lr = 0.001)
friend = Agent(policy)
friend.train(train_env,epochs=300,batch_size=3,optimizer_epochs=5,clip_epsilon=0.2,start_seed=2025)





agents_folderpath = "C:/workspace/cs5180rl-main/cs5180-project/experiments/nn_policy_test/agents"
friend.save(f"{agents_folderpath}/{agent_name}")

print(f"Finished training after {time.time()-start:4.1f}s")