"""Some testing with pytorch"""

import torch
from tqdm import tqdm
from collections import OrderedDict 

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time

from treescan.policies import DiscreteReinforce, DiscretePPO, DiscreteBatchReinforce, DiscreteBatchReinforceBaseline, DiscreteAdvantageActorCritic
from treescan.networks.treeworld import SimpleConvNetwork
from treescan.agents import Agent

from treescan.environments import TreeWorld



        


train_env = TreeWorld(render_mode=None,step_limit=999,obs_as_tensor=True)
obs,_ = train_env.reset(seed=2025)
# print(obs)

obs_channels = obs.shape[0]
print(obs_channels)
action_list = [a for a in train_env.ACTIONS]
action_dim = len(action_list)



agent_name = "Bob"

start = time.time()

# PPO
logit_network = SimpleConvNetwork(input_channels=obs_channels,output_width=action_dim)
value_network = SimpleConvNetwork(input_channels=obs_channels,output_width=1)
policy = DiscretePPO(logit_network,value_network,actions=action_list,logit_lr=0.001,value_lr = 0.001)
friend = Agent(policy)
friend.train(train_env,epochs=30,batch_size=3,optimizer_epochs=5,clip_epsilon=0.2,start_seed=2025)





agents_folderpath = "C:/workspace/cs5180-project/experiments/tree_test/agents"
friend.save(f"{agents_folderpath}/{agent_name}")

print(f"Finished training after {time.time()-start:4.1f}s")