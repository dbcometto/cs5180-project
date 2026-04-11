"""First messing with stuff"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict

from treescan.policies import RandomTabularPolicy, MCTabularFirstVisitEpsilonControl
from treescan.agents import Agent
from treescan.networks.treeworld import SimpleConvNetwork
from treescan.policies import DiscreteReinforce

from treescan.environments import TreeWorld
import time
start = time.time()

friend_name = "MarvJr"
test_name = "test26"
gamma = 0.99

agents_folderpath = "C:/workspace/cs5180-project/experiments/ppo_training/agents"
friend_folderpath = f"{agents_folderpath}/{friend_name}"
# friend = Agent.load(friend_folderpath)
friend = Agent.load_from_checkpoint(friend_folderpath, 26)


test_env = TreeWorld(render_mode=None,step_limit=999,obs_as_tensor=True, use_fixed_map=False, enable_extra_channels=True, do_smooth_complete_reward=True, do_smooth_end_dist=True)

friend_test = friend.test(test_env,episodes=100,runs=8,start_seed=2025, folderpath=friend_folderpath, test_name=test_name, gamma=gamma)

print(f"Finished testing after {time.time()-start:4.1f}s")