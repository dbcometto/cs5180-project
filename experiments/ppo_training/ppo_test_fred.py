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


# Config
friend_name = "Jeremy"
test_name = "test1765"

step_limit = 999
gamma = 0.99

do_resume = False
ckpt = 0




# Main
agents_folderpath = "C:/workspace/cs5180-project/experiments/ppo_training/agents"
friend_folderpath = f"{agents_folderpath}/{friend_name}"

if do_resume:
    friend = Agent.load_from_checkpoint(friend_folderpath, ckpt)
else:
    friend = Agent.load(friend_folderpath)


test_env = TreeWorld(render_mode=None, step_limit=step_limit, obs_as_tensor=True, use_fixed_map=False, enable_extra_channels=False)

friend_test = friend.test(test_env, episodes=100, runs=4, start_seed=2025, folderpath=friend_folderpath, test_name=test_name, gamma=gamma)

print(f"Finished testing after {time.time()-start:4.1f}s")