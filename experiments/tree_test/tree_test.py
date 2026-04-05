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

friend_name = "Bob"
test_name = "test0"

agents_folderpath = "C:/workspace/cs5180-project/experiments/tree_test/agents"
friend_folderpath = f"{agents_folderpath}/{friend_name}"
friend = Agent.load(friend_folderpath)


test_env = TreeWorld(render_mode=None,step_limit=999,obs_as_tensor=True)

friend_test = friend.test(test_env,episodes=50,runs=4,start_seed=2025, folderpath=friend_folderpath, test_name=test_name)