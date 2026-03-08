"""First messing with stuff"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict

from treescan.policies import RandomTabularPolicy, MCTabularFirstVisitEpsilonControl
from treescan.agents import Agent
from treescan.networks.gridworld import SuperSimpleLogitNetwork
from treescan.policies import DiscreteReinforce

from treescan.environments import GridWorld

friend_name = "perry3"
test_name = "test0"

agents_folderpath = "C:/workspace/cs5180rl-main/cs5180-project/experiments/nn_policy_test/agents"
friend_folderpath = f"{agents_folderpath}/{friend_name}"
friend = Agent.load(friend_folderpath)


test_env = GridWorld(render_mode=None,flatten_obs=True,one_hot_obs=True)

friend_test = friend.test(test_env,episodes=50,runs=4,start_seed=2025, folderpath=friend_folderpath, test_name=test_name)