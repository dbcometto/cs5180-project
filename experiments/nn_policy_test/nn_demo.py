"""First messing with stuff"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from treescan.policies import RandomTabularPolicy, MCTabularFirstVisitEpsilonControl
from treescan.agents import Agent
from treescan.policies import DiscreteReinforce
from treescan.networks.gridworld import SuperSimpleLogitNetwork

from treescan.environments import GridWorld
import torch
from collections import OrderedDict


friend_name = "perry2"

agents_folderpath = "C:/workspace/cs5180rl-main/cs5180-project/experiments/nn_policy_test/agents"
friend_folderpath = f"{agents_folderpath}/{friend_name}"
friend = Agent.load(friend_folderpath)

demo_env = GridWorld(render_mode="human",flatten_obs=True,one_hot_obs=True)


# bob.generate_trajectory(demo_env,seed=2029)
friend.generate_trajectory(demo_env,seed=2026)
