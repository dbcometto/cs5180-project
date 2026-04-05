"""First messing with stuff"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from treescan.policies import RandomTabularPolicy, MCTabularFirstVisitEpsilonControl
from treescan.agents import Agent
from treescan.policies import DiscreteReinforce
from treescan.networks.treeworld import SimpleConvNetwork

from treescan.environments import TreeWorld
import torch
from collections import OrderedDict


friend_name = "Bob"

agents_folderpath = "C:/workspace/cs5180-project/experiments/tree_test/agents"
friend_folderpath = f"{agents_folderpath}/{friend_name}"
friend = Agent.load(friend_folderpath)

demo_env = TreeWorld(render_mode=None,step_limit=999,obs_as_tensor=True)


# bob.generate_trajectory(demo_env,seed=2029)
friend.generate_trajectory(demo_env,seed=2026)
