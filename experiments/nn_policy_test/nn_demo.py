"""First messing with stuff"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from treescan.policies import RandomTabularPolicy, MCTabularFirstVisitEpsilonControl
from treescan.agents import Agent
from treescan.policies import DiscreteReinforce
from treescan.networks.gridworld import SimpleNetwork

from treescan.environments import GridWorld
import torch
from collections import OrderedDict


agents_folderpath = "C:/workspace/cs5180rl-main/cs5180-project/experiments/nn_policy_test/agents"
bill = Agent.load(f"{agents_folderpath}/bill")
# better_bob = Agent.load(f"{agents_folderpath}/better_bob")


demo_env = GridWorld(render_mode="human",flatten_obs=True,one_hot_obs=True)


# bob.generate_trajectory(demo_env,seed=2029)
bill.generate_trajectory(demo_env,seed=2026)
