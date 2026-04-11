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


friend_name = "MarvJr"
seed=2028

agents_folderpath = "C:/workspace/cs5180-project/experiments/ppo_training/agents"
friend_folderpath = f"{agents_folderpath}/{friend_name}"
# friend = Agent.load(friend_folderpath)
friend = Agent.load_from_checkpoint(friend_folderpath,26)



demo_env = TreeWorld(render_mode="human",step_limit=499, obs_as_tensor=True, use_fixed_map=False, enable_extra_channels=True)


# bob.generate_trajectory(demo_env,seed=2029)
torch.manual_seed(seed)
friend.generate_trajectory(demo_env,seed=seed)
print("Finished")
demo_env.hold_frame(duration=5000)
