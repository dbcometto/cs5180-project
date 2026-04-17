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
from tqdm import tqdm


# Config
friend_name = "Todd"
start_seed= 2101 # 2027 is classic, 2030 is good, 2031 is really good
demos = 1000


# Setup
agents_folderpath = "C:/workspace/cs5180-project/experiments/ppo_training/agents"
friend_folderpath = f"{agents_folderpath}/{friend_name}"
friend = Agent.load(friend_folderpath)
# friend = Agent.load_from_checkpoint(friend_folderpath,10727)
demo_env = TreeWorld(render_mode="human",step_limit=199,obs_as_tensor=True, use_fixed_map=False, do_expand_rendering=True, render_label=friend_name)
demo_env.metadata["render_fps"]=8


# Loop
seed = start_seed
for i in tqdm(range(demos),desc="Demos",leave=True):
    torch.manual_seed(seed)
    friend.generate_trajectory(demo_env,seed=seed)
    # print("Finished")
    demo_env.hold_frame(duration=2000)
    seed += 1
