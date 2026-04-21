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

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from version_configs import makeEnv


# Config
version = "v8"  # which trained agent to demo
friend_name = "ddqn_v8"
start_seed= 2102 # 2027 is classic, 2030 is good, 2031 is really good
demos = 100


# Setup
# agents_folderpath = "/Users/adamlewis/Desktop/Northeastern/Reinforcement Learning/Project/cs5180-project/experiments/dqn_test/agents"
agents_folderpath = "C:\\workspace\\cs5180-project\\experiments\\dqn_test\\agents"
friend_folderpath = f"{agents_folderpath}/{friend_name}"
friend = Agent.load(friend_folderpath)
# friend = Agent.load_from_checkpoint(friend_folderpath,10727)
demo_env = makeEnv(version)
demo_env.metadata["render_fps"]=32
demo_env.do_expand_rendering = True
demo_env.render_label = "DDQN V8"


# Loop
seed = start_seed
for i in tqdm(range(demos),desc="Demos",leave=True):
    torch.manual_seed(seed)
    friend.generate_trajectory(demo_env,seed=seed)
    # print("Finished")
    demo_env.hold_frame(duration=2000)
    seed += 1
