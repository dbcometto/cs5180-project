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
from tqdm import tqdm
import os

start = time.time()




# Config
episodes = 100
runs = 10
start_seed = 2025
test_name = "final_test_rtg_1000"
folderpath = "C:\\workspace\\cs5180-project\\experiments"


# World - Standard Reward for Comparison
step_limit = 999
do_extra_info = True
use_fixed_map = False
enable_extra_channels = True
enable_extra_dist_channel = True
do_smooth_complete_reward = False
do_smooth_end_dist = False
do_gate_ending = False
do_reward_tree_complete = False
gamma = 1.0 # Compare with reward-to-go

datarequest = {
    "Fred": {
        "folder": "ppo_training",
        "enable_extra_channels": False,
        "enable_extra_dist_channel": False,
        "checkpoint": None,
    },
    "Jeremy": {
        "folder": "ppo_training",
        "enable_extra_channels": False,
        "enable_extra_dist_channel": False,
        "checkpoint": None,
    },
    "dqn_v5": {
        "folder": "dqn_test",
        "enable_extra_channels": True,
        "enable_extra_dist_channel": False,
        "checkpoint": None,
    },
    "ddqn_v5": {
        "folder": "dqn_test",
        "enable_extra_channels": True,
        "enable_extra_dist_channel": False,
        "checkpoint": None,
    },
    "dqn_v8": {
        "folder": "dqn_test",
        "enable_extra_channels": True,
        "enable_extra_dist_channel": False,
        "checkpoint": None,
    },
    "ddqn_v8": {
        "folder": "dqn_test",
        "enable_extra_channels": True,
        "enable_extra_dist_channel": False,
        "checkpoint": None,
    },
}


# Main loop
pbar = tqdm(datarequest.items(),desc="Testing Agents",leave=True)
for agent_name, data in pbar:
    pbar.set_postfix({"Agent":agent_name})
    # Data
    enable_extra_channels = data["enable_extra_channels"]
    enable_extra_dist_channel = data["enable_extra_dist_channel"]
    agent_folder = data["folder"]
    checkpoint = data["checkpoint"]

    # Make agent
    agent_path = os.path.join(folderpath,agent_folder,"agents",agent_name)
    if checkpoint is None:
        friend = Agent.load(agent_path)
    else:
        friend = Agent.load_from_checkpoint(agent_path, checkpoint)

    # Env
    test_env = TreeWorld(render_mode=None, obs_as_tensor=True, do_extra_info=do_extra_info,
                         step_limit=step_limit,
                         use_fixed_map=use_fixed_map,
                         enable_extra_channels=enable_extra_channels,         # Obs Based on agent
                         enable_extra_dist_channel=enable_extra_dist_channel,
                         do_smooth_complete_reward=do_smooth_complete_reward, # Standard reward and ending
                         do_smooth_end_dist=do_smooth_end_dist,             
                         do_gate_ending=do_gate_ending,
                         do_reward_tree_complete=do_reward_tree_complete,
                        )

    friend_test = friend.test(test_env,episodes=episodes,runs=runs,start_seed=start_seed, folderpath=agent_path, test_name=test_name, gamma=gamma)

print(f"Finished testing after {time.time()-start:4.1f}s")