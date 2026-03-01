"""First messing with stuff"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict

from treescan.policies import RandomTabularPolicy, MCTabularFirstVisitEpsilonControl
from treescan.agents import Agent
from treescan.networks.gridworld import SimpleNetwork
from treescan.policies import DiscreteReinforce

from treescan.environments import GridWorld



agents_folderpath = "C:/workspace/cs5180rl-main/cs5180-project/experiments/nn_policy_test/agents"
bill = Agent.load(f"{agents_folderpath}/bill2")


test_env = GridWorld(render_mode=None,flatten_obs=True,one_hot_obs=True)

bill_test = bill.test(test_env,episodes=100,runs=2,start_seed=2025)

bill_test_return_avg = np.mean(bill_test["episode_returns"],axis=0)


fig,axs = plt.subplots(2,2,figsize=(16,8))
# axs[0,0].plot(np.cumsum(bill.training_results["episode_returns"]),label="bill")
axs[0,0].plot(bill.training_results["episode_returns"],label="bill")
axs[0,0].set_xlabel("Episodes")
# axs[0,0].set_ylabel("Total Return")
axs[0,0].set_ylabel("Return")
axs[0,0].set_title("Bob Training Results")
axs[0,0].grid(True)
axs[0,0].legend()

# axs[0,1].plot(np.cumsum(bill_test_return_avg),label="bill")
axs[0,1].plot(bill_test_return_avg,label="bill")
axs[0,1].set_xlabel("Episodes")
# axs[0,1].set_ylabel("Total Return")
axs[0,1].set_ylabel("Return")
axs[0,1].set_title("Testing Results")
axs[0,1].grid(True)
axs[0,1].legend()



plt.show()