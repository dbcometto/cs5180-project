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

friend_name = "MarvJr"
test_names = ["test0","test26"]

agents_folderpath = "C:/workspace/cs5180-project/experiments/ppo_training/agents"
friend_folderpath = f"{agents_folderpath}/{friend_name}"
friend = Agent.load(friend_folderpath)
# friend = Agent.load_from_checkpoint(friend_folderpath,4721)

for test_name in test_names:
    friend_test = friend.load_test(friend_folderpath,test_name)

    friend_test_return_avg = np.mean(np.array(friend_test["episode_returns"]),axis=0)
    friend_test_return_std = np.std(np.array(friend_test["episode_returns"]),axis=0)

    friend_test_length_avg = np.mean(np.array(friend_test["episode_lengths"]),axis=0)
    friend_test_length_std = np.std(np.array(friend_test["episode_lengths"]),axis=0)


    fig,axs = plt.subplots(2,2,figsize=(16,8))
    fig.suptitle(f"Test: {test_name}")

    # axs[0,0].plot(np.cumsum(friend.training_results["training_returns"]),label="friend")
    axs[0,0].plot(friend.training_results["training_returns"],label="friend")
    axs[0,0].set_xlabel("Episodes")
    # axs[0,0].set_ylabel("Total Return")
    axs[0,0].set_ylabel("Return")
    axs[0,0].set_title("Friend Training Results (Returns)")
    axs[0,0].grid(True)
    axs[0,0].legend()

    # axs[1,0].plot(np.cumsum(friend.training_results["training_lengths"]),label="friend")
    axs[1,0].plot(friend.training_results["training_lengths"],label="friend")
    axs[1,0].set_xlabel("Episodes")
    axs[1,0].set_ylabel("Steps")
    axs[1,0].set_title("Friend Training Results (Length)")
    axs[1,0].grid(True)
    axs[1,0].legend()

    # axs[0,1].plot(np.cumsum(friend_test_return_avg),label="friend")
    axs[0,1].plot(friend_test_return_avg,label="Mean")
    axs[0,1].fill_between(range(friend_test_return_avg.shape[0]),friend_test_return_avg-friend_test_return_std,friend_test_return_avg+friend_test_return_std,alpha=0.2,label="Std")
    axs[0,1].set_xlabel("Episodes")
    # axs[0,1].set_ylabel("Total Return")
    axs[0,1].set_ylabel("Return")
    axs[0,1].set_title("Testing Results")
    axs[0,1].grid(True)
    axs[0,1].legend()

    # axs[0,1].plot(np.cumsum(friend_test_return_avg),label="friend")
    axs[1,1].plot(friend_test_length_avg,label="Mean")
    axs[1,1].fill_between(range(friend_test_length_avg.shape[0]),friend_test_length_avg-friend_test_length_std,friend_test_length_avg+friend_test_length_std,alpha=0.2,label="Std")
    axs[1,1].set_xlabel("Episodes")
    # axs[0,1].set_ylabel("Total Return")
    axs[1,1].set_ylabel("Length")
    axs[1,1].set_title("Testing Results")
    axs[1,1].grid(True)
    axs[1,1].legend()





plt.show()