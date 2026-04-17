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


# Config
datarequest = {
    "Fred": {
        "test_names": ["test0", 
                    #    "test10727", 
                    #    "test11985", 
                    #    "test12226",
                    #    "test13450",
                    #    "test14102",
                    #    "test16493",
                       "test17667",],
        "use_mc": False,
        "batch_size": 32,
        "label": "MC PPO"
    },
    "Jeremy": {
        "test_names": ["test0", 
                    #    "test472",
                       "test1054",],
        "use_mc": True,
        "batch_size": 32,
        "label": "GAE PPO"
    },
    "Rod": {
        "test_names": [],
            # "test0", 
            #            "test472",],
        "use_mc": True,
        "batch_size": 32,
        "label": "GAE PPO with Shorter Truncation"
    },
    "Todd": {
        "test_names": [],
            # "test0", 
            #            "test472",],
        "use_mc": True,
        "batch_size": 32,
        "label": "GAE PPO with Shorter Truncation and Smoothed Reward"
    },
    "Ned": {
        "test_names": ["test0", 
                       "test1356",],
        "use_mc": True,
        "batch_size": 32,
        "label": "GAE PPO with Smoothed Reward, Shorter Truncation, and Increased Entropy"
    },
    # "Larry": {
    #     "test_names": ["test0", 
    #                 #    "test642", 
    #                 #    "test1457", 
    #                 #    "test1625",
    #                    "test2510",],
    #     "use_mc": True,
    #     "batch_size": 32
    # },
    # "Paul": {
    #     "test_names": ["test0", 
    #                    "test15406",],
    #     "use_mc": True,
    #     "batch_size": 32
    # },
    # "James": {
    #     "test_names": ["test0", 
    #                    "test8018",],
    #     "use_mc": False,
    #     "batch_size": 32
    # },
}
agents_folderpath = "C:/workspace/cs5180-project/experiments/ppo_training/agents"


# Helpers
def rolling_avg(arr, window=40):
    """Do a rolling average on the data"""
    result = np.convolve(arr, np.ones(window)/window, mode='same')

    half = window // 2
    result[:half] = np.nan
    result[-half:] = np.nan
    return result


# Main Plotting
for friend_name, data in datarequest.items():
    test_names = data["test_names"]
    use_mc = data["use_mc"]

    friend_folderpath = f"{agents_folderpath}/{friend_name}"
    friend = Agent.load(friend_folderpath)
    # friend = Agent.load_from_checkpoint(friend_folderpath,4721)

    fig,axs = plt.subplots(2,3,figsize=(20,7.5))

    try:
        label = data["label"]
    except:
        label = None

    if label is not None:
        fig.suptitle(f"Performance | {label}")
    else:
        fig.suptitle(f"Performance | Agent: {friend_name}")



    # Training Results
    training_results = np.array(friend.training_results["training_returns"]) if not use_mc else np.array(friend.training_results["training_mcreturns"])
    training_lengths = np.array(friend.training_results["training_lengths"])

    batch_size = data["batch_size"]
    if batch_size is not None:
        last_batch = training_results.shape[0] // batch_size
        training_results = training_results[:last_batch*batch_size].reshape(-1, batch_size).mean(axis=1)
        training_lengths = training_lengths[:last_batch*batch_size].reshape(-1, batch_size).mean(axis=1)

    axs[0,0].plot(training_results,label="Returns",alpha=0.4)
    axs[0,0].plot(rolling_avg(training_results),label="Rolling Avg")
    axs[0,0].set_title("Training Results (Returns)")
    if batch_size is not None:
        axs[0,0].set_xlabel("Batch")
        axs[0,0].set_ylabel("Average Return")
    else:
        axs[0,0].set_xlabel("Episode")
        axs[0,0].set_ylabel("Return")
    axs[0,0].grid(True)
    axs[0,0].legend()

    axs[1,0].plot(training_lengths,label="Lengths",alpha=0.4)
    axs[1,0].plot(rolling_avg(training_lengths),label="Rolling Avg")
    axs[1,0].set_title("Training Results (Length)")
    if batch_size is not None:
        axs[1,0].set_xlabel("Batch")
        axs[1,0].set_ylabel("Average Steps per Episode")
    else:
        axs[1,0].set_xlabel("Episode")
        axs[1,0].set_ylabel("Steps per Episode")
    axs[1,0].grid(True)
    axs[1,0].legend()

    for test_name in test_names:
        friend_test = friend.load_test(friend_folderpath,test_name)

        friend_test_return_avg = np.mean(np.array(friend_test["episode_returns"]),axis=0)
        friend_test_return_std = np.std(np.array(friend_test["episode_returns"]),axis=0)

        friend_test_length_avg = np.mean(np.array(friend_test["episode_lengths"]),axis=0)
        friend_test_length_std = np.std(np.array(friend_test["episode_lengths"]),axis=0)

        axs[0,1].plot(friend_test_return_avg,label=f"{test_name}: Mean")
        axs[0,1].fill_between(range(friend_test_return_avg.shape[0]),friend_test_return_avg-friend_test_return_std,friend_test_return_avg+friend_test_return_std,alpha=0.2,label=f"{test_name}: Std")
        
        axs[1,1].plot(friend_test_length_avg,label=f"{test_name}: Mean")
        axs[1,1].fill_between(range(friend_test_length_avg.shape[0]),friend_test_length_avg-friend_test_length_std,friend_test_length_avg+friend_test_length_std,alpha=0.2,label=f"{test_name}: Std")
        

    axs[0,1].set_xlabel("Episodes")
    axs[0,1].set_ylabel("Return")
    axs[0,1].set_title("Testing Results (Returns)")
    axs[0,1].grid(True)
    axs[0,1].legend(loc='lower right') #bbox_to_anchor=(0.85,-0.3))

    axs[1,1].set_xlabel("Episodes")
    axs[1,1].set_ylabel("Steps per Episode")
    axs[1,1].set_title("Testing Results (Lengths)")
    axs[1,1].grid(True)


    # fig.subplots_adjust(right=0.75)
    # fig.tight_layout()


    # fig,axs = plt.subplots(2,1,figsize=(8,8))
    # fig.suptitle(f"Losses | Agent: {friend_name}")

    actor_loss = [d[0] for d in friend.training_results["training_losses"]]
    critic_loss = [d[1] for d in friend.training_results["training_losses"]]

    axs[0,2].plot(actor_loss,label="Loss",alpha=0.4)
    axs[0,2].plot(rolling_avg(actor_loss),label="Rolling Avg")
    axs[0,2].set_xlabel("Batch")
    axs[0,2].set_ylabel("Loss")
    axs[0,2].set_title("Actor Loss")
    axs[0,2].grid(True)
    axs[0,2].legend()

    axs[1,2].plot(critic_loss,label="Loss",alpha=0.4)
    axs[1,2].plot(rolling_avg(critic_loss),label="Rolling Avg")
    axs[1,2].set_xlabel("Batch")
    axs[1,2].set_ylabel("Loss")
    axs[1,2].set_title("Critic Loss")
    axs[1,2].grid(True)
    axs[1,2].legend()

    fig.tight_layout()

    print(f"Agent: {friend_name} | Total training steps: {np.sum(friend.training_results["training_lengths"])}")


plt.show()