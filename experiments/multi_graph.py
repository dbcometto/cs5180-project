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
import os

# Config
do_plot_individual = False
main_folderpath = "C:\\workspace\\cs5180-project\\experiments"
datarequest = {
    "Fred": {
        "folder": "ppo_training",
        "test_names": ["final_test"],
        "use_mc": False,
        "batch_size": None,
        "plot_loss": True,
        "color": "maroon",
        "label": "MC PPO",
    },
    "Larry": {
        "folder": "ppo_training",
        "test_names": ["final_test"],
        "use_mc": True,
        "batch_size": None,
        "plot_loss": True,
        "color": "darkorange",
        "label": "GAE PPO",
    },
    "dqn_v5": {
        "folder": "dqn_test",
        "test_names": ["final_test"],
        "use_mc": False,
        "batch_size": None,
        "plot_loss": False,
        "color": "forestgreen",
        "label": "DQN (v5)",
    },
    "ddqn_v5": {
        "folder": "dqn_test",
        "test_names": ["final_test"],
        "use_mc": False,
        "batch_size": None,
        "plot_loss": False,
        "color": "darkturquoise",
        "label": "DDQN (v5)",
    },
}

# Helpers
def rolling_avg(arr, window=40):
    """Do a rolling average on the data"""
    result = np.convolve(arr, np.ones(window)/window, mode='same')

    half = window // 2
    result[:half] = np.nan
    result[-half:] = np.nan
    return result


# Main Plotting

main_fig,main_axs = plt.subplots(2,4,figsize=(21,8))
main_fig.suptitle(f"Agent Performance")

for friend_name, data in datarequest.items():
    folder = data["folder"]
    test_names = data["test_names"]
    use_mc = data["use_mc"]
    plot_loss = data["plot_loss"]
    color = data["color"]
    label = data["label"]

    # Make Agent
    agent_folderpath = os.path.join(main_folderpath,folder,"agents",friend_name)
    friend = Agent.load(agent_folderpath)


    # Make figure
    if do_plot_individual:
        fig,axs = plt.subplots(2,3,figsize=(20,7.5))
        fig.suptitle(f"Performance | Agent: {label} | Total training steps: {np.sum(friend.training_results["training_lengths"]):,d}")



    # Training Results
    training_results = np.array(friend.training_results["training_returns"]) if not use_mc else np.array(friend.training_results["training_mcreturns"])
    training_lengths = np.array(friend.training_results["training_lengths"])

    batch_size = data["batch_size"]
    if batch_size is not None:
        last_batch = training_results.shape[0] // batch_size
        training_results = training_results[:last_batch*batch_size].reshape(-1, batch_size).mean(axis=1)
        training_lengths = training_lengths[:last_batch*batch_size].reshape(-1, batch_size).mean(axis=1)

    main_axs[0,0].plot(training_results,alpha=0.1,color=color)
    main_axs[0,0].plot(rolling_avg(training_results),label=f"{label}",color=color)

    main_axs[1,0].plot(training_lengths,alpha=0.1,color=color)
    main_axs[1,0].plot(rolling_avg(training_lengths),label=f"{label}",color=color)

    if do_plot_individual:
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
        friend_test = friend.load_test(agent_folderpath,test_name)

        friend_test_return_avg = np.mean(np.array(friend_test["episode_returns"]),axis=0)
        friend_test_return_std = np.std(np.array(friend_test["episode_returns"]),axis=0)

        friend_test_length_avg = np.mean(np.array(friend_test["episode_lengths"]),axis=0)
        friend_test_length_std = np.std(np.array(friend_test["episode_lengths"]),axis=0)

        main_axs[0,1].plot(friend_test_return_avg,label=f"{label}",color=color)
        main_axs[0,1].fill_between(range(friend_test_return_avg.shape[0]),friend_test_return_avg-friend_test_return_std,friend_test_return_avg+friend_test_return_std,color=color,alpha=0.1)

        main_axs[1,1].plot(friend_test_length_avg,label=f"{label}",color=color)
        main_axs[1,1].fill_between(range(friend_test_length_avg.shape[0]),friend_test_length_avg-friend_test_length_std,friend_test_length_avg+friend_test_length_std,color=color,alpha=0.1)

        if do_plot_individual:
            axs[0,1].plot(friend_test_return_avg,label=f"{test_name}: Mean")
            axs[0,1].fill_between(range(friend_test_return_avg.shape[0]),friend_test_return_avg-friend_test_return_std,friend_test_return_avg+friend_test_return_std,alpha=0.2,label=f"{test_name}: Std")

            axs[1,1].plot(friend_test_length_avg,label=f"{test_name}: Mean")
            axs[1,1].fill_between(range(friend_test_length_avg.shape[0]),friend_test_length_avg-friend_test_length_std,friend_test_length_avg+friend_test_length_std,alpha=0.2,label=f"{test_name}: Std")

        # Percent Explored
        alpha = 0.1
        metric = 100*np.array([[d["percent_explored"] for d in j] for j in friend_test["last_env_info"]])
        avg = np.mean(np.array(metric),axis=0)
        std_dev = np.std(np.array(metric),axis=0)

        main_axs[0,2].plot(avg,label=f"{label}",color=color)
        main_axs[0,2].fill_between(range(avg.shape[0]),avg-std_dev,avg+std_dev,color=color,alpha=alpha)

        # Percent Complete
        metric = 100*np.array([[d["percent_complete"] for d in j] for j in friend_test["last_env_info"]])
        avg = np.mean(np.array(metric),axis=0)
        std_dev = np.std(np.array(metric),axis=0)

        main_axs[1,2].plot(avg,label=f"{label}",color=color)
        main_axs[1,2].fill_between(range(avg.shape[0]),avg-std_dev,avg+std_dev,color=color,alpha=alpha)

        # Final Distance
        metric = np.array([[d["dist_from_start"] for d in j] for j in friend_test["last_env_info"]])
        avg = np.mean(np.array(metric),axis=0)
        std_dev = np.std(np.array(metric),axis=0)

        main_axs[0,3].plot(avg,label=f"{label}",color=color)
        main_axs[0,3].fill_between(range(avg.shape[0]),avg-std_dev,avg+std_dev,color=color,alpha=alpha)

        # Num Scans
        metric = np.array([[d["count_scans"] for d in j] for j in friend_test["last_env_info"]])
        avg = np.mean(np.array(metric),axis=0)
        std_dev = np.std(np.array(metric),axis=0)

        main_axs[1,3].plot(avg,label=f"{label}",color=color)
        main_axs[1,3].fill_between(range(avg.shape[0]),avg-std_dev,avg+std_dev,color=color,alpha=alpha)
        


    if do_plot_individual:
        axs[0,1].set_xlabel("Episodes")
        axs[0,1].set_ylabel("Return")
        axs[0,1].set_title("Testing Results (Returns)")
        axs[0,1].grid(True)
        axs[0,1].legend(loc='lower right') #bbox_to_anchor=(0.85,-0.3))

        axs[1,1].set_xlabel("Episodes")
        axs[1,1].set_ylabel("Steps per Episode")
        axs[1,1].set_title("Testing Results (Lengths)")
        axs[1,1].grid(True)

        if plot_loss:
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
        else:
            axs[0,2].axis('off')
            axs[1,2].axis('off')

        fig.tight_layout()


    print(f"Agent: {label} | Total training steps: {np.sum(friend.training_results["training_lengths"])}")



main_axs[0,0].set_title("Training Results (Returns)")
main_axs[0,0].set_xlabel("Episode")
main_axs[0,0].set_ylabel("Return")
main_axs[0,0].grid(True)
main_axs[0,0].legend(loc='lower right')

main_axs[1,0].set_title("Training Results (Lengths)")
main_axs[1,0].set_xlabel("Episode")
main_axs[1,0].set_ylabel("Steps per Episode")
main_axs[1,0].grid(True)
# main_axs[0,1].legend(loc='lower right')
main_axs[1,0].legend(loc='upper right')

main_axs[0,1].set_title("Testing Results (Returns)")
main_axs[0,1].set_xlabel("Episodes")
main_axs[0,1].set_ylabel("Return")
main_axs[0,1].grid(True)
main_axs[0,1].legend(loc='lower left')

main_axs[1,1].set_title("Testing Results (Lengths)")
main_axs[1,1].set_xlabel("Episodes")
main_axs[1,1].set_ylabel("Steps per Episode")
main_axs[1,1].grid(True)
main_axs[1,1].legend(loc='upper left')

main_axs[0,2].set_title("Testing Results (Percent Explored)")
main_axs[0,2].set_xlabel("Episodes")
main_axs[0,2].set_ylabel("Percent")
main_axs[0,2].grid(True)
main_axs[0,2].legend(loc='upper left')

main_axs[1,2].set_title("Testing Results (Percent Complete)")
main_axs[1,2].set_xlabel("Episodes")
main_axs[1,2].set_ylabel("Percent")
main_axs[1,2].grid(True)
main_axs[1,2].legend(loc='upper left')

main_axs[0,3].set_title("Testing Results (Final Distance from Start)")
main_axs[0,3].set_xlabel("Episodes")
main_axs[0,3].set_ylabel("Distance")
main_axs[0,3].grid(True)
main_axs[0,3].legend(loc='upper left')

main_axs[1,3].set_title("Testing Results (Number of Scans)")
main_axs[1,3].set_xlabel("Count")
main_axs[1,3].set_ylabel("Percent")
main_axs[1,3].grid(True)
main_axs[1,3].legend(loc='upper left')
main_axs[1,3].set_ylim([-5,60])

main_fig.tight_layout()

plt.show()