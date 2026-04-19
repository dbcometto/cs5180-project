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
is_test_discounted = False
base_test = "final_test_rtg"
datarequest = {
    "Fred": {
        "folder": "ppo_training",
        "test_names": [base_test],
        "use_mc": False,
        "batch_size": 32,
        "plot_loss": True,
        "color": "maroon",
        "label": "MC PPO",
    },
    "Jeremy": {
        "folder": "ppo_training",
        "test_names": [base_test],
        "use_mc": True,
        "batch_size": 32,
        "plot_loss": True,
        "color": "orangered",
        "label": "GAE PPO",
    },
    # "Rod": {
    #     "folder": "ppo_training",
    #     "test_names": [],
    #     "use_mc": True,
    #     "batch_size": 32,
    #     "plot_loss": True,
    #     "color": "darkorange",
    #     "label": "GAE V3",
    # },
    # "Todd": {
    #     "folder": "ppo_training",
    #     "test_names": [],
    #     "use_mc": True,
    #     "batch_size": 32,
    #     "plot_loss": True,
    #     "color": "limegreen",
    #     "label": "GAE V4",
    # },
    # "Ned": {
    #     "folder": "ppo_training",
    #     "test_names": [],
    #     "use_mc": True,
    #     "batch_size": 32,
    #     "plot_loss": True,
    #     "color": "dodgerblue",
    #     "label": "GAE V5",
    # },
    "dqn_v5": {
        "folder": "dqn_test",
        "test_names": [base_test],
        "use_mc": False,
        "batch_size": None,
        "plot_loss": False,
        "color": "forestgreen",
        "label": "DQN-V5",
    },
    "ddqn_v5": {
        "folder": "dqn_test",
        "test_names": [base_test],
        "use_mc": False,
        "batch_size": None,
        "plot_loss": False,
        "color": "turquoise",
        "label": "DDQN-V5",
    },
    "dqn_v8": {
        "folder": "dqn_test",
        "test_names": [base_test],
        "use_mc": False,
        "batch_size": None,
        "plot_loss": False,
        "color": "magenta",
        "label": "DQN-V8",
    },
    "ddqn_v8": {
        "folder": "dqn_test",
        "test_names": [base_test],
        "use_mc": False,
        "batch_size": None,
        "plot_loss": False,
        "color": "limegreen",
        "label": "DDQN-V8",
    },
}
alpha = 0.2
num_episodes = 10





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

agent_results = {
    "returns": {},
    "lengths": {},
    "percent_explored": {},
    "percent_complete": {},
    "dist_from_start": {},
    "count_scans": {}
}
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

    main_axs[0,0].plot(training_results,alpha=alpha,color=color,zorder=2.02)
    main_axs[0,0].plot(rolling_avg(training_results),label=f"{label}",color=color,zorder=2.03)

    main_axs[1,0].plot(training_lengths,alpha=alpha,color=color,zorder=2.02)
    main_axs[1,0].plot(rolling_avg(training_lengths),label=f"{label}",color=color,zorder=2.03)

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
        fullmean_return = np.median(friend_test["episode_returns"])
        agent_results["returns"][friend_name] = np.array(friend_test["episode_returns"])
        print(f"Returns: {label}, {fullmean_return}")

        friend_test_length_avg = np.mean(np.array(friend_test["episode_lengths"]),axis=0)
        friend_test_length_std = np.std(np.array(friend_test["episode_lengths"]),axis=0)
        fullmean_lengths = np.median(friend_test["episode_lengths"])
        agent_results["lengths"][friend_name] = np.array(friend_test["episode_lengths"])
        
        print(f"Lengths: {label}, {fullmean_lengths}")

        # main_axs[0,1].plot(friend_test_return_avg,label=f"{label}",color=color)
        # main_axs[0,1].plot([0, num_episodes],[fullmean_return,fullmean_return],color=color,linestyle='--')
        # main_axs[0,1].fill_between(range(friend_test_return_avg.shape[0]),friend_test_return_avg-friend_test_return_std,friend_test_return_avg+friend_test_return_std,color=color,alpha=alpha)

        # main_axs[1,1].plot(friend_test_length_avg,label=f"{label}",color=color)
        # main_axs[1,1].plot([0, num_episodes],[fullmean_lengths,fullmean_lengths],color=color,linestyle='--')
        # main_axs[1,1].fill_between(range(friend_test_length_avg.shape[0]),friend_test_length_avg-friend_test_length_std,friend_test_length_avg+friend_test_length_std,color=color,alpha=alpha)

        if do_plot_individual:
            axs[0,1].plot(friend_test_return_avg,label=f"{test_name}: Mean")
            axs[0,1].fill_between(range(friend_test_return_avg.shape[0]),friend_test_return_avg-friend_test_return_std,friend_test_return_avg+friend_test_return_std,alpha=0.2,label=f"{test_name}: Std")

            axs[1,1].plot(friend_test_length_avg,label=f"{test_name}: Mean")
            axs[1,1].fill_between(range(friend_test_length_avg.shape[0]),friend_test_length_avg-friend_test_length_std,friend_test_length_avg+friend_test_length_std,alpha=0.2,label=f"{test_name}: Std")

        # Percent Explored
        metric = 100*np.array([[d["percent_explored"] for d in j] for j in friend_test["last_env_info"]])
        avg = np.mean(np.array(metric),axis=0)
        std_dev = np.std(np.array(metric),axis=0)
        fullmean = np.median(metric)
        print(f"Percent Explored: {label}, {fullmean}")
        agent_results["percent_explored"][friend_name] = metric

        # main_axs[0,2].plot(avg,label=f"{label}",color=color)
        # main_axs[0,2].plot([0, num_episodes],[fullmean,fullmean],color=color,linestyle='--')
        # main_axs[0,2].fill_between(range(avg.shape[0]),avg-std_dev,avg+std_dev,color=color,alpha=alpha)

        # Percent Complete
        metric = 100*np.array([[d["percent_complete"] for d in j] for j in friend_test["last_env_info"]])
        avg = np.mean(np.array(metric),axis=0)
        std_dev = np.std(np.array(metric),axis=0)
        fullmean = np.median(metric)
        print(f"Percent Complete: {label}, {fullmean}")
        agent_results["percent_complete"][friend_name] = metric

        # main_axs[1,2].plot(avg,label=f"{label}",color=color)
        # main_axs[1,2].plot([0, num_episodes],[fullmean,fullmean],color=color,linestyle='--')
        # main_axs[1,2].fill_between(range(avg.shape[0]),avg-std_dev,avg+std_dev,color=color,alpha=alpha)

        # Final Distance
        metric = np.array([[d["dist_from_start"] for d in j] for j in friend_test["last_env_info"]])
        avg = np.mean(np.array(metric),axis=0)
        std_dev = np.std(np.array(metric),axis=0)
        fullmean = np.median(metric)
        print(f"Dist from Start: {label}, {fullmean}")
        agent_results["dist_from_start"][friend_name] = metric

        # main_axs[0,3].plot(avg,label=f"{label}",color=color)
        # main_axs[0,3].plot([0, num_episodes],[fullmean,fullmean],color=color,linestyle='--')
        # main_axs[0,3].fill_between(range(avg.shape[0]),avg-std_dev,avg+std_dev,color=color,alpha=alpha)

        # Num Scans
        metric = np.array([[d["count_scans"] for d in j] for j in friend_test["last_env_info"]])
        avg = np.mean(np.array(metric),axis=0)
        std_dev = np.std(np.array(metric),axis=0)
        fullmean = np.median(metric)
        print(f"Count Scans: {label}, {fullmean}")
        agent_results["count_scans"][friend_name] = metric

        # main_axs[1,3].plot(avg,label=f"{label}",color=color)
        # main_axs[1,3].plot([0, num_episodes],[fullmean,fullmean],color=color,linestyle='--')
        # main_axs[1,3].fill_between(range(avg.shape[0]),avg-std_dev,avg+std_dev,color=color,alpha=alpha)
        


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
# main_axs[0,0].set_xlabel("Episode")
# main_axs[0,0].set_xlabel("Batch")
main_axs[0,0].set_xlabel("Training Step (Batch or Episode)")
main_axs[0,0].set_ylabel("Discounted Total Return")
# main_axs[0,0].set_ylabel("Average Return")
main_axs[0,0].grid(True)
main_axs[0,0].legend(loc='upper right',fontsize="small")

main_axs[1,0].set_title("Training Results (Lengths)")
# main_axs[1,0].set_xlabel("Episode")
# main_axs[1,0].set_xlabel("Batch")
main_axs[1,0].set_xlabel("Training Step (Batch or Episode)")
main_axs[1,0].set_ylabel("Steps per Episode")
# main_axs[1,0].set_ylabel("Average Steps per Episode")
main_axs[1,0].grid(True)
# main_axs[0,1].legend(loc='lower right')
main_axs[1,0].legend(loc='upper right',fontsize="small")



vplot = main_axs[0,1].violinplot([v.flatten() for v in agent_results["returns"].values()],showmedians=True)
for body,color in zip(vplot['bodies'],[agent['color'] for agent in datarequest.values()]):
    body.set_facecolor(color)
    body.set_alpha(0.2)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = vplot[partname]
    vp.set_colors([agent['color'] for agent in datarequest.values()])
main_axs[0,1].set_title("Testing Results (Returns)")
main_axs[0,1].set_xticks(range(1, len(datarequest)+1))
main_axs[0,1].set_xticklabels([agent['label'] for agent in datarequest.values()])
main_axs[0,1].set_xlabel("Agent")
if not is_test_discounted:
    main_axs[0,1].set_ylabel("Total Accumulated Reward")
else:
    main_axs[0,1].set_ylabel("Discounted Total Return")
main_axs[0,1].grid(True)
# main_axs[0,1].legend(loc='lower left')
main_axs[0,1].set_ylim([-100,100])
main_axs[0,1].text(3.3,10,f"Min: {np.min(agent_results["returns"]["ddqn_v5"]):5.1f}",color=datarequest["ddqn_v5"]["color"])
main_axs[0,1].text(4.3,27,f"Min: {np.min(agent_results["returns"]["dqn_v8"]):5.1f}",color=datarequest["dqn_v8"]["color"])
main_axs[0,1].text(5.3,44,f"Min: {np.min(agent_results["returns"]["ddqn_v8"]):5.1f}",color=datarequest["ddqn_v8"]["color"])
main_axs[0,1].plot([0.8,6.2],[76.7,76.7],color="black",linestyle="--")
main_axs[0,1].text(0.9,79,"Perfect Score", color="black")


vplot = main_axs[1,1].violinplot([v.flatten() for v in agent_results["lengths"].values()],showmedians=True)
for body,color in zip(vplot['bodies'],[agent['color'] for agent in datarequest.values()]):
    body.set_facecolor(color)
    body.set_alpha(0.2)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = vplot[partname]
    vp.set_colors([agent['color'] for agent in datarequest.values()])
main_axs[1,1].set_title("Testing Results (Lengths)")
# main_axs[1,1].set_xlabel("Episode")
main_axs[1,1].set_xticks(range(1, len(datarequest)+1))
main_axs[1,1].set_xticklabels([agent['label'] for agent in datarequest.values()])
main_axs[1,1].set_xlabel("Agent")
main_axs[1,1].set_ylabel("Steps per Episode")
main_axs[1,1].grid(True)
# main_axs[1,1].legend(loc='upper left')


vplot = main_axs[0,2].violinplot([v.flatten() for v in agent_results["percent_explored"].values()],showmedians=True)
for body,color in zip(vplot['bodies'],[agent['color'] for agent in datarequest.values()]):
    body.set_facecolor(color)
    body.set_alpha(0.2)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = vplot[partname]
    vp.set_colors([agent['color'] for agent in datarequest.values()])
main_axs[0,2].set_title("Testing Results (Percent Explored)")
# main_axs[0,2].set_xlabel("Episode")
main_axs[0,2].set_xticks(range(1, len(datarequest)+1))
main_axs[0,2].set_xticklabels([agent['label'] for agent in datarequest.values()])
main_axs[0,2].set_xlabel("Agent")
main_axs[0,2].set_ylabel("Percent")
main_axs[0,2].grid(True)
# main_axs[0,2].legend(loc='upper left')


vplot = main_axs[1,2].violinplot([v.flatten() for v in agent_results["percent_complete"].values()],showmedians=True)
for body,color in zip(vplot['bodies'],[agent['color'] for agent in datarequest.values()]):
    body.set_facecolor(color)
    body.set_alpha(0.2)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = vplot[partname]
    vp.set_colors([agent['color'] for agent in datarequest.values()])
main_axs[1,2].set_title("Testing Results (Percent Tree Faces Scanned)")
# main_axs[1,2].set_xlabel("Episode")
main_axs[1,2].set_xticks(range(1, len(datarequest)+1))
main_axs[1,2].set_xticklabels([agent['label'] for agent in datarequest.values()])
main_axs[1,2].set_xlabel("Agent")
main_axs[1,2].set_ylabel("Percent")
main_axs[1,2].grid(True)
# main_axs[1,2].legend(loc='upper left')


vplot = main_axs[0,3].violinplot([v.flatten() for v in agent_results["dist_from_start"].values()],showmedians=True)
for body,color in zip(vplot['bodies'],[agent['color'] for agent in datarequest.values()]):
    body.set_facecolor(color)
    body.set_alpha(0.2)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = vplot[partname]
    vp.set_colors([agent['color'] for agent in datarequest.values()])
main_axs[0,3].set_title("Testing Results (Final Distance from Start)")
# main_axs[0,3].set_xlabel("Episode")
main_axs[0,3].set_xticks(range(1, len(datarequest)+1))
main_axs[0,3].set_xticklabels([agent['label'] for agent in datarequest.values()])
main_axs[0,3].set_xlabel("Agent")
main_axs[0,3].set_ylabel("Distance")
main_axs[0,3].grid(True)
# main_axs[0,3].legend(loc='upper left')


vplot = main_axs[1,3].violinplot([v.flatten() for v in agent_results["count_scans"].values()],showmedians=True)
for body,color in zip(vplot['bodies'],[agent['color'] for agent in datarequest.values()]):
    body.set_facecolor(color)
    body.set_alpha(0.2)
for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = vplot[partname]
    vp.set_colors([agent['color'] for agent in datarequest.values()])
main_axs[1,3].set_title("Testing Results (Number of Scans)")
# main_axs[1,3].set_xlabel("Episode")
main_axs[1,3].set_xticks(range(1, len(datarequest)+1))
main_axs[1,3].set_xticklabels([agent['label'] for agent in datarequest.values()])
main_axs[1,3].set_xlabel("Agent")
main_axs[1,3].set_ylabel("Count")
main_axs[1,3].grid(True)
# main_axs[1,3].legend(loc='upper left')
main_axs[1,3].set_ylim([-5,100])
main_axs[1,3].text(0.5,90,f"Max: {np.max(agent_results["count_scans"]["Fred"])}",color=datarequest["Fred"]["color"])
main_axs[1,3].text(3.5,90,f"Max: {np.max(agent_results["count_scans"]["ddqn_v5"])}",color=datarequest["ddqn_v5"]["color"])
main_axs[1,3].text(4.5,90,f"Max: {np.max(agent_results["count_scans"]["dqn_v8"])}",color=datarequest["dqn_v8"]["color"])
main_axs[1,3].text(5.5,90,f"Max: {np.max(agent_results["count_scans"]["ddqn_v8"])}",color=datarequest["ddqn_v8"]["color"])

main_fig.tight_layout()

plt.show()