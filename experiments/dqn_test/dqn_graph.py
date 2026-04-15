# plots dqn and ddqn training learning curves for a given version

import numpy as np
import matplotlib.pyplot as plt

from treescan.agents import Agent
from version_configs import versionDescription


agentsFolderpath = "/Users/adamlewis/Desktop/Northeastern/Reinforcement Learning/Project/cs5180-project/experiments/dqn_test/agents"

# change this to swap which training run to plot
version = "v8"


def rolling_stats(data, window=50):
    """compute rolling mean and std over a window"""
    means = np.empty(len(data))
    stds = np.empty(len(data))
    for i in range(len(data)):
        start = max(0, i - window + 1)
        chunk = data[start:i+1]
        means[i] = np.mean(chunk)
        stds[i] = np.std(chunk)
    return means, stds


def hyperparam_text(policy):
    """build a readable hyperparameter summary from a loaded policy"""
    lr = policy.optimizer.param_groups[0]["lr"]
    return (
        f"lr = {lr:g}\n"
        f"gamma = {policy.gamma}\n"
        f"buffer = {policy.bufferSize}\n"
        f"batch = {policy.batchSize}\n"
        f"eps decay = {policy.epsilonDecaySteps}\n"
        f"target update = {policy.targetUpdateFreq}"
    )


# load agents
dqnAgent = Agent.load(f"{agentsFolderpath}/dqn_{version}")
ddqnAgent = Agent.load(f"{agentsFolderpath}/ddqn_{version}")

dqnReturns = np.array(dqnAgent.training_results["training_returns"])
ddqnReturns = np.array(ddqnAgent.training_results["training_returns"])

window = 50

dqnMean, dqnStd = rolling_stats(dqnReturns, window)
ddqnMean, ddqnStd = rolling_stats(ddqnReturns, window)

fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
fig.suptitle(f"DQN vs Double DQN Training Performance — {version.upper()}",
             fontsize=14, fontweight="bold", y=0.98)
ax.set_title(versionDescription(version), fontsize=11, style="italic", pad=10)

# dqn
episodes = np.arange(len(dqnMean))
ax.plot(episodes, dqnMean, color="red", alpha=0.85, label="DQN")
ax.fill_between(episodes, dqnMean - dqnStd, dqnMean + dqnStd, color="red", alpha=0.2)

# ddqn
episodes = np.arange(len(ddqnMean))
ax.plot(episodes, ddqnMean, color="blue", alpha=0.85, label="Double DQN")
ax.fill_between(episodes, ddqnMean - ddqnStd, ddqnMean + ddqnStd, color="blue", alpha=0.2)

ax.set_xlabel("Training Episode")
ax.set_ylabel("Discounted Return (rolling mean ± std)")
ax.legend(loc="upper left", fontsize=11)
ax.grid(True, alpha=0.4)

# hyperparameter box in bottom-right
hpText = hyperparam_text(dqnAgent.policy)
ax.text(0.98, 0.02, hpText,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))

plt.tight_layout()
plt.show()
