# loads and tests dqn and ddqn agents on treeworld for a given version

import numpy as np

from treescan.agents import Agent
from version_configs import makeEnv, versionDescription


agentsFolderpath = "/Users/adamlewis/Desktop/Northeastern/Reinforcement Learning/Project/cs5180-project/experiments/dqn_test/agents"

# change this to swap which version to test
version = "v7"

print(f"\nTesting {version.upper()} — {versionDescription(version)}\n")

dqnAgent = Agent.load(f"{agentsFolderpath}/dqn_{version}")
ddqnAgent = Agent.load(f"{agentsFolderpath}/ddqn_{version}")

# greedy behavior at test time
dqnAgent.policy.epsilon = 0
ddqnAgent.policy.epsilon = 0

testEnv = makeEnv(version)

dqnResults = dqnAgent.test(testEnv, episodes=50, runs=4, start_seed=3000,
                           folderpath=f"{agentsFolderpath}/dqn_{version}", test_name="test0")

ddqnResults = ddqnAgent.test(testEnv, episodes=50, runs=4, start_seed=3000,
                             folderpath=f"{agentsFolderpath}/ddqn_{version}", test_name="test0")

print(f"\n=== {version.upper()} Results ===")
print(f"dqn  - avg return: {np.mean(dqnResults['episode_returns']):.3f}  avg length: {np.mean(dqnResults['episode_lengths']):.1f}")
print(f"ddqn - avg return: {np.mean(ddqnResults['episode_returns']):.3f}  avg length: {np.mean(ddqnResults['episode_lengths']):.1f}")
