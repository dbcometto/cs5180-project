# loads and tests dqn and ddqn agents on gridworld

import numpy as np

from treescan.environments import GridWorld
from treescan.agents import Agent


agentsFolderpath = "/Users/adamlewis/Desktop/Northeastern/Reinforcement Learning/Project/cs5180-project/experiments/dqn_test/agents"

dqnAgent = Agent.load(f"{agentsFolderpath}/dqn")
ddqnAgent = Agent.load(f"{agentsFolderpath}/ddqn")

# set epsilon to 0 so both agents act greedily during testing
dqnAgent.policy.epsilon = 0
ddqnAgent.policy.epsilon = 0

# removed one_hot_obs — raw 4-dim obs [agent_row, agent_col, goal_row, goal_col] is much easier to learn from
# testEnv = GridWorld(render_mode=None, step_limit=999, fixed_goal=True, flatten_obs=True, one_hot_obs=True)
testEnv = GridWorld(render_mode=None, step_limit=999, fixed_goal=True, flatten_obs=True)

dqnResults = dqnAgent.test(testEnv, episodes=50, runs=4, start_seed=3000,
                           folderpath=f"{agentsFolderpath}/dqn", test_name="test0")

ddqnResults = ddqnAgent.test(testEnv, episodes=50, runs=4, start_seed=3000,
                             folderpath=f"{agentsFolderpath}/ddqn", test_name="test0")

print(f"dqn  - avg return: {np.mean(dqnResults['episode_returns']):.3f}  avg length: {np.mean(dqnResults['episode_lengths']):.1f}")
print(f"ddqn - avg return: {np.mean(ddqnResults['episode_returns']):.3f}  avg length: {np.mean(ddqnResults['episode_lengths']):.1f}")
