# trains and saves dqn and ddqn agents on gridworld

import time

from treescan.environments import GridWorld
from treescan.networks.gridworld import SuperSimpleLogitNetwork
from treescan.policies import DiscreteDQN, DiscreteDoubleDQN
from treescan.agents import Agent


agentsFolderpath = "/Users/adamlewis/Desktop/Northeastern/Reinforcement Learning/Project/cs5180-project/experiments/dqn_test/agents"

# removed one_hot_obs — raw 4-dim obs [agent_row, agent_col, goal_row, goal_col] is much easier to learn from
# trainEnv = GridWorld(render_mode=None, step_limit=999, fixed_goal=True, flatten_obs=True, one_hot_obs=True)
trainEnv = GridWorld(render_mode=None, step_limit=999, fixed_goal=True, flatten_obs=True)
obs, _ = trainEnv.reset(seed=2025)

obsDim = len(obs)
actionList = [a for a in trainEnv.ACTIONS]
actionDim = len(actionList)


# train dqn
print("training dqn...")
start = time.time()

dqnNetwork = SuperSimpleLogitNetwork(input_width=obsDim, hidden_width=64, output_width=actionDim)
dqnPolicy = DiscreteDQN(dqnNetwork, actions=actionList, obs_dim=obsDim,
                        lr=1e-3, gamma=0.99,
                        bufferSize=50000, batchSize=32,
                        epsilonStart=1.0, epsilonEnd=0.01,
                        epsilonDecaySteps=50000, targetUpdateFreq=1000)
dqnAgent = Agent(dqnPolicy)
dqnAgent.train(trainEnv, totalSteps=200000, startTrainingStep=1000, updateFreq=4, startSeed=2025)
dqnAgent.save(f"{agentsFolderpath}/dqn")
print(f"dqn done in {time.time()-start:.1f}s")


# train ddqn
print("training ddqn...")
start = time.time()

ddqnNetwork = SuperSimpleLogitNetwork(input_width=obsDim, hidden_width=64, output_width=actionDim)
ddqnPolicy = DiscreteDoubleDQN(ddqnNetwork, actions=actionList, obs_dim=obsDim,
                               lr=1e-3, gamma=0.99,
                               bufferSize=50000, batchSize=32,
                               epsilonStart=1.0, epsilonEnd=0.01,
                               epsilonDecaySteps=50000, targetUpdateFreq=1000)
ddqnAgent = Agent(ddqnPolicy)
ddqnAgent.train(trainEnv, totalSteps=200000, startTrainingStep=1000, updateFreq=4, startSeed=2025)
ddqnAgent.save(f"{agentsFolderpath}/ddqn")
print(f"ddqn done in {time.time()-start:.1f}s")
