# trains and saves dqn and ddqn agents on treeworld

import time

from treescan.environments import TreeWorld
from treescan.networks.treeworld import SimpleConvNetwork
from treescan.policies import DiscreteDQN, DiscreteDoubleDQN
from treescan.agents import Agent


agentsFolderpath = "/Users/adamlewis/Desktop/Northeastern/Reinforcement Learning/Project/cs5180-project/experiments/dqn_test/agents"

trainEnv = TreeWorld(render_mode=None, step_limit=200, obs_as_tensor=True, use_fixed_map=False,
                     enable_extra_channels=True, do_smooth_complete_reward=True,
                     discourage_early_end=True, do_reward_tree_complete=True)
# v7 reward shaping: encourage scanning near trees and completing each tree
trainEnv.REWARD_STEP = -0.1
trainEnv.REWARD_SCAN = -0.2         # was -0.7; wasted scans no longer catastrophic
trainEnv.REWARD_NEW_FACE = 2.5      # was 0.8; new-face scans are clearly worth it
trainEnv.REWARD_EXPLORE_TILE = 0.3  # was 0.2; push agent to explore toward far trees
trainEnv.REWARD_TREE_COMPLETE = 3.0 # bonus per tree with all 4 faces scanned
obs, _ = trainEnv.reset(seed=2025)

obsChannels = obs.shape[0]
obsDim = obs.shape
actionList = [a for a in trainEnv.ACTIONS]
actionDim = len(actionList)


# train dqn
print("training dqn...")
start = time.time()

dqnNetwork = SimpleConvNetwork(input_channels=obsChannels, output_width=actionDim)
dqnPolicy = DiscreteDQN(dqnNetwork, actions=actionList, obs_dim=obsDim,
                        lr=3e-4, gamma=0.99,
                        bufferSize=100000, batchSize=64,
                        epsilonStart=1.0, epsilonEnd=0.01,
                        epsilonDecaySteps=250000, targetUpdateFreq=2000)
dqnAgent = Agent(dqnPolicy)

dqnFolderpath = f"{agentsFolderpath}/dqn_v7"

try:
    dqnAgent.train(trainEnv, totalSteps=500000, startTrainingStep=5000, updateFreq=4, startSeed=2025,
                   folderpath=dqnFolderpath, checkpointInterval=25000)
except KeyboardInterrupt:
    print("Interrupting...")
except Exception as e:
    print("Exception occurred")
    raise
finally:
    dqnAgent.save(dqnFolderpath)
    print(f"dqn done in {time.time()-start:.1f}s")


# train ddqn
print("training ddqn...")
start = time.time()

ddqnNetwork = SimpleConvNetwork(input_channels=obsChannels, output_width=actionDim)
ddqnPolicy = DiscreteDoubleDQN(ddqnNetwork, actions=actionList, obs_dim=obsDim,
                               lr=3e-4, gamma=0.99,
                               bufferSize=100000, batchSize=64,
                               epsilonStart=1.0, epsilonEnd=0.01,
                               epsilonDecaySteps=250000, targetUpdateFreq=2000)
ddqnAgent = Agent(ddqnPolicy)

ddqnFolderpath = f"{agentsFolderpath}/ddqn_v7"

try:
    ddqnAgent.train(trainEnv, totalSteps=500000, startTrainingStep=5000, updateFreq=4, startSeed=2025,
                    folderpath=ddqnFolderpath, checkpointInterval=25000)
except KeyboardInterrupt:
    print("Interrupting...")
except Exception as e:
    print("Exception occurred")
    raise
finally:
    ddqnAgent.save(ddqnFolderpath)
    print(f"ddqn done in {time.time()-start:.1f}s")
