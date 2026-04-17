# v8: trains dqn and ddqn with prioritized experience replay on TreeWorld (random maps).
#
# motivation:
#   - v7 agents on random maps often wander back and forth in a small area instead of
#     exploring, scanning all tree faces, and then calling END.
#   - PER oversamples high-td-error transitions so END and near-END transitions get
#     more gradient signal per unit time.
#   - additional per-action priority boost on END further oversamples that action,
#     which is rare in the buffer (at most once per episode).
#   - extra WAIT penalty added via a gym RewardWrapper discourages the sit in place
#     sub-optimum observed in the v7 demo.
#   - BetterConvNetwork (extra residual layers) gives more capacity to learn the
#     random-map generalization problem.

import time

import gymnasium as gym

from treescan.environments import TreeWorld
from treescan.networks.treeworld import BetterConvNetwork
from treescan.policies import DiscreteDQNPrioritized, DiscreteDoubleDQNPrioritized
from treescan.agents import Agent


agentsFolderpath = "/Users/adamlewis/Desktop/Northeastern/Reinforcement Learning/Project/cs5180-project/experiments/dqn_test/agents"


# wrapper adds an extra penalty when the agent picks the WAIT action. the base env
# charges REWARD_STEP for every step, but does not specifically discourage WAIT
# this extra penalty makes the cheap "do nothing" action strictly worse than moving.
class WaitPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, waitPenalty=-0.3):
        super().__init__(env)
        self.waitPenalty = waitPenalty
        self._waitAction = int(env.unwrapped.ACTIONS.WAIT)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if int(action) == self._waitAction:
            reward += self.waitPenalty
        return obs, reward, terminated, truncated, info


def buildTrainEnv():
    env = TreeWorld(render_mode=None, step_limit=200, obs_as_tensor=True,
                    use_fixed_map=False,
                    enable_extra_channels=True, do_smooth_complete_reward=True,
                    discourage_early_end=True, do_reward_tree_complete=True)
    # v8 reward shaping (mirrors v7 but with a stiffer per-step penalty)
    env.REWARD_STEP = -0.15
    env.REWARD_SCAN = -0.2
    env.REWARD_NEW_FACE = 2.5
    env.REWARD_EXPLORE_TILE = 0.3
    env.REWARD_TREE_COMPLETE = 3.0
    return WaitPenaltyWrapper(env, waitPenalty=-0.3)


trainEnv = buildTrainEnv()
obs, _ = trainEnv.reset(seed=2025)

obsChannels = obs.shape[0]
obsDim = obs.shape
actionList = [a for a in trainEnv.unwrapped.ACTIONS]
actionDim = len(actionList)

# END gets a 5x priority multiplier in the replay buffer so the agent trains on
# it disproportionately often relative to its sampling frequency.
endActionIndex = int(trainEnv.unwrapped.ACTIONS.END)
actionBoost = {endActionIndex: 5.0}


TOTAL_STEPS = 250_000


# train dqn
print("training dqn (v8, PER)...")
start = time.time()

dqnNetwork = BetterConvNetwork(input_channels=obsChannels, output_width=actionDim)
dqnPolicy = DiscreteDQNPrioritized(
    dqnNetwork, actions=actionList, obs_dim=obsDim,
    lr=3e-4, gamma=0.99,
    bufferSize=200000, batchSize=64,
    epsilonStart=1.0, epsilonEnd=0.01,
    epsilonDecaySteps=125000, targetUpdateFreq=2000,
    alpha=0.6, betaStart=0.4, betaAnnealSteps=TOTAL_STEPS,
    actionBoost=actionBoost,
)
dqnAgent = Agent(dqnPolicy)

dqnFolderpath = f"{agentsFolderpath}/dqn_v8"

try:
    dqnAgent.train(trainEnv, totalSteps=TOTAL_STEPS, startTrainingStep=5000, updateFreq=4, startSeed=2025,
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
print("training ddqn (v8, PER)...")
start = time.time()

trainEnv = buildTrainEnv()
trainEnv.reset(seed=2025)

ddqnNetwork = BetterConvNetwork(input_channels=obsChannels, output_width=actionDim)
ddqnPolicy = DiscreteDoubleDQNPrioritized(
    ddqnNetwork, actions=actionList, obs_dim=obsDim,
    lr=3e-4, gamma=0.99,
    bufferSize=200000, batchSize=64,
    epsilonStart=1.0, epsilonEnd=0.01,
    epsilonDecaySteps=125000, targetUpdateFreq=2000,
    alpha=0.6, betaStart=0.4, betaAnnealSteps=TOTAL_STEPS,
    actionBoost=actionBoost,
)
ddqnAgent = Agent(ddqnPolicy)

ddqnFolderpath = f"{agentsFolderpath}/ddqn_v8"

try:
    ddqnAgent.train(trainEnv, totalSteps=TOTAL_STEPS, startTrainingStep=5000, updateFreq=4, startSeed=2025,
                    folderpath=ddqnFolderpath, checkpointInterval=25000)
except KeyboardInterrupt:
    print("Interrupting...")
except Exception as e:
    print("Exception occurred")
    raise
finally:
    ddqnAgent.save(ddqnFolderpath)
    print(f"ddqn done in {time.time()-start:.1f}s")
