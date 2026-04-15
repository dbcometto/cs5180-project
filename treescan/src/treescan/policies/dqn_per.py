# dqn and ddqn with prioritized experience replay (schaul et al. 2016)
#
# differences from dqn.py / ddqn.py:
#   - replay buffer samples transitions with probability proportional to |td error|^alpha
#   - importance-sampling weights correct for the biased sampling in the loss
#   - optional per-action priority multiplier: lets us oversample rare but important
#     transitions (like END) even before their td error is known
# the per approach is motivated by the observation that in TreeWorld the END action
# is taken rarely (only once per episode, and only when the agent chooses it), so
# uniform sampling gives it little signal.

import copy
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from treescan.policies.base import Policy


# sumtree: a binary tree where each internal node holds the sum of its children.
# lets us sample an index with probability proportional to its priority in O(log N).
class SumTree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.size = 0

    def total(self):
        return self.tree[0]

    def update(self, dataIndex, priority):
        treeIndex = dataIndex + self.capacity - 1
        delta = priority - self.tree[treeIndex]
        self.tree[treeIndex] = priority
        # propagate the change up to the root
        parent = (treeIndex - 1) // 2
        while parent >= 0:
            self.tree[parent] += delta
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def add(self, dataIndex, priority):
        self.update(dataIndex, priority)
        if self.size < self.capacity:
            self.size += 1

    def find(self, value):
        # walk down the tree: go left if value <= left child sum, else go right
        index = 0
        while index < self.capacity - 1:
            left = 2 * index + 1
            right = left + 1
            if value <= self.tree[left]:
                index = left
            else:
                value -= self.tree[left]
                index = right
        dataIndex = index - (self.capacity - 1)
        return dataIndex, self.tree[index]


class PrioritizedReplayBuffer:

    def __init__(self, bufferSize, alpha=0.6, actionBoost=None):
        # alpha controls how much prioritization is used (0 = uniform, 1 = fully proportional)
        # actionBoost: dict mapping actionIndex -> priority multiplier (e.g. END -> 5.0)
        self.bufferSize = bufferSize
        self.alpha = alpha
        self.actionBoost = actionBoost if actionBoost is not None else {}
        self.tree = SumTree(bufferSize)
        self.buffer = [None] * bufferSize
        self.nextIndex = 0
        self.maxPriority = 1.0
        self.epsilon = 1e-6

    def add(self, obs, actionIndex, reward, nextObs, done):
        self.buffer[self.nextIndex] = (obs, actionIndex, reward, nextObs, done)
        # new transitions get max priority so they are seen at least once
        priority = self.maxPriority
        boost = self.actionBoost.get(actionIndex, 1.0)
        self.tree.add(self.nextIndex, (priority * boost) ** self.alpha)
        self.nextIndex = (self.nextIndex + 1) % self.bufferSize

    def sample(self, batchSize, beta=0.4):
        total = self.tree.total()
        segment = total / batchSize
        indices = np.empty(batchSize, dtype=np.int64)
        priorities = np.empty(batchSize, dtype=np.float64)

        # stratified sampling: one sample per equal-sized segment of the priority mass
        for i in range(batchSize):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            dataIndex, priority = self.tree.find(value)
            indices[i] = dataIndex
            priorities[i] = priority

        sampleProbs = priorities / total
        # importance-sampling weights: correct for non-uniform sampling
        weights = (self.tree.size * sampleProbs) ** (-beta)
        weights = weights / weights.max()

        observations, actions, rewards, nextObservations, dones = [], [], [], [], []
        for index in indices:
            obs, action, reward, nextObs, done = self.buffer[index]
            observations.append(obs.numpy())
            actions.append(action)
            rewards.append(reward)
            nextObservations.append(nextObs.numpy())
            dones.append(done)

        return (
            torch.tensor(np.array(observations), dtype=torch.float),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(np.array(nextObservations), dtype=torch.float),
            torch.tensor(dones, dtype=torch.float),
            torch.tensor(weights, dtype=torch.float),
            indices,
        )

    def update_priorities(self, indices, tdErrors):
        tdErrors = np.abs(tdErrors) + self.epsilon
        for dataIndex, tdError in zip(indices, tdErrors):
            actionIndex = self.buffer[dataIndex][1]
            boost = self.actionBoost.get(actionIndex, 1.0)
            priority = (tdError * boost) ** self.alpha
            self.tree.update(dataIndex, priority)
            if tdError > self.maxPriority:
                self.maxPriority = tdError

    def __len__(self):
        return self.tree.size


class _PERTrainerMixin:
    """Shared train/save/load plumbing for DQN and DDQN with prioritized replay."""

    def _computeEpsilon(self, step):
        fraction = min(step / self.epsilonDecaySteps, 1.0)
        return self.epsilonStart + fraction * (self.epsilonEnd - self.epsilonStart)

    def _computeBeta(self, step):
        # linearly anneal IS weight exponent beta from betaStart to 1.0 over training
        fraction = min(step / self.betaAnnealSteps, 1.0)
        return self.betaStart + fraction * (1.0 - self.betaStart)

    def _updateTargetNetwork(self):
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())

    def choose_action(self, env, obs):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            qValues = self.qNetwork(obs)
            actionIndex = qValues.argmax(dim=-1).item()
        return self.actions[actionIndex]

    def train(self, env, totalSteps=100000, startTrainingStep=1000,
              updateFreq=4, startSeed=None,
              folderpath=None, checkpointInterval=10000, resumeStep=None):

        replayBuffer = PrioritizedReplayBuffer(
            self.bufferSize, alpha=self.alpha, actionBoost=self.actionBoost
        )

        if resumeStep is not None and folderpath is not None:
            try:
                resumeStep, seed, trainingReturns, trainingLengths, trainingLosses = self.load_checkpoint(folderpath, resumeStep)
                print(f"Resuming from step {resumeStep}")
            except Exception as e:
                raise RuntimeError(f"Failed to resume from checkpoint at step {resumeStep}") from e
        else:
            trainingReturns = []
            trainingLengths = []
            trainingLosses = []
            seed = startSeed

        currentRewards = []
        startStep = resumeStep + 1 if resumeStep is not None else 0
        step = startStep

        obs, _ = env.reset(seed=seed)
        if startSeed is not None:
            seed += 1

        desc = getattr(self, "_tqdmDesc", "DQN-PER")

        try:
            for step in tqdm(range(startStep, totalSteps), desc=desc, leave=False):

                self.epsilon = self._computeEpsilon(step)

                action = self.choose_action(env, obs)
                nextObs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                replayBuffer.add(obs, self.actionToIndex[action], reward, nextObs, float(done))
                currentRewards.append(reward)

                if done:
                    episodeReturn = 0
                    for r in reversed(currentRewards):
                        episodeReturn = r + self.gamma * episodeReturn
                    trainingReturns.append(episodeReturn)
                    trainingLengths.append(len(currentRewards))

                    currentRewards = []
                    obs, _ = env.reset(seed=seed)
                    if startSeed is not None:
                        seed += 1
                else:
                    obs = nextObs

                if step >= startTrainingStep and len(replayBuffer) >= self.batchSize:

                    if step % updateFreq == 0:
                        beta = self._computeBeta(step)
                        loss = self._updateBehaviorNetwork(replayBuffer, beta)
                        trainingLosses.append(loss)

                    if step % self.targetUpdateFreq == 0:
                        self._updateTargetNetwork()

                if folderpath is not None:
                    if step % checkpointInterval == 0:
                        self.save_checkpoint(folderpath, step, seed, trainingReturns, trainingLengths, trainingLosses)

            if folderpath is not None:
                self.save_checkpoint(folderpath, step, seed, trainingReturns, trainingLengths, trainingLosses)
                print(f"Finished training and saved checkpoint at step {step} to file at {folderpath}")

        except KeyboardInterrupt:
            if folderpath is not None:
                print(f"Interrupted at step {step} and saved checkpoint to file at {folderpath}")
            else:
                print(f"Interrupted at step {step} and not saved (no filepath provided)")

        except Exception as e:
            if folderpath is not None:
                print(f"Exception at step {step} and saved checkpoint to file at {folderpath} | Exception: {e}")
            else:
                print(f"Exception at step {step} and not saved (no filepath provided) | Exception: {e}")
            raise e

        finally:
            if folderpath is not None:
                self.save_checkpoint(folderpath, step, seed, trainingReturns, trainingLengths, trainingLosses)

        return {
            "training_returns": trainingReturns,
            "training_lengths": trainingLengths,
            "training_losses": trainingLosses,
            "total_steps": totalSteps,
            "gamma": self.gamma,
            "start_seed": startSeed,
            "step_completed": step,
        }

    def save(self, folderpath):
        os.makedirs(folderpath, exist_ok=True)
        with open(f"{folderpath}/policy.pkl", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, folderpath):
        with open(f"{folderpath}/policy.pkl", "rb") as file:
            return pickle.load(file)

    @classmethod
    def load_from_checkpoint(cls, folderpath, checkpointStep):
        with open(f"{folderpath}/policy.pkl", "rb") as file:
            policy = pickle.load(file)
        policy.load_checkpoint(folderpath, checkpointStep)
        return policy

    def save_checkpoint(self, folderpath, step, seed, trainingReturns, trainingLengths, trainingLosses):
        path = f"{folderpath}/checkpoints"
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            "step": step,
            "seed": seed,
            "qNetwork": self.qNetwork.state_dict(),
            "targetNetwork": self.targetNetwork.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "trainingReturns": trainingReturns,
            "trainingLengths": trainingLengths,
            "trainingLosses": trainingLosses,
            "rng_state": torch.get_rng_state(),
        }
        torch.save(checkpoint, f"{path}/ckpt_{step}.pt")

    def load_checkpoint(self, folderpath, step):
        path = f"{folderpath}/checkpoints/ckpt_{step}.pt"
        checkpoint = torch.load(path, weights_only=False)
        self.qNetwork.load_state_dict(checkpoint["qNetwork"])
        self.targetNetwork.load_state_dict(checkpoint["targetNetwork"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "rng_state" in checkpoint.keys():
            torch.set_rng_state(checkpoint["rng_state"])
        return checkpoint["step"], checkpoint["seed"], checkpoint["trainingReturns"], checkpoint["trainingLengths"], checkpoint["trainingLosses"]


class DiscreteDQNPrioritized(_PERTrainerMixin, Policy):

    _tqdmDesc = "DQN-PER"

    def __init__(self, qNetwork, actions, obs_dim,
                 lr=1e-3,
                 gamma=0.99,
                 bufferSize=50000,
                 batchSize=32,
                 epsilonStart=1.0,
                 epsilonEnd=0.01,
                 epsilonDecaySteps=50000,
                 targetUpdateFreq=1000,
                 alpha=0.6,
                 betaStart=0.4,
                 betaAnnealSteps=1000000,
                 actionBoost=None):

        self.qNetwork = qNetwork
        self.targetNetwork = copy.deepcopy(qNetwork)
        self.targetNetwork.load_state_dict(qNetwork.state_dict())

        self.actions = actions
        self.actionToIndex = {a: i for i, a in enumerate(actions)}

        with torch.no_grad():
            dummy = torch.zeros(obs_dim)
            if self.qNetwork(dummy).shape[-1] != len(actions):
                raise ValueError("network output size does not match number of actions")

        self.optimizer = torch.optim.Adam(self.qNetwork.parameters(), lr=lr)

        self.gamma = gamma
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.epsilonStart = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.epsilonDecaySteps = epsilonDecaySteps
        self.targetUpdateFreq = targetUpdateFreq

        self.alpha = alpha
        self.betaStart = betaStart
        self.betaAnnealSteps = betaAnnealSteps
        self.actionBoost = actionBoost if actionBoost is not None else {}

        self.epsilon = epsilonStart

    def _updateBehaviorNetwork(self, replayBuffer, beta):
        observations, actions, rewards, nextObservations, dones, weights, indices = replayBuffer.sample(self.batchSize, beta=beta)

        predictedQValues = self.qNetwork(observations).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            nextQValues = self.targetNetwork(nextObservations).max(dim=1)[0]
            tdTargets = rewards + self.gamma * nextQValues * (1 - dones)

        tdErrors = tdTargets - predictedQValues
        # IS-weighted huber loss (less sensitive to the occasional large td error)
        loss = (weights * torch.nn.functional.smooth_l1_loss(predictedQValues, tdTargets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        replayBuffer.update_priorities(indices, tdErrors.detach().cpu().numpy())

        return loss.item()


class DiscreteDoubleDQNPrioritized(_PERTrainerMixin, Policy):

    _tqdmDesc = "DDQN-PER"

    def __init__(self, qNetwork, actions, obs_dim,
                 lr=1e-3,
                 gamma=0.99,
                 bufferSize=50000,
                 batchSize=32,
                 epsilonStart=1.0,
                 epsilonEnd=0.01,
                 epsilonDecaySteps=50000,
                 targetUpdateFreq=1000,
                 alpha=0.6,
                 betaStart=0.4,
                 betaAnnealSteps=1000000,
                 actionBoost=None):

        self.qNetwork = qNetwork
        self.targetNetwork = copy.deepcopy(qNetwork)
        self.targetNetwork.load_state_dict(qNetwork.state_dict())

        self.actions = actions
        self.actionToIndex = {a: i for i, a in enumerate(actions)}

        with torch.no_grad():
            dummy = torch.zeros(obs_dim)
            if self.qNetwork(dummy).shape[-1] != len(actions):
                raise ValueError("network output size does not match number of actions")

        self.optimizer = torch.optim.Adam(self.qNetwork.parameters(), lr=lr)

        self.gamma = gamma
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.epsilonStart = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.epsilonDecaySteps = epsilonDecaySteps
        self.targetUpdateFreq = targetUpdateFreq

        self.alpha = alpha
        self.betaStart = betaStart
        self.betaAnnealSteps = betaAnnealSteps
        self.actionBoost = actionBoost if actionBoost is not None else {}

        self.epsilon = epsilonStart

    def _updateBehaviorNetwork(self, replayBuffer, beta):
        observations, actions, rewards, nextObservations, dones, weights, indices = replayBuffer.sample(self.batchSize, beta=beta)

        predictedQValues = self.qNetwork(observations).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            bestActions = self.qNetwork(nextObservations).argmax(dim=1)
            nextQValues = self.targetNetwork(nextObservations).gather(1, bestActions.unsqueeze(1)).squeeze(1)
            tdTargets = rewards + self.gamma * nextQValues * (1 - dones)

        tdErrors = tdTargets - predictedQValues
        loss = (weights * torch.nn.functional.smooth_l1_loss(predictedQValues, tdTargets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        replayBuffer.update_priorities(indices, tdErrors.detach().cpu().numpy())

        return loss.item()
