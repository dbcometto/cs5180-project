# deep q-network (dqn) policy

import copy
import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm

from treescan.policies.base import Policy


# replay buffer stores transitions and lets us sample random batches for training
class ReplayBuffer:

    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.buffer = []
        self.nextIndex = 0

    def add(self, obs, actionIndex, reward, nextObs, done):
        transition = (obs, actionIndex, reward, nextObs, done)
        if self.nextIndex >= len(self.buffer):
            self.buffer.append(transition)
        else:
            self.buffer[self.nextIndex] = transition
        # wrap around when full
        self.nextIndex = (self.nextIndex + 1) % self.bufferSize

    def sample(self, batchSize):
        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batchSize)]
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
        )

    def __len__(self):
        return len(self.buffer)


class DiscreteDQN(Policy):

    def __init__(self, qNetwork, actions, obs_dim,
                 lr=1e-3,
                 gamma=0.99,
                 bufferSize=50000,
                 batchSize=32,
                 epsilonStart=1.0,
                 epsilonEnd=0.01,
                 epsilonDecaySteps=50000,
                 targetUpdateFreq=1000):

        # behavior network: the network being actively trained
        self.qNetwork = qNetwork
        # target network: frozen copy used to compute stable td targets
        self.targetNetwork = copy.deepcopy(qNetwork)
        self.targetNetwork.load_state_dict(qNetwork.state_dict())

        self.actions = actions
        self.actionToIndex = {a: i for i, a in enumerate(actions)}

        # make sure the network output size matches the number of actions
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

        # epsilon is updated during training and used by choose_action
        self.epsilon = epsilonStart

    def choose_action(self, env, obs):
        # epsilon-greedy: random action with probability epsilon, greedy otherwise
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            qValues = self.qNetwork(obs)
            actionIndex = qValues.argmax(dim=-1).item()
        return self.actions[actionIndex]

    def _computeEpsilon(self, step):
        # linearly decay epsilon from epsilonStart down to epsilonEnd over epsilonDecaySteps
        fraction = min(step / self.epsilonDecaySteps, 1.0)
        return self.epsilonStart + fraction * (self.epsilonEnd - self.epsilonStart)

    def _updateBehaviorNetwork(self, replayBuffer):
        observations, actions, rewards, nextObservations, dones = replayBuffer.sample(self.batchSize)

        # get the q value for the action that was actually taken
        predictedQValues = self.qNetwork(observations).gather(1, actions.unsqueeze(1)).squeeze(1)

        # dqn td target: r + gamma * max_a Q_target(s') * (1 - done)
        with torch.no_grad():
            nextQValues = self.targetNetwork(nextObservations).max(dim=1)[0]
            tdTargets = rewards + self.gamma * nextQValues * (1 - dones)

        loss = torch.nn.functional.mse_loss(predictedQValues, tdTargets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _updateTargetNetwork(self):
        # hard update: copy behavior network weights directly into the target network
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())

    def train(self, env, totalSteps=100000, startTrainingStep=1000,
              updateFreq=4, startSeed=None):

        replayBuffer = ReplayBuffer(self.bufferSize)

        trainingReturns = []
        trainingLengths = []
        trainingLosses = []
        currentRewards = []

        seed = startSeed
        obs, _ = env.reset(seed=seed)
        if startSeed is not None:
            seed += 1

        for step in tqdm(range(totalSteps), desc="DQN", leave=False):

            self.epsilon = self._computeEpsilon(step)

            # take an action and observe the result
            action = self.choose_action(env, obs)
            nextObs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replayBuffer.add(obs, self.actionToIndex[action], reward, nextObs, float(done))
            currentRewards.append(reward)

            if done:
                # compute discounted return for this episode
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

            # wait until we have enough experience before training
            if step >= startTrainingStep and len(replayBuffer) >= self.batchSize:

                if step % updateFreq == 0:
                    loss = self._updateBehaviorNetwork(replayBuffer)
                    trainingLosses.append(loss)

                if step % self.targetUpdateFreq == 0:
                    self._updateTargetNetwork()

        return {
            "training_returns": trainingReturns,
            "training_lengths": trainingLengths,
            "training_losses": trainingLosses,
            "total_steps": totalSteps,
            "gamma": self.gamma,
            "start_seed": startSeed,
        }

    def save(self, folderpath):
        os.makedirs(folderpath, exist_ok=True)
        with open(f"{folderpath}/policy.pkl", "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, folderpath):
        with open(f"{folderpath}/policy.pkl", "rb") as file:
            return pickle.load(file)
