# double deep q-network (ddqn) policy
#
# the only difference from dqn is in the td target:
#   dqn:  uses the target network to both SELECT and EVALUATE the best next action
#   ddqn: uses the behavior network to SELECT the best next action,
#         then uses the target network to EVALUATE it
# this decoupling reduces overestimation bias in the q values

import copy
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from treescan.policies.base import Policy
from treescan.policies.dqn import ReplayBuffer


class DiscreteDoubleDQN(Policy):

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
        # target network: frozen copy used to evaluate actions
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

        # ddqn td target:
        # step 1: use behavior network to find the best action in the next state
        # step 2: use target network to evaluate that action (not pick it)
        with torch.no_grad():
            bestActions = self.qNetwork(nextObservations).argmax(dim=1)
            nextQValues = self.targetNetwork(nextObservations).gather(1, bestActions.unsqueeze(1)).squeeze(1)
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
              updateFreq=4, startSeed=None,
              folderpath=None, checkpointInterval=10000, resumeStep=None):

        replayBuffer = ReplayBuffer(self.bufferSize)

        # resume from checkpoint or start fresh
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

        try:
            for step in tqdm(range(startStep, totalSteps), desc="DDQN", leave=False):

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

                # save checkpoint at intervals
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
        """Load the policy from a checkpoint file"""
        with open(f"{folderpath}/policy.pkl", "rb") as file:
            policy = pickle.load(file)
        policy.load_checkpoint(folderpath, checkpointStep)
        return policy

    def save_checkpoint(self, folderpath, step, seed, trainingReturns, trainingLengths, trainingLosses):
        """Save a training checkpoint to a file"""
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
        """Load a training checkpoint from a file"""
        path = f"{folderpath}/checkpoints/ckpt_{step}.pt"
        checkpoint = torch.load(path, weights_only=False)

        self.qNetwork.load_state_dict(checkpoint["qNetwork"])
        self.targetNetwork.load_state_dict(checkpoint["targetNetwork"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "rng_state" in checkpoint.keys():
            torch.set_rng_state(checkpoint["rng_state"])

        return checkpoint["step"], checkpoint["seed"], checkpoint["trainingReturns"], checkpoint["trainingLengths"], checkpoint["trainingLosses"]
