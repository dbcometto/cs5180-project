"""Defines a random policy"""
import gymnasium as gym

from typing import Optional

from treescan.policies.base import Policy


class RandomTabularPolicy(Policy):
    """A Policy that takes a random action"""

    def __init__(self,env: gym.Env,actions: list,probs: Optional[dict] = None):
        """Instantiate the policy
        
        Args:
            env (gym.Env): a gym env for the random generator
            actions (list): a list of possible actions available at every state
            probs (dict, optional): a dictionary mapping options to probabilities
        """

        self.np_random = env.np_random

        self.actions = actions
        self.probs = probs if probs else [1/len(actions)]*len(actions)


    def choose_action(self, state):
        """Return an action based on the state"""
        return self.np_random.choice(self.actions,p=self.probs)

    def update(self):
        """Random policy has no training"""
        pass 