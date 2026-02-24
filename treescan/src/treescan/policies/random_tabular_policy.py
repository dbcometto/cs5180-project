"""Defines a random policy"""
import gymnasium as gym

from typing import Optional
import pickle
import os

from treescan.policies.base import Policy


class RandomTabularPolicy(Policy):
    """A Policy that takes a random action"""

    def __init__(self, actions: list,probs: Optional[dict] = None):
        """Instantiate the policy
        
        Args:
            actions (list): a list of possible actions available at every state
            probs (dict, optional): a dictionary mapping options to probabilities
        """

        self.actions = actions
        self.probs = probs if probs else [1/len(actions)]*len(actions)


    def choose_action(self, env: gym.Env, state):
        """Return an action based on the state"""
        return env.np_random.choice(self.actions,p=self.probs)

    def train(self):
        """Random policy has no training"""
        pass 

    def save(self,folderpath):
        """Save the policy to a file"""
        os.makedirs(folderpath, exist_ok=True)
        with open(f"{folderpath}/policy.pkl","wb") as file:
            pickle.dump(self,file)

    @classmethod
    def load(cls,folderpath):
        """Load the policy from a file"""

        with open(f"{folderpath}/policy.pkl","rb") as file:
            return pickle.load(file)

