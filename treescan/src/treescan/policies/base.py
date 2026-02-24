"""An abstract policy class"""

from abc import ABC, abstractmethod

class Policy(ABC):

    @abstractmethod
    def choose_action(self,state):
        """Return an action based on the state"""
        pass

    @abstractmethod
    def train(self,environment):
        """Train the policy on an environment"""
        pass

    @abstractmethod
    def save(self,folderpath):
        """Save the policy to a file"""
        pass

    @classmethod
    @abstractmethod
    def load(cls,folderpath):
        """Load the policy from a file"""
        pass
    
    