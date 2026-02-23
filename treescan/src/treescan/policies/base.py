"""An abstract policy class"""

from abc import ABC, abstractmethod

class Policy(ABC):

    @abstractmethod
    def choose_action(self,state):
        """Return an action based on the state"""
        pass

    @abstractmethod
    def update(self,*args,**kwargs):
        """Given data, update the policy"""
        pass
    