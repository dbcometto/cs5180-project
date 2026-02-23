"""An agent class"""
from treescan.policies import Policy

class Agent():

    def __init__(self, policy: Policy):
        """Create an agent, who holds a policy and data"""
        self.policy = policy

    def choose_action(self,state):
        """Return an action based on the state"""
        return self.policy.choose_action(state)
    
    def train_policy(self,training_function):
        training_function(self.policy)

    def save(self,filepath):
        """Save the agent to a file"""
        pass

    @classmethod    
    def load(cls,filepath):
        """Create an agent from a file"""
        pass