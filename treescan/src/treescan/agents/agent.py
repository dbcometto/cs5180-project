"""An agent class"""
import numpy as np
from tqdm import tqdm

from typing import Optional
import json
import pickle
import os
import importlib

from treescan.policies import Policy
from treescan.utils import generate_trajectory



class Agent():

    def __init__(self, policy: Policy):
        """Create an agent, who holds a policy and data"""
        self.policy = policy
        self.training_results = None

    def choose_action(self, env, state):
        """Return an action based on the state"""
        return self.policy.choose_action(env, state)
        
    def generate_trajectory(self, environment, max_steps: Optional[int]=1000, seed: Optional[int]=None):
        """Generate a trajectory and return the transition table"""
        return generate_trajectory(environment,self.policy,seed=seed)
    
    def train(self, environment, *args, **kwargs):
        """Train the agent on an environment, see policy for details"""
        results = self.policy.train(environment,*args,**kwargs)
        self.training_results = results
        return results
    
    def test(self, environment, episodes: int, runs: Optional[int] = 1, gamma: Optional[float] = 1.0, max_steps: Optional[int] = 1000, start_seed: Optional[int]=None):
        """Test the agent on an environment"""
        episode_lengths = np.empty((runs,episodes))*np.nan
        episode_returns = np.empty((runs,episodes))*np.nan

        if start_seed is not None:
            seed = start_seed

        for j in tqdm(range(runs),desc="Runs",leave=False):
            for i in tqdm(range(episodes),desc="Episodes",leave=False):
                T = self.generate_trajectory(environment,max_steps=max_steps,seed=seed)
            
                G = 0
                for transition in reversed(T):
                    s,a,next_s,r,term,trunc,_ = transition
                    G = r + gamma*G

                episode_lengths[j,i] = len(T)
                episode_returns[j,i] = G

                if start_seed is not None:
                    seed += 1

            info = {
                "episode_lengths": episode_lengths,
                "episode_returns": episode_returns,
            }

        return info



    def save(self, folderpath):
        """Save the agent to a file"""

        agent_data = {
            "policy_class": self.policy.__class__.__name__,
            "policy_module": self.policy.__class__.__module__,
            "training_results": self.training_results,
        }

        # Save data
        os.makedirs(folderpath, exist_ok=True)
        with open(f"{folderpath}/data.json","w") as file:
            json.dump(agent_data,file,indent=4)

        # Save policy
        self.policy.save(folderpath)

        

    @classmethod    
    def load(cls, folderpath):
        """Create an agent from a file"""

        # Load data
        data_path = f"{folderpath}/data.json"
        if os.path.exists(data_path):
            with open(data_path,"r") as file:
                agent_data = json.load(file)

            # Make Policy
            module = importlib.import_module(agent_data["policy_module"])
            PolicyCls = getattr(module,agent_data["policy_class"])
            policy = PolicyCls.load(folderpath)

            # Make agent
            agent = cls(policy)
            agent.training_results = agent_data["training_results"]

            return agent
        else:
            raise FileNotFoundError(f"Cannot find agent at path {folderpath}")