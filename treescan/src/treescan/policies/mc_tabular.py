"""Defines MC policies"""
import gymnasium as gym
from tqdm import tqdm
import numpy as np

from typing import Optional
import pickle
import os

from treescan.policies.base import Policy
from treescan.utils import generate_trajectory


class MCTabularFirstVisitEpsilonControl(Policy):
    """A tabular first-visit Monte-Carlo control policy"""

    def __init__(self, actions: list, states: list, epsilon: Optional[float] = 0.1, seed: Optional[int] = None):
        """Instantiate the policy with random actions
        
        Args:
            actions (list): a list of possible actions available at every state
            states (list): a list of every state
        """

        rng = np.random.default_rng(seed)
        

        self.actions = actions
        self.states = states
        self.table = {s: rng.choice(self.actions) for s in self.states}

        self.Q = {s: {a: 0 for a in self.actions} for s in self.states}

        self.epsilon = epsilon

        


    def choose_action(self, env: gym.Env, state):
        """Return an action based on the state"""
        if env.np_random.random() < self.epsilon:
            return env.np_random.choice(self.actions)
        else:
            return self.table[env.encode_obs(state)]

    def train(self, env: gym.Env, episodes: Optional[int] = 1, gamma: Optional[float] = 1.0):
        """Generates a trajectory for each episode and trains the agent on them
        
        Args:
            env (gym.Env): the environment
            epiodes (int, optional): number of training episodes
            gamma (int, optional): discount factor
            
        Returns:
            info (dict): 
                - 'episode_lengths' (list): lengths of each training episode
                - 'episode_returns' (list): rewards of each training epiosde
        """
        Returns = {s:{a: [] for a in self.actions} for s in self.states}
        episode_returns = []
        episode_lengths = []

        for i in tqdm(range(episodes),desc="MC",leave=False,position=1):
            T = generate_trajectory(env,self)

            G = 0
            for j,transition in enumerate(reversed(T)):
                s,a,next_s,r,term,trunc,_ = transition
                G = r + gamma*G

                s = env.encode_obs(s)
                a = env.encode_act(a)

                if (s,a) not in [(env.encode_obs(t[0]),env.encode_act(t[1])) for t in T[0:len(T)-1-j]]:
                    Returns[s][a].append(G)
                    self.Q[s][a] = np.mean(Returns[s][a]) if len(Returns[s][a]) > 0 else 0

                    maxQ = max(self.Q[s].values())
                    astarlist = [act for act in self.actions if self.Q[s][act] == maxQ]
                    astar = env.np_random.choice(astarlist)
                    
                    self.table[s] = astar


            episode_lengths.append(len(T))
            episode_returns.append(G)

        info = {
            "episode_lengths": episode_lengths,
            "episode_returns": episode_returns,
        }

                  
        return info
    
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