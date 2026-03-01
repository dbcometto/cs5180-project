"""Defines MC policies"""
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch

from typing import Optional
import pickle
import os
from collections import OrderedDict

from treescan.policies.base import Policy
from treescan.utils import generate_trajectory

class DiscreteReinforce(Policy):
    """A network policy using the REINFORCE algorithm"""

    def __init__(self, network: torch.nn.Module, actions: list, obs_dim: int, lr: Optional[float] = 0.001, weight_decay: Optional[float] = 0):
        """Instantiate the policy on a network
        
        Args:
            network (torch.nn.Module): A Torch network approximating optimal action logits from observation
            actions (list): a list of all possible actions
            obs_dim (int): the length of the flattened observation
            lr (float): learning rate
            weight_decay (float): weight decay
        """
        
        self.network = network
        self.actions = actions
        self.action_index = {a: i for i,a in enumerate(actions)}

        dummy_obs = torch.zeros(obs_dim)
        if self.network(dummy_obs).shape[0] != len(self.actions):
            raise ValueError("Network, state, and action shapes do not align")
        
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=lr,weight_decay=weight_decay)
 


    def choose_action(self, env: gym.Env, obs):
        """Return an action based on the state"""
        with torch.no_grad():
            probabilites = torch.softmax(self.network(obs),dim=-1).detach().numpy()
        return env.np_random.choice(self.actions,p=probabilites)
    

    def loss_fn(self, obs, a, G):
        return -torch.log(self.network(obs)[self.action_index[a]])*G


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
        episode_returns = []
        episode_lengths = []
        episode_losses = []

        for i in tqdm(range(episodes),desc="REINFORCE",leave=False,position=1):
            T = generate_trajectory(env,self)

            G = 0
            losses = []
            for j,transition in enumerate(reversed(T)):
                obs,a,next_obs,r,term,trunc,_ = transition
                G = r + gamma*G

                self.optimizer.zero_grad()
                loss = self.loss_fn(obs,a,G)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())


            episode_lengths.append(len(T))
            episode_returns.append(G)
            episode_losses.append(np.mean(losses))


        info = {
            "episode_lengths": episode_lengths,
            "episode_returns": episode_returns,
            "episode_losses": episode_losses,
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