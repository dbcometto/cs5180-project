"""Defines A2C policies"""
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





class DiscreteAdvantageActorCritic(Policy):
    """A network policy using the A2C algorithm"""

    def __init__(self, logit_network: torch.nn.Module, value_network: torch.nn.Module, actions: list, obs_dim: int, logit_lr: Optional[float] = 0.001, value_lr: Optional[float] = 0.001, logit_weight_decay: Optional[float] = 0, value_weight_decay: Optional[float] = 0):
        """Instantiate the policy on a network
        
        Args:
            network (torch.nn.Module): A Torch network approximating optimal action logits from observation
            actions (list): a list of all possible actions
            obs_dim (int): the length of the flattened observation
            lr (float): learning rate
            weight_decay (float): weight decay
        """
        
        self.logit_network = logit_network
        self.value_network = value_network
        self.actions = actions
        self.action_to_index = {a: i for i,a in enumerate(actions)}
        self.index_to_action = {i: a for i,a in enumerate(actions)}

        dummy_obs = torch.zeros(obs_dim)
        if self.logit_network(dummy_obs).shape[1] != len(self.actions):
            raise ValueError("Network, state, and action shapes do not align")
        
        self.logit_optimizer = torch.optim.Adam(self.logit_network.parameters(),lr=logit_lr,weight_decay=logit_weight_decay)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(),lr=value_lr,weight_decay=value_weight_decay)
 


    def choose_action(self, env: gym.Env, obs):
        """Return an action and a log probability based on the state"""
        logits = self.logit_network(obs)
        dist = torch.distributions.Categorical(logits=logits)

        a = dist.sample()
        return a.item()
    
    

    def logit_loss_fn(self, log_probs, advantage):
        """Calculate loss for logit network"""
        # # TODO: maybe normalize
        # if G.numel() > 1:
        #     norm_G = (G - torch.mean(G))/(torch.std(G) + 1e-8)
        # else:
        #     norm_G = G
            
        loss = -advantage*log_probs
        return torch.mean(loss)
    
    def value_loss_fn(self, G, value):
        """Calculate loss for value network"""
        loss = torch.nn.functional.mse_loss(value,G)
        return loss


    def train(self, env: gym.Env, epochs: Optional[int] = 1,  batch_size: Optional[int] = 1, gamma: Optional[float] = 1.0, entropy: Optional[float] = 0.0, start_seed: Optional[int] = None):
        """Generates a trajectory for each episode and trains the agent on them
        
        Args:
            env (gym.Env): the environment
            epochs (int, optional): number of training batches
            batch_size (int, optional): episodes per training batch
            gamma (int, optional): discount factor
            start_seed (int, optional): starting trajectory seed
            
        Returns:
            info (dict): 
                - 'training_lengths' (list): lengths of each training episode
                - 'training_returns' (list): rewards of each training episode
                - 'training_losses' (list): losses at each training batch
        """
        training_returns = []
        training_lengths = []
        training_losses = []

        losses = []

        if start_seed is not None:
            seed = start_seed

        for i in tqdm(range(epochs),desc="A2C",leave=False,position=1):

            
            T_batch = []
            for i in tqdm(range(batch_size),desc="Batch",leave=False,position=2):
                T = generate_trajectory(env,self,seed=seed)
                training_lengths.append(len(T))

                G = 0
                
                for j,transition in enumerate(reversed(T)):
                    obs,a,next_obs,r,term,trunc,_ = transition
                    a = self.action_to_index[a]

                    G = r + gamma*G

                    logits = self.logit_network(obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(torch.tensor(a,dtype=int))

                    value = self.value_network(obs)

                    advantage = G - value.detach()

                    new_transition = (log_prob.squeeze(),G,value.squeeze(),advantage.squeeze())
                    T_batch.append(new_transition)

                training_returns.append(G)

                if start_seed is not None:
                    seed += 1

            log_prob_batch = torch.stack([t[0] for t in T_batch],dim=0)
            G_batch = torch.tensor([t[1] for t in T_batch], dtype=torch.float)
            value_batch = torch.stack([t[2] for t in T_batch],dim=0)
            advantage_batch = torch.stack([t[3] for t in T_batch],dim=0)

            self.logit_optimizer.zero_grad()
            logit_loss = self.logit_loss_fn(log_prob_batch,advantage_batch)
            logit_loss.backward()
            self.logit_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss = self.value_loss_fn(G_batch,value_batch)
            value_loss.backward()
            self.value_optimizer.step()

            losses.append([logit_loss.item(),value_loss.item()])



        info = {
            "training_lengths": training_lengths,
            "training_returns": training_returns,
            "training_losses": losses,
            "epochs": epochs,
            "batch_size": batch_size,
            "gamma": gamma,
            "start_seed": start_seed
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