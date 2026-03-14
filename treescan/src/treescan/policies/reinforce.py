"""Defines REINFORCE policies"""
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch

from typing import Optional
import pickle
import os
from collections import OrderedDict

from treescan.policies.base import Policy
from treescan.utils import generate_trajectory, generate_trajectory_with_prob

class DiscreteReinforce(Policy):
    """A network policy using the REINFORCE algorithm"""

    def __init__(self, network: torch.nn.Module, actions: list, obs_dim: int, lr: Optional[float] = 0.001, weight_decay: Optional[float] = 0):
        """Instantiate the policy on a network
        
        Args:
            network (torch.nn.Module): A Torch network approximating optimal action probabilities from observation
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
                - 'training_lengths' (list): lengths of each training episode
                - 'training_returns' (list): rewards of each training epiosde
        """
        training_returns = []
        training_lengths = []
        training_losses = []

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


            training_lengths.append(len(T))
            training_returns.append(G)
            training_losses.append(np.mean(losses))


        info = {
            "training_lengths": training_lengths,
            "training_returns": training_returns,
            "training_losses": training_losses,
            "gamma": gamma,
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
        













class DiscreteBatchReinforce(Policy):
    """A network policy using the REINFORCE algorithm on batches of trajectories"""

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
        self.action_to_index = {a: i for i,a in enumerate(actions)}
        self.index_to_action = {i: a for i,a in enumerate(actions)}

        dummy_obs = torch.zeros(obs_dim)
        if self.network(dummy_obs).shape[1] != len(self.actions):
            raise ValueError("Network, state, and action shapes do not align")
        
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=lr,weight_decay=weight_decay)
 


    def choose_action(self, env: gym.Env, obs):
        """Return an action and a log probability based on the state"""
        logits = self.network(obs)
        dist = torch.distributions.Categorical(logits=logits)

        a = dist.sample()
        return a.item()
    
    def choose_action_and_return_prob(self, env: gym.Env, obs):
        """Return an action and a log probability based on the state"""
        logits = self.network(obs)
        dist = torch.distributions.Categorical(logits=logits)

        a = dist.sample()
        log_prob = dist.log_prob(a)
        return a.item(), log_prob
    

    def loss_fn(self, log_probs, G):
        losses = -G*log_probs
        return losses.mean()


    def train(self, env: gym.Env, epochs: Optional[int] = 1,  batch_size: Optional[int] = 1, gamma: Optional[float] = 1.0):
        """Generates a trajectory for each episode and trains the agent on them
        
        Args:
            env (gym.Env): the environment
            epochs (int, optional): number of training batches
            batch_size (int, optional): episodes per training batch
            gamma (int, optional): discount factor
            
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

        for i in tqdm(range(epochs),desc="REINFORCE",leave=False,position=1):

            
            T_batch = []
            for i in tqdm(range(batch_size),desc="Batch",leave=False,position=2):
                T = generate_trajectory(env,self)
                training_lengths.append(len(T))

                G = 0
                
                for j,transition in enumerate(reversed(T)):
                    obs,a,next_obs,r,term,trunc,_, = transition
                    a = self.action_to_index[a]

                    G = r + gamma*G

                    logits = self.network(obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(torch.tensor([a],dtype=int))

                    new_transition = (log_prob,G)
                    T_batch.append(new_transition)

                training_returns.append(G)

            log_prob_batch = torch.stack([t[0] for t in T_batch])
            G_batch = torch.tensor([t[1] for t in T_batch], dtype=float)

            self.optimizer.zero_grad()
            loss = self.loss_fn(log_prob_batch,G_batch)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())



        info = {
            "training_lengths": training_lengths,
            "training_returns": training_returns,
            "training_losses": losses,
            "epochs": epochs,
            "batch_size": batch_size,
            "gamma": gamma,
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
        





class DiscreteBatchReinforceBaseline(Policy):
    """A network policy using the REINFORCE algorithm on batches of trajectories"""

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
        self.action_to_index = {a: i for i,a in enumerate(actions)}
        self.index_to_action = {i: a for i,a in enumerate(actions)}

        dummy_obs = torch.zeros(obs_dim)
        if self.network(dummy_obs).shape[1] != len(self.actions):
            raise ValueError("Network, state, and action shapes do not align")
        
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=lr,weight_decay=weight_decay)
 


    def choose_action(self, env: gym.Env, obs):
        """Return an action and a log probability based on the state"""
        logits = self.network(obs)
        dist = torch.distributions.Categorical(logits=logits)

        a = dist.sample()
        return a.item()
    
    def choose_action_and_return_prob(self, env: gym.Env, obs):
        """Return an action and a log probability based on the state"""
        logits = self.network(obs)
        dist = torch.distributions.Categorical(logits=logits)

        a = dist.sample()
        log_prob = dist.log_prob(a)
        return a.item(), log_prob
    

    def loss_fn(self, log_probs, G, entropy):
        if G.numel() > 1:
            norm_G = (G - torch.mean(G))/(torch.std(G) + 1e-8)
        else:
            norm_G = G
            
        baseline = torch.mean(norm_G)
        advantages = norm_G - baseline
        losses = -advantages*log_probs - entropy
        return torch.mean(losses)


    def train(self, env: gym.Env, epochs: Optional[int] = 1,  batch_size: Optional[int] = 1, gamma: Optional[float] = 1.0, entropy: Optional[float] = 0.0):
        """Generates a trajectory for each episode and trains the agent on them
        
        Args:
            env (gym.Env): the environment
            epochs (int, optional): number of training batches
            batch_size (int, optional): episodes per training batch
            gamma (int, optional): discount factor
            
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

        for i in tqdm(range(epochs),desc="REINFORCE",leave=False,position=1):

            
            T_batch = []
            for i in tqdm(range(batch_size),desc="Batch",leave=False,position=2):
                T = generate_trajectory(env,self)
                training_lengths.append(len(T))

                G = 0
                
                for j,transition in enumerate(reversed(T)):
                    obs,a,next_obs,r,term,trunc,_, = transition
                    a = self.action_to_index[a]

                    G = r + gamma*G

                    logits = self.network(obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(torch.tensor([a],dtype=int))

                    new_transition = (log_prob,G)
                    T_batch.append(new_transition)

                training_returns.append(G)

            log_prob_batch = torch.stack([t[0] for t in T_batch])
            G_batch = torch.tensor([t[1] for t in T_batch], dtype=float)

            self.optimizer.zero_grad()
            loss = self.loss_fn(log_prob_batch,G_batch,entropy)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())



        info = {
            "training_lengths": training_lengths,
            "training_returns": training_returns,
            "training_losses": losses,
            "epochs": epochs,
            "batch_size": batch_size,
            "gamma": gamma,
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