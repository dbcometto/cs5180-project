"""Defines MC policies"""
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch

from typing import Optional
import pickle
import os

from treescan.policies.base import Policy
from treescan.utils import generate_trajectory_with_prob

class DiscretePPO(Policy):
    """A network policy using the PPO algorithm"""

    def __init__(self, policy_network: torch.nn.Module, value_network: torch.nn.Module, 
                 actions: list, obs_dim: int, 
                 policy_lr: Optional[float] = 0.001, policy_weight_decay: Optional[float] = 0, value_lr: Optional[float] = 0.001, value_weight_decay: Optional[float] = 0):
        """Instantiate the policy on a network
        
        Args:
            network (torch.nn.Module): A Torch network approximating optimal action logits from observation
            actions (list): a list of all possible actions
            obs_dim (int): the length of the flattened observation
            lr (float): learning rate
            weight_decay (float): weight decay
        """
        
        self.policy_network = policy_network
        self.value_network = value_network
        self.actions = actions
        self.action_index = {a: i for i,a in enumerate(actions)}

        dummy_obs = torch.zeros(obs_dim)
        if self.policy_network(dummy_obs).shape[1] != len(self.actions):
            raise ValueError("Network, state, and action shapes do not align")
        
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(),lr=policy_lr,weight_decay=policy_weight_decay)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(),lr=value_lr,weight_decay=value_weight_decay)
 


    def choose_action(self, env: gym.Env, obs):
        """Return an action based on the state"""
        with torch.no_grad():
            probabilites = torch.softmax(self.policy_network(obs).squeeze(0),dim=-1).detach().numpy()
        return env.np_random.choice(self.actions,p=probabilites)
    
    def choose_action_and_return_prob(self, env: gym.Env, obs):
        """Return an action based on the state"""
        with torch.no_grad():
            probabilites = torch.softmax(self.policy_network(obs).squeeze(0),dim=-1).detach().numpy()

        a = env.np_random.choice(self.actions,p=probabilites)
        return a,probabilites[self.action_index[a]]
    


    def train(self, env: gym.Env, epsilon: Optional[float] = 0.01, ppo_epochs: Optional[int] = 1, epochs: Optional[int] = 1, episodes: Optional[int] = 1, gamma: Optional[float] = 1.0):
        """Generates a trajectory for each episode and trains the agent on them
        
        Args:
            env (gym.Env): the environment
            epsilon (float): proximity hyperparameter
            ppo_epochs (int, optional): number of PPO epochs
            epochs (int, optional): number of optimization steps epochs
            episodes (int, optional): number of episodes per epoch (batch size)
            gamma (int, optional): discount factor
            
        Returns:
            info (dict): 
                - 'episode_lengths' (list): lengths of each training episode
                - 'episode_returns' (list): rewards of each training epiosde
        """
        episode_returns = []
        episode_lengths = []
        episode_policy_losses = []
        episode_value_losses = []

        
        for i in tqdm(range(ppo_epochs),desc="PPO Epochs",leave=False,position=1):

            T_batch = []
            for i in tqdm(range(episodes),desc="PPO Episodes",leave=False,position=2):
                T = generate_trajectory_with_prob(env,self)

                G = 0
                for j,transition in enumerate(reversed(T)):
                    obs,a,next_obs,r,term,trunc,_,old_prob = transition
                    G = r + gamma*G
                    
                    value = self.value_network.forward(obs)
                    log_old_prob = np.log(old_prob)

                    advantage = G - value.detach().numpy()

                    new_transition = (obs.detach(),a,log_old_prob,G,advantage)
                    T_batch.append(new_transition)


            obs_batch = torch.stack([t[0].unsqueeze(0) for t in T_batch])
            a_batch = torch.tensor([t[1] for t in T_batch],dtype=torch.int) # TODO: make categorical
            log_old_prob_batch = torch.tensor([t[2] for t in T_batch],dtype=torch.float)
            G_batch = torch.tensor([t[3] for t in T_batch],dtype=torch.float)
            Adv_batch = torch.tensor([t[4] for t in T_batch],dtype=torch.float)

            #TODO: maybe normalize adv_batch

            # policy_losses = []
            # value_losses = []
            for k in tqdm(range(epochs),desc="Optimization Epochs",leave=False,position=3):

                new_value_batch = self.value_network.forward(obs_batch)
                log_new_prob_batch = torch.log(self.policy_network.forward(obs_batch)[a_batch])

                ratio_batch = torch.exp(log_new_prob_batch-log_old_prob_batch)

                boundedAdv_batch = torch.clip(
                    ratio_batch,(1-epsilon),(1+epsilon)
                ) * Adv_batch

                loss_policy_batch = torch.min( 
                    ratio_batch*Adv_batch, boundedAdv_batch
                )

                loss_policy = torch.mean(loss_policy_batch)
                self.policy_optimizer.zero_grad()
                loss_policy.backward()
                self.policy_optimizer.step()

                loss_value = torch.mean((new_value_batch - G_batch)**2)
                self.value_optimizer.zero_grad()
                loss_value.backward()
                self.value_optimizer.step()
        

                # policy_losses.append(loss_policy.item())
                # value_losses.append(loss_value.item())


            episode_lengths.append(len(T))
            episode_returns.append(G)
            episode_policy_losses.append(loss_policy.item())
            episode_value_losses.append(loss_value.item())


        info = {
            "episode_lengths": episode_lengths,
            "episode_returns": episode_returns,
            "episode_policy_losses": episode_policy_losses,
            "episode_value_losses": episode_value_losses,
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