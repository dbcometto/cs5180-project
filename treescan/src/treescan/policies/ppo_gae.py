"""Defines PPO policies"""
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



class DiscretePPOGAE(Policy):
    """A network policy using the PPO algorithm"""

    def __init__(self, logit_network: torch.nn.Module, value_network: torch.nn.Module, 
                 actions: list, obs_dim=None, 
                 logit_lr: Optional[float] = 0.001, value_lr: Optional[float] = 0.001, 
                 logit_weight_decay: Optional[float] = 0, value_weight_decay: Optional[float] = 0,
                 entropy_bonus: Optional[float] = 0.01, do_normalize_advantage: Optional[bool] = True):
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

        self.entropy_bonus = entropy_bonus
        self.do_normalize_advantage = do_normalize_advantage

        # dummy_obs = torch.zeros(obs_dim)
        # if self.logit_network(dummy_obs).shape[1] != len(self.actions):
        #     raise ValueError("Network, state, and action shapes do not align")
        
        self.logit_optimizer = torch.optim.Adam(self.logit_network.parameters(),lr=logit_lr,weight_decay=logit_weight_decay)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(),lr=value_lr,weight_decay=value_weight_decay)
 


    def choose_action(self, env: gym.Env, obs: torch.tensor):
        """Return an action and a log probability based on the state"""
        logits = self.logit_network(obs)
        dist = torch.distributions.Categorical(logits=logits)

        a = dist.sample()
        return a.item()



    def logit_loss_fn(self, advantage, old_log_probs, new_log_probs, epsilon, entropy):
        """Calculate loss for logit network"""
        ratio = torch.exp(new_log_probs-old_log_probs)
        g_val = torch.clip(ratio,1-epsilon,1+epsilon)

        loss = -torch.min(ratio*advantage, g_val*advantage) 
            
        return torch.mean(loss) - self.entropy_bonus*entropy
    


    def value_loss_fn(self, G, value):
        """Calculate loss for value network"""
        loss = torch.nn.functional.mse_loss(value,G)
        return loss


    def train(self, env: gym.Env, epochs: Optional[int] = 1,  
              batch_size: Optional[int] = 1, optimizer_epochs: Optional[int]=1, lambda_gae: Optional[float]=1,
              clip_epsilon: Optional[float]=0.2, gamma: Optional[float] = 1.0, start_seed: Optional[int] = None,
              resume_epoch: Optional[int] = None, folderpath: Optional[str] = None, checkpoint_interval: Optional[int] = 100):
        """Generates a trajectory for each episode and trains the agent on them
        
        Args:
            env (gym.Env): the environment
            epochs (int, optional): number of training batches
            batch_size (int, optional): episodes per training batch
            optimizer_epochs (int, optional): Number of optimizer steps per batch
            clip_epsilon (float, optional): PPO surrogate loss clip value
            gamma (int, optional): discount factor
            start_seed (int, optional): starting trajectory seed
            
            
        Returns:
            info (dict): 
                - 'training_lengths' (list): lengths of each training episode
                - 'training_returns' (list): rewards of each training episode
                - 'training_losses' (list): losses at each training batch
        """
        info = {}

        if resume_epoch is not None and folderpath is not None:
            try:
                load_epoch, seed, training_returns, training_lengths, losses, training_mcreturns, entropies = self.load_checkpoint(folderpath,resume_epoch)
                print(f"Resuming from Epoch {load_epoch}")
            except Exception as e:
                raise RuntimeError(f"Failed to resume from checkpoint at Epoch {resume_epoch}") from e

        else:
            training_returns = []
            training_mcreturns = []
            training_lengths = []
            losses = []
            entropies = []

            seed = start_seed

        start_epoch = resume_epoch+1 if resume_epoch is not None else 0
        epoch = start_epoch

        try:
            pbarepoch = tqdm(range(start_epoch,epochs),desc="PPO GAE",leave=False,position=1)
            for epoch in pbarepoch:

                # Create batch of data
                T_batch = []
                total_length = 0
                for i in tqdm(range(batch_size),desc="Batch",leave=False,position=2):
                    if seed is not None:
                        torch.manual_seed(seed)

                    T = generate_trajectory(env,self,seed=seed)
                    training_lengths.append(len(T))
                    total_length += len(T)

                    # Calculate values
                    G = 0
                    a_gae = 0
                    for j,transition in enumerate(reversed(T)):
                        obs,a,next_obs,r,term,trunc,_ = transition

                        # Grab data
                        a = self.action_to_index[a]
                        with torch.no_grad(): 
                            logits = self.logit_network(obs).squeeze(0) # Otherwise is [1, A] and broadcasting is wrong!!
                            dist = torch.distributions.Categorical(logits=logits)
                            log_prob = dist.log_prob(torch.tensor(a,dtype=torch.int64))

                            value = self.value_network(obs).squeeze(0).squeeze(-1) # Squeezing to get rid of batch dimension after network and make scalar
                            next_value = self.value_network(next_obs).squeeze(0).squeeze(-1) if not term else torch.zeros_like(value)

                        # Calculations
                        G = r + gamma*G*(1-term)
                        delta = r + gamma*next_value*(1-term) - value
                        a_gae = delta + gamma * lambda_gae * a_gae*(1-term)
                        G_gae = a_gae + value

                        # print(f"obs: {obs}")
                        # print(f"term:{term} | r:{r} | value:{value} | next_value:{next_value} | delta:{delta} | a_gae:{a_gae} | G_gae: {G_gae} | G_MC: {G}")
                        # print(f"logits: {logits[0].detach()}")
                        # print(f"logit shape: {logits.detach().shape}")

                        # Save transition
                        new_transition = (obs.detach(),
                                          a,
                                          G_gae.detach(),
                                          log_prob.detach(),
                                          value.detach(),
                                          a_gae.detach())
                        T_batch.append(new_transition)

                    # Save Info
                    training_returns.append(G_gae.item())
                    training_mcreturns.append(G)

                    # Update PBar
                    avg_length = total_length/batch_size

                    # Seed
                    if start_seed is not None:
                        seed += 1

                T.clear()

                # Create Tensors
                obs_batch = torch.stack([t[0] for t in T_batch],dim=0).detach()
                a_batch = torch.tensor([t[1] for t in T_batch],dtype=torch.int64).detach()
                G_batch = torch.tensor([t[2] for t in T_batch], dtype=torch.float).detach()
                old_log_prob_batch = torch.stack([t[3] for t in T_batch],dim=0).detach()
                # value_batch = torch.stack([t[4] for t in T_batch],dim=0).detach()
                adv_batch = torch.stack([t[5] for t in T_batch],dim=0).detach()

                T_batch.clear()


                # Normalize
                adv_mean = adv_batch.mean().item()
                adv_std = adv_batch.std().item()
                if adv_batch.shape[0] > 1 and self.do_normalize_advantage:
                    adv_batch = (adv_batch - torch.mean(adv_batch))/(torch.std(adv_batch) + 1e-8)


                # Optimize networks
                for optim_step in tqdm(range(optimizer_epochs),desc="Optimizer Step",leave=False,position=2): 

                    # New values
                    new_logits = self.logit_network(obs_batch)
                    dist = torch.distributions.Categorical(logits=new_logits)
                    new_log_prob_batch = dist.log_prob(a_batch)
                    # print(f"adv*new_log_prob max: {torch.max((adv_batch * new_log_prob_batch))} | min: {torch.min((adv_batch * new_log_prob_batch))}")

                    new_value_batch = self.value_network(obs_batch).squeeze(-1) # make it [batch,] or scalar per batch

                    
                    # print(f"new_log_prob_batch shape: {new_log_prob_batch.detach().shape}")
                    # print(f"old_log_prob_batch shape: {old_log_prob_batch.detach().shape}")
                    # print(f"new_value_batch shape: {new_value_batch.detach().shape}")

                    # Optimizer steps
                    self.logit_optimizer.zero_grad()
                    entropy = torch.mean(dist.entropy())
                    # print(f"entropy: {entropy.item()} | advantage max: {torch.max(adv_batch)}, min:  {torch.min(adv_batch)}")
                    logit_loss = self.logit_loss_fn(adv_batch,old_log_prob_batch,new_log_prob_batch,clip_epsilon,entropy)
                    logit_loss.backward()
                    self.logit_optimizer.step()

                    self.value_optimizer.zero_grad()
                    value_loss = self.value_loss_fn(G_batch,new_value_batch)
                    value_loss.backward()
                    self.value_optimizer.step()


                # append final loss
                losses.append([logit_loss.item(),value_loss.item()])
                entropies.append(entropy.item())

                # Update PBar
                pbarepoch.set_postfix_str(f"Len:{avg_length:5.1f}|Entrpy: {entropy:5.3f}|Adv mn:{adv_mean:5.2f}/std:{adv_std:5.2f}")


                # Save intermediate checkpoint
                if folderpath is not None:
                    if epoch % checkpoint_interval == 0:
                        self.save_checkpoint(
                            folderpath,
                            epoch,
                            seed,
                            training_returns,
                            training_lengths,
                            losses,
                            training_mcreturns,
                            entropies
                        )

            # Save final checkpoint
            if folderpath is not None:
                self.save_checkpoint(
                                folderpath,
                                epoch,
                                seed,
                                training_returns,
                                training_lengths,
                                losses,
                                training_mcreturns,
                                entropies
                            )
                print(f"Finished training and saved checkpoint at epoch {epoch} to file at {folderpath}")



        # Handle Interruptions
        except KeyboardInterrupt:
            if folderpath is not None:
                print(f"Interrupted at epoch {epoch} and saved checkpoint to file at {folderpath}")
            else:
                print(f"Interrupted at epoch {epoch} and not saved (no filepath provided)")
        
        except Exception as e:
            if folderpath is not None:
                print(f"Exception at epoch {epoch} and saved checkpoint to file at {folderpath} | Exception: {e}")
            else:
                print(f"Exception at epoch {epoch} and not saved (no filepath provided)  | Exception: {e}")
            raise

        finally:
            if folderpath is not None:
                self.save_checkpoint(
                                folderpath,
                                epoch,
                                seed,
                                training_returns,
                                training_lengths,
                                losses,
                                training_mcreturns,
                                entropies
                            )
                
            info = {
                "training_lengths": training_lengths,
                "training_returns": training_returns,
                "training_mcreturns": training_mcreturns,
                "training_losses": losses,
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer_epochs": optimizer_epochs,
                "clip_epsilon": clip_epsilon,
                "gamma": gamma,
                "start_seed": start_seed,
                "epoch_completed": epoch,
                "entropies": entropies,
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
        




    @classmethod
    def load_from_checkpoint(cls,folderpath,checkpoint_epoch):
        """Load the policy from a checkpoint file"""

        with open(f"{folderpath}/policy.pkl","rb") as file:
            policy = pickle.load(file)
        
        policy.load_checkpoint(folderpath,checkpoint_epoch)

        return policy
        



    
    def save_checkpoint(self, folderpath, epoch, seed, training_returns, training_lengths, losses, training_mcreturns, entropies):
        """Save a training checkpoint to a file"""
        path = f"{folderpath}/checkpoints"
        os.makedirs(path, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "seed": seed,
            "logit_network": self.logit_network.state_dict(),
            "value_network": self.value_network.state_dict(),
            "logit_optimizer": self.logit_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "training_returns": training_returns,
            "training_mcreturns": training_mcreturns,
            "training_lengths": training_lengths,
            "losses": losses,
            "rng_state": torch.get_rng_state(),
            "entropies": entropies
        }

        torch.save(checkpoint, f"{path}/ckpt_{epoch}.pt")






    def load_checkpoint(self, folderpath, epoch):
        """Load a training checkpoint from a file"""
        path = f"{folderpath}/checkpoints/ckpt_{epoch}.pt"
        checkpoint = torch.load(path, weights_only=False)

        self.logit_network.load_state_dict(checkpoint["logit_network"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.logit_optimizer.load_state_dict(checkpoint["logit_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])

        if "rng_state" in checkpoint.keys():
            torch.set_rng_state(checkpoint["rng_state"])

        return checkpoint["epoch"], checkpoint["seed"], checkpoint["training_returns"], checkpoint["training_lengths"], checkpoint["losses"], checkpoint["training_mcreturns"], checkpoint["entropies"]