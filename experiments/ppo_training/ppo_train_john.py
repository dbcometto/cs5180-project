"""Some testing with pytorch"""

import torch
from tqdm import tqdm
from collections import OrderedDict 

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time

from treescan.policies import DiscreteReinforce, DiscretePPO, DiscreteBatchReinforce, DiscreteBatchReinforceBaseline, DiscreteAdvantageActorCritic, DiscretePPOGAE
from treescan.networks.treeworld import SimpleConvNetwork
from treescan.agents import Agent

from treescan.environments import TreeWorld



        

seed= 2025
train_env = TreeWorld(render_mode=None, step_limit=999, obs_as_tensor=True, use_fixed_map=False)
obs,_ = train_env.reset(seed=seed)
torch.manual_seed(seed)
# print(obs)

obs_channels = obs.shape[0]
# print(obs_channels)
action_list = [a for a in train_env.ACTIONS]
action_dim = len(action_list)



agent_name = "John"
agents_folderpath = "C:/workspace/cs5180-project/experiments/ppo_training/agents"
folderpath = f"{agents_folderpath}/{agent_name}"


start = time.time()

# PPO
logit_network = SimpleConvNetwork(input_channels=obs_channels, output_width=action_dim)
value_network = SimpleConvNetwork(input_channels=obs_channels, output_width=1)
policy = DiscretePPOGAE(logit_network, value_network, actions=action_list, 
                     logit_lr=0.001, value_lr = 0.001, entropy_bonus=0.0,
                     do_normalize_advantage=True)
friend = Agent(policy)



try:
    friend.train(train_env,epochs=100_000, batch_size=32, optimizer_epochs=8, clip_epsilon=0.2, 
                 start_seed=seed, gamma=0.99, lambda_gae=0.95,
                 folderpath = folderpath, checkpoint_interval = 500, resume_epoch=None)

except KeyboardInterrupt:
    print("Interrupting...")

except Exception as e:
    print("Exception occurred")
    raise

finally:
    friend.save(f"{agents_folderpath}/{agent_name}")

print(f"Finished training after {time.time()-start:4.1f}s")