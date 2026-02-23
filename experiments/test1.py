"""First attempt messing with stuff"""

import gymnasium as gym
import numpy as np

from treescan.policies import RandomTabularPolicy
from treescan.agents import Agent

from treescan.environments import GridWorld


env = GridWorld()

obs,info = env.reset(seed=2025)
print(obs)

policy = RandomTabularPolicy(env,env.ACTIONS)
fred = Agent(policy)

steps = 10

for i in range(steps):
    act = fred.choose_action(obs)
    new_obs,_,_,_,_ = env.step(act)

    print(f"{act} - {new_obs}")