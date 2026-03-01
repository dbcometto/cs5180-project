"""First attempt messing with stuff"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from treescan.policies import RandomTabularPolicy, MCTabularFirstVisitEpsilonControl
from treescan.agents import Agent

from treescan.environments import GridWorld


env = GridWorld(render_mode=None,step_limit=999,fixed_goal=True)
obs,_ = env.reset(seed=2025)
print(obs)


env1 = gym.wrappers.FlattenObservation(
    GridWorld(render_mode=None,step_limit=999,fixed_goal=True)
)
obs,_ = env1.reset(seed=2025)
print(obs)
print(obs.shape)





