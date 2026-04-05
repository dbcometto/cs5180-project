"""First messing with stuff"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from treescan.policies import RandomTabularPolicy, MCTabularFirstVisitEpsilonControl
from treescan.agents import Agent

from treescan.environments import TreeWorld

# agents_folderpath = "C:/workspace/cs5180rl-main/cs5180-project/experiments/test1/agents"
# fred = Agent.load(f"{agents_folderpath}/fred")
# bob = Agent.load(f"{agents_folderpath}/bob")
# bob2 = Agent.load(f"{agents_folderpath}/bob2")
# better_bob = Agent.load(f"{agents_folderpath}/better_bob")


demo_env = TreeWorld(render_mode="human")


# bob.generate_trajectory(demo_env,seed=2029)
# bob2.generate_trajectory(demo_env,seed=2029)


bob = Agent(
    RandomTabularPolicy(demo_env.ACTIONS)
)

# state,info = demo_env.reset()
# # print(state.shape)
# # print (demo_env._agent_location)
# obs = demo_env._get_obs()
# print(obs)

bob_test = bob.test(demo_env,episodes=100,runs=5,start_seed=2025)