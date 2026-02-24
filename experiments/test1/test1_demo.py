"""First messing with stuff"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from treescan.policies import RandomTabularPolicy, MCTabularFirstVisitEpsilonControl
from treescan.agents import Agent

from treescan.environments import GridWorld

agents_folderpath = "C:/workspace/cs5180rl-main/cs5180-project/experiments/test1/agents"
fred = Agent.load(f"{agents_folderpath}/fred")
bob = Agent.load(f"{agents_folderpath}/bob")
bob2 = Agent.load(f"{agents_folderpath}/bob2")
# better_bob = Agent.load(f"{agents_folderpath}/better_bob")


demo_env = GridWorld(render_mode="human")


# bob.generate_trajectory(demo_env,seed=2029)
bob2.generate_trajectory(demo_env,seed=2029)
