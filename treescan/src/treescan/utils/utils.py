"""A collection of utility functions"""
import gymnasium as gym
from treescan.policies import Policy

def generate_trajectory(env: gym.Env, policy: Policy, max_steps=1000):
    T = []

    s,_ = env.reset()
    for _ in range(max_steps):
        a = policy.choose_action(s)
        next_s,reward,terminated,truncated,_ = env.step(a) 
        done = terminated or truncated

        T.append((s,a,next_s,reward,done))
        s = next_s

        if done:
            break

    return T