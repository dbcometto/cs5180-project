"""A collection of utility functions"""
import gymnasium as gym

from typing import Optional

from treescan.policies import Policy

def generate_trajectory(env: gym.Env, policy: Policy, max_steps=1000, seed: Optional[int]=None):
    """Roll out a trajectory and return the transition table
    
    Args:
        env (gym.Env): the environment
        policy (Policy): the policy
        max_steps (int, optional): the max steps
    """
    T = []

    s,info = env.reset(seed)
    for _ in range(max_steps):
        a = policy.choose_action(env,s)
        next_s,reward,terminated,truncated,info = env.step(a) 
        

        T.append((s,a,next_s,reward,terminated,truncated,info))
        s = next_s

        if terminated or truncated:
            break

    return T