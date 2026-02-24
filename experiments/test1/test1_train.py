"""First attempt messing with stuff"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from treescan.policies import RandomTabularPolicy, MCTabularFirstVisitEpsilonControl
from treescan.agents import Agent

from treescan.environments import GridWorld


# my_env = GridWorld(render_mode="human")

# policy = RandomTabularPolicy(my_env.ACTIONS)
# fred = Agent(policy)


# obs,info = my_env.reset(seed=2025)
# print(obs)

# steps = 10
# for i in range(steps):
#     act = fred.choose_action(my_env,obs)
#     new_obs,_,_,_,_ = my_env.step(act)
#     print(f"{act} - {new_obs}")
# my_env.hold_frame()



# done = False
# obs,info = my_env.reset(seed=2025)
# print(obs)
# while not done:
#     act = fred.choose_action(my_env,obs)
#     new_obs,reward,term,trunc,info = my_env.step(act)
#     done = term or trunc
#     print(f"{act} - {new_obs}")
# my_env.hold_frame()

# obs,info = my_env.reset(seed=2025)
# print(obs)
# print(fred.generate_trajectory(my_env))
# my_env.hold_frame()



train_env = GridWorld(render_mode=None,step_limit=999,fixed_goal=True)

states = train_env.return_state_list()

# fred = Agent(
#     RandomTabularPolicy(train_env.ACTIONS)
# )

new_friend = Agent(
    MCTabularFirstVisitEpsilonControl(train_env.ACTIONS,states)
)


train_results = new_friend.train(train_env,episodes=1000)
# print(train_results)

agents_folderpath = "C:/workspace/cs5180rl-main/cs5180-project/experiments/test1/agents"
# fred.save(f"{agents_folderpath}/fred")
new_friend.save(f"{agents_folderpath}/bob2")




