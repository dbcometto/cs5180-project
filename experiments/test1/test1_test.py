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
# better_bob = Agent.load(f"{agents_folderpath}/better_bob")
bob2 = Agent.load(f"{agents_folderpath}/bob2")
# bob3 = Agent.load(f"{agents_folderpath}/bob3")


test_env = GridWorld(render_mode=None)
bob_test = bob.test(test_env,episodes=100,runs=5,start_seed=2025)
# better_bob_test = better_bob.test(test_env,episodes=100,runs=5,start_seed=2025)
bob2_test = bob2.test(test_env,episodes=100,runs=5,start_seed=2025)
# bob3_test = bob3.test(test_env,episodes=100,runs=5,start_seed=2025)
fred_test = fred.test(test_env,episodes=100,runs=5,start_seed=2025)

bob_test_return_avg = np.mean(bob_test["episode_returns"],axis=0)
bob2_test_return_avg = np.mean(bob2_test["episode_returns"],axis=0)
# bob3_test_return_avg = np.mean(bob3_test["episode_returns"],axis=0)
# better_bob_test_return_avg = np.mean(better_bob_test["episode_returns"],axis=0)
fred_test_return_avg = np.mean(fred_test["episode_returns"],axis=0)


fig,axs = plt.subplots(2,3,figsize=(24,8))
axs[0,0].plot(np.cumsum(bob.training_results["episode_returns"]),label="Bob")
axs[0,0].set_xlabel("Episodes")
axs[0,0].set_ylabel("Total Return")
axs[0,0].set_title("Bob Training Results")
axs[0,0].grid(True)
axs[0,0].legend()

# axs[0,1].plot(np.cumsum(better_bob.training_results["episode_returns"]),label="Better Bob")
# axs[0,1].set_xlabel("Episodes")
# axs[0,1].set_ylabel("Total Return")
# axs[0,1].set_title("Better Bob Training Results")
# axs[0,1].grid(True)
# axs[0,1].legend()

axs[1,0].plot(np.cumsum(bob2.training_results["episode_returns"]),label="Bob2")
axs[1,0].set_xlabel("Episodes")
axs[1,0].set_ylabel("Total Return")
axs[1,0].set_title("Bob2 Training Results")
axs[1,0].grid(True)
axs[1,0].legend()

# axs[1,1].plot(np.cumsum(bob3.training_results["episode_returns"]),label="Bob3")
# axs[1,1].set_xlabel("Episodes")
# axs[1,1].set_ylabel("Total Return")
# axs[1,1].set_title("Bob3 Training Results")
# axs[1,1].grid(True)
# axs[1,1].legend()

axs[0,2].plot(np.cumsum(bob_test_return_avg),label="Bob")
# axs[0,2].plot(np.cumsum(better_bob_test_return_avg),label="Better Bob")
axs[0,2].plot(np.cumsum(bob2_test_return_avg),label="Bob2")
# axs[0,2].plot(np.cumsum(bob3_test_return_avg),label="Bob3")
axs[0,2].plot(np.cumsum(fred_test_return_avg),label="Fred")
axs[0,2].set_xlabel("Episodes")
axs[0,2].set_ylabel("Total Return")
axs[0,2].set_title("Testing Results")
axs[0,2].grid(True)
axs[0,2].legend()



plt.show()