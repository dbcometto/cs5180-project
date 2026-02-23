"""Package for CS-5180 Project"""

from gymnasium.envs.registration import register

register(
    id="treescan/GridWorld-v0",
    entry_point="treescan.environments.gridworld:GridWorld",
)