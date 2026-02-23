"""Defines the gridworld RL environment, a step towards the ForestWorld environment"""

import gymnasium as gym
import numpy as np

from enum import IntEnum
from typing import Optional


class GridWorld(gym.Env):
    """A simple grid world"""

    def __init__(self):
        """Create the simple grid world"""

        self._step_limit = 99
        self._current_step = -1

        self._map = np.array([[0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,1,1,1,1,1,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],],dtype=int)
        self._map_width = len(self._map)
        
        self._agent_location = np.array([-1,-1], dtype=int)
        self._target_location = np.array([-1,-1], dtype=int)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0,self._map_width,shape=(2,),dtype=int),
                "target": gym.spaces.Box(0,self._map_width,shape=(2,),dtype=int)
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_position_change = {
            self.ACTIONS.LEFT: np.array([0,-1], dtype=int),
            self.ACTIONS.RIGHT: np.array([0,1], dtype=int),
            self.ACTIONS.UP: np.array([-1,0], dtype=int),
            self.ACTIONS.DOWN: np.array([1,0], dtype=int)
        }

    class ACTIONS(IntEnum):
        LEFT = 0
        RIGHT = 1
        UP = 2
        DOWN = 3

    class TILES(IntEnum):
        CLEAR = 0
        OBSTACLE = 1



    def _get_obs(self):
        """Return an observaton"""
        return {"agent": self._agent_location, "target": self._target_location}


    def _get_info(self):
        """Return extra info"""
        return {
                "current_step": self._current_step,
                "distance": np.linalg.norm(self._target_location-self._agent_location,ord=1),
                }
    


    def reset(self,seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        self._current_step = 0

        self._agent_location = self.np_random.integers(0,self._map_width,size=(2,),dtype=int)
        while not self._map[tuple(self._agent_location)] == self.TILES.CLEAR:
            self._agent_location = self.np_random.integers(0,self._map_width,size=(2,),dtype=int)

        self._target_location = self._agent_location
        while np.array_equal(self._target_location,self._agent_location) or not self._map[tuple(self._target_location)] == self.TILES.CLEAR:
            self._target_location = self.np_random.integers(0,self._map_width,size=(2,),dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    

    def step(self,action):
        """Take an action: update the environment and return the observation"""

        proposed_location = np.clip(self._agent_location + self._action_to_position_change[action],0,self._map_width-1,dtype=int)

        if self._map[tuple(proposed_location)] == self.TILES.CLEAR:
            self._agent_location = proposed_location

        terminated = np.array_equal(self._agent_location,self._target_location)
        truncated = self._current_step >= self._step_limit

        reward = 1 if terminated else -0.01

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    

    def render(self):
        """Render the simple grid world"""
        pass

    def close(self):
        """Close the simple grid world"""
        pass