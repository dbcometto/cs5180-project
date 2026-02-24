"""Defines the gridworld RL environment, a step towards the ForestWorld environment"""

import gymnasium as gym
import numpy as np
import pygame

from enum import IntEnum
from typing import Optional


class GridWorld(gym.Env):
    """A simple grid world"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,render_mode = None, step_limit = 99, fixed_goal = True):
        """Create the simple grid world"""
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None

        self._step_limit = step_limit
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

        self.fixed_goal = fixed_goal
        if self.fixed_goal:
            self._target_location = np.array([4,5],dtype=int) # Tabular really struggles when goal moves...
        else:
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

    def encode_obs(self,obs):
        """Encode an observation as a tuple"""
        return ((obs["agent"][0],obs["agent"][1]),(obs["target"][0],obs["target"][1]))
    
    def encode_act(self,act):
        """Return the action's integer"""
        return int(act)

    def return_state_list(self):
        """Return a list of all possible states"""
        coords = range(self._map_width)
        return [((r,c),(gr,gc)) for r in coords for c in coords for gr in coords for gc in coords]

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

        if not self.fixed_goal:
            self._target_location = self._agent_location
            while not self._map[tuple(self._target_location)] == self.TILES.CLEAR:
                self._target_location = self.np_random.integers(0,self._map_width,size=(2,),dtype=int)

        self._agent_location = self.np_random.integers(0,self._map_width,size=(2,),dtype=int)
        while np.array_equal(self._target_location,self._agent_location) or not self._map[tuple(self._agent_location)] == self.TILES.CLEAR:
            self._agent_location = self.np_random.integers(0,self._map_width,size=(2,),dtype=int)


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    def step(self,action):
        """Take an action: update the environment and return the observation"""
        self._current_step += 1

        proposed_location = np.clip(self._agent_location + self._action_to_position_change[action],0,self._map_width-1,dtype=int)

        if self._map[tuple(proposed_location)] == self.TILES.CLEAR:
            self._agent_location = proposed_location

        terminated = np.array_equal(self._agent_location,self._target_location)
        truncated = self._current_step >= self._step_limit

        reward = 1 if terminated else -0.01

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    

    def render(self):
        """Return render results of the simple grid world"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        """Render one step of the simple grid world"""
        # Note: pygame is (x,y) not (r,c) but origin is top left
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self._map_width
        ) 

        # Draw Obstacles
        for r,row in enumerate(self._map):
            for c,tile in enumerate(row):
                if tile == self.TILES.OBSTACLE:
                    pygame.draw.rect(
                        canvas,
                        (50, 50, 50),
                        pygame.Rect(
                            pix_square_size * np.array([c,r]),
                            (pix_square_size, pix_square_size),
                        ),
                    )


        # Draw target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location[::-1],
                (pix_square_size, pix_square_size),
            ),
        )

        # Draw agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw gridlines
        for x in range(self._map_width + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # Logic
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def hold_frame(self,duration: Optional[int] = 20000):
        """Hold final frame of render
        
        Args:
            duration (int): milliseconds to hold frame
        """
        if self.render_mode == "human":
            pygame.time.wait(duration)


    def close(self):
        """Close the simple grid world"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()