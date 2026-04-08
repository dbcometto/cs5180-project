"""The TreeWorld environment"""

import gymnasium as gym
import numpy as np
import pygame
import torch

from enum import IntEnum
from typing import Optional


class TreeWorld(gym.Env):
    """A grid world that implements tree scanning"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Rendering
    BACKGROUND_COLOR = "#FFFFFF"
    GRID_COLOR = "#3A3A3A"
    AGENT_COLOR = "#7D0ACA"
    GRASS_COLOR = "#B6DD91"
    TREE_COLOR = "#059635"
    ROCK_COLOR = "#584935"
    PANEL_SEPARATION = 15
    UNKNOWN_COLOR = "#8D8D8D"
    ONLY2D_COLOR = "#FFFCDF"
    OCCUPIED_COLOR = "#8D6000"
    SCANNED_COLOR = "#FFC918"

    # Env Params
    RANGE_2D = 4
    RANGE_3D = 3
    NUM_TREES = 6
    NUM_ROCKS = 10

    # Reward Params
    REWARD_STEP = -0.05
    REWARD_EXPLORE_TILE = 0.2
    REWARD_SCAN = -0.7
    REWARD_NEW_FACE = 0.8
    REWARD_NEW_3D = 0.05
    REWARD_FAIL = -20
    REWARD_FAR_END = -5
    REWARD_PERCENT_MAX = 10
    REWARD_COMPLETE = 20
    FAIL_PERCENT = 0.5
    COMPLETE_PERCENT = 0.99
    END_DIST_THRESHOLD = 2

    def __init__(self, 
                 map_rows = 10, map_cols = 15, 
                 step_limit = 99, window_width = 1024, window_height = 512,
                 render_mode = None, 
                 use_fixed_map = True, 
                 flatten_obs = False, one_hot_obs = False, obs_as_tensor = True):
        """Create the tree world"""

        self.do_flatten_obs = flatten_obs
        self.do_one_hot = one_hot_obs
        self.obs_as_tensor = obs_as_tensor

        self._step_limit = step_limit
        self._current_step = -1

        # Establish hidden map
        self.use_fixed_map = use_fixed_map
        if self.use_fixed_map:
            self._hidden_map = np.array([   [0,0,0,0,0,0,0,0,0,0,2,0,0,0,0], # 1: trees, 2: rocks
                                            [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                            [0,0,2,0,0,0,0,2,0,0,1,0,0,0,0],
                                            [0,0,0,0,0,1,0,0,0,0,0,0,2,0,0],
                                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                            [0,0,0,0,0,2,2,0,0,0,0,0,0,0,0],
                                            [0,0,1,0,0,2,2,0,0,0,0,0,0,0,0],
                                            [0,0,0,0,0,0,0,2,0,0,0,1,0,0,0],
                                            [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                            [0,0,0,0,0,0,0,0,0,2,0,0,0,0,0],],dtype=int)
            map_rows = 10 
            map_cols = 15
        else:
            self._hidden_map = None
        
        # Establish knowledge maps
        self._map_rows = map_rows
        self._map_cols = map_cols
        self._reset_knowledge_maps()
        
        
        # Establish agent location
        self._agent_location = np.array([-1,-1], dtype=int)
        self._agent_start_location = np.array([-1,-1], dtype=int)
        

        # Define Gym Spaces
        self.observation_space = gym.spaces.Box(0,1,shape=(11,map_rows,map_cols))
        self.action_space = gym.spaces.Discrete(7)

        # Define Movement Array
        self._action_to_position_change = {
            self.ACTIONS.WAIT: np.array([0,0], dtype=int),
            self.ACTIONS.LEFT: np.array([0,-1], dtype=int),
            self.ACTIONS.RIGHT: np.array([0,1], dtype=int),
            self.ACTIONS.UP: np.array([-1,0], dtype=int),
            self.ACTIONS.DOWN: np.array([1,0], dtype=int),
            self.ACTIONS.SCAN: np.array([0,0], dtype=int),
            self.ACTIONS.END: np.array([0,0], dtype=int),
        }


        # Set up Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window_width = window_width
        self.window_height = window_height
        self.info_width_start = self.window_width/2 + self.PANEL_SEPARATION

        self.panel_height = self.window_height
        self.panel_width = int((self.window_width-self.PANEL_SEPARATION)/2)

        # if self._map_rows < self._map_cols:
        #     self.hidden_height = int(self.window_height*self._map_rows/self._map_cols)
        #     self.hidden_width = int(self.window_width/2-self.PANEL_SEPARATION)
        # else:
        #     self.hidden_height = int(self.window_height)
        #     self.hidden_width = int((self.window_width/2-self.PANEL_SEPARATION)*self._map_rows/self._map_cols)

        self.pix_square_size = min(self.panel_width/self._map_cols, self.panel_height/self._map_rows)

        

        self.window = None
        self.clock = None
        
        # if self._map_rows > self._map_cols


    # Helper Classes
    class ACTIONS(IntEnum):
        WAIT = 0
        LEFT = 1
        RIGHT = 2
        UP = 3
        DOWN = 4
        SCAN = 5
        END = 6

    class TILES(IntEnum):
        CLEAR = 0
        TREE = 1
        ROCK = 2

    def _generate_map(self, rows, cols):
        """Randomly generate a hidden map"""
        map = np.zeros((rows,cols),dtype=int)
        indices = np.arange(rows*cols)
        total_placements = self.NUM_TREES + self.NUM_ROCKS
        chosen = self.np_random.choice(indices,size=total_placements,replace=False)

        tree_idxs = chosen[:self.NUM_TREES]
        rock_idxs = chosen[self.NUM_TREES:]

        flat = map.flat
        flat[tree_idxs] = self.TILES.TREE
        flat[rock_idxs] = self.TILES.ROCK

        return map



    # Tabular Helpers

    def encode_act(self,act):
        """Return the action's integer"""
        return int(act)

    def encode_obs(self,obs):
        """Cannot encode an observation as a tuple"""
        raise Exception("TreeWorld is Non-Tabular")
    
    def return_state_list(self):
        """Cannot eturn a list of all possible states"""
        raise Exception("TreeWorld is Non-Tabular")



    # Obs/Info Helpers

    def _get_obs(self):
        """Return an observaton"""

        obs = np.stack([
            self._map_agent_location,
            self._map_start_location,
            self._map_2dknown,
            self._map_3dknown,
            self._map_occupied,
            self._map_rocks,
            self._map_trees,
            self._map_scanned_left,
            self._map_scanned_right,
            self._map_scanned_up,
            self._map_scanned_down,
        ], axis=0)

        if self.obs_as_tensor:
            obs = torch.tensor(obs,dtype=torch.float32)

        # if self.do_one_hot:
        #     obs = self._obs_to_one_hot(obs)

        # if self.do_flatten_obs:
        #     obs = self._flatten_obs(obs)

        return obs

    def _get_info(self):
        """Return extra info"""
        return {
                "current_step": self._current_step,
                }
    
    # def _obs_to_one_hot(self,obs):
    #     agent_pos = np.zeros_like(self._map)
    #     agent_pos[obs["agent"]] = 1

    #     goal_pos = np.zeros_like(self._map)
    #     goal_pos[obs["target"]] = 1

    #     return {"agent": agent_pos, "target": goal_pos}

    # def _flatten_obs(self,obs):
    #     """Flatten the obs into a list"""
    #     return torch.tensor([x for arr in obs.values() for x in arr.ravel()],dtype=torch.float)




    # Knowledge Map Helpers   

    def _reset_knowledge_maps(self):
        """Zero out all knowledge maps"""
        self._map_agent_location = np.zeros((self._map_rows,self._map_cols), dtype=int)     # all zeros except for a 1 at agent location
        self._map_2dknown = np.zeros((self._map_rows,self._map_cols), dtype=int)            # 0 where unknown, 1 where 2D known
        self._map_3dknown = np.zeros((self._map_rows,self._map_cols), dtype=int)            # 0 where unknown, 1 where 3D known
        self._map_occupied = np.zeros((self._map_rows,self._map_cols), dtype=int)           # 0 where free, 1 where occupied
        self._map_rocks = np.zeros((self._map_rows,self._map_cols), dtype=int)              # 1 where known rock, 0 elsewhere
        self._map_trees = np.zeros((self._map_rows,self._map_cols), dtype=int)              # 1 where known tree, 0 elsewhere
        self._map_scanned_left = np.zeros((self._map_rows,self._map_cols), dtype=int)      # 1 where tree is scanned, 0 elsewhere
        self._map_scanned_right = np.zeros((self._map_rows,self._map_cols), dtype=int)      # 1 where tree is scanned, 0 elsewhere
        self._map_scanned_up = np.zeros((self._map_rows,self._map_cols), dtype=int)      # 1 where tree is scanned, 0 elsewhere
        self._map_scanned_down = np.zeros((self._map_rows,self._map_cols), dtype=int)      # 1 where tree is scanned, 0 elsewhere

        self._map_start_location = np.zeros((self._map_rows,self._map_cols), dtype=int)     # all zeros except for a 1 at agent start location
        try:
            self._map_start_location[tuple(self._agent_start_location)] = 1
        except:
            pass


    def _update_map_agent_location(self):
        """Update the agent location map with the agent's current location"""
        self._map_agent_location = np.zeros((self._map_rows,self._map_cols), dtype=int)
        try:
            self._map_agent_location[tuple(self._agent_location)] = 1
        except:
            pass


    def _update_map_2dscan(self):
        """Update the agent based off of a 2D scan, return the number of new tiles"""
        distances = self._calc_distance_grid()
        newtiles = 0
        for row in range(self._map_rows):
            for col in range(self._map_cols):
                dist = distances[row,col]
                if dist < self.RANGE_2D:
                    if self._check_visibility(row,col):
                        if self._map_2dknown[row,col] == 0:
                            self._map_2dknown[row,col] = 1
                            newtiles += 1
                        self._map_occupied[row,col] = 0 if self._hidden_map[row,col] == self.TILES.CLEAR else 1
        return newtiles


    def _update_map_3dscan(self):
        """Update the agent based off of a 2D scan, and return new tiles and new sides"""
        distances = self._calc_distance_grid()
        newtiles = 0
        newsides = 0
        for row in range(self._map_rows):
            for col in range(self._map_cols):
                dist = distances[row,col]
                if dist < self.RANGE_3D:
                    if self._check_visibility(row,col):
                        if self._map_3dknown[row,col] == 0:
                            self._map_3dknown[row,col] = 1
                            newtiles += 1

                        self._map_occupied[row,col] = 0 if self._hidden_map[row,col] == self.TILES.CLEAR else 1
                        self._map_rocks[row,col] = 1 if self._hidden_map[row,col] == self.TILES.ROCK else 0
                        self._map_trees[row,col] = 1 if self._hidden_map[row,col] == self.TILES.TREE else 0

                        # Check visibility
                        if self._hidden_map[row,col] == self.TILES.TREE:
                            agent_row,agent_col = self._agent_location
                            
                            if agent_col < col:
                                if self._map_scanned_left[row,col] == 0:
                                    newsides += 1
                                self._map_scanned_left[row,col] = 1
                            if agent_col > col:
                                if self._map_scanned_right[row,col] == 0:
                                    newsides += 1
                                self._map_scanned_right[row,col] = 1

                            if agent_row < row:
                                if self._map_scanned_up[row,col] == 0:
                                    newsides += 1
                                self._map_scanned_up[row,col] = 1
                            if agent_row > row:
                                if self._map_scanned_down[row,col] == 0:
                                    newsides += 1
                                self._map_scanned_down[row,col] = 1

        return newtiles, newsides


    def _calc_distance_grid(self):
        """Assign a 2-norm distance to every grid location"""
        distances = np.zeros((self._map_rows,self._map_cols))
        for row in range(self._map_rows):
            for col in range(self._map_cols):
                loc = np.array([row,col])
                dist = np.linalg.norm(self._agent_location-loc)
                distances[row,col] = dist

        return distances

    def _find_cell_path(self,r_target,c_target):
        """Uses the Bresenham Algorithm to determine which cells are viewed from the agent's location towards the target cell"""
        r, c =  self._agent_location
        dr = abs(r_target - r)
        dc = abs(c_target - c)

        r_step = 1 if r_target > r else -1
        c_step = 1 if c_target > c else -1

        cells = [[r,c]]
        if dc >= dr:
            p = 2*dr - dc
            while c != c_target:
                c += c_step
                if p < 0:
                    p += 2*dr
                else:
                    r += r_step
                    p += 2*(dr-dc)
                cells.append([r,c])
        else:
            p = 2*dc - dr
            while r != r_target:
                r += r_step
                if p < 0:
                    p += 2*dc
                else:
                    c += c_step
                    p += 2*(dc-dr)
                cells.append([r,c])
            

        return cells
    
    def _check_visibility(self,r,c):
        """Check whether a cell is visible from the agent's location"""
        cells_viewed = self._find_cell_path(r,c)

        visible = True
        for cell in cells_viewed[1:-1]:
            if self._hidden_map[*cell] != self.TILES.CLEAR:
                visible = False

        return visible
    
    



    # Reset and helpers

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        self._current_step = 0

        if not self.use_fixed_map:
            self._hidden_map = self._generate_map(self._map_rows, self._map_cols)

        # Choose a new starting location
        location_chosen = False
        while not location_chosen or not self._hidden_map[tuple(self._agent_start_location)] == self.TILES.CLEAR:
            self._agent_start_location = self._get_random_location()
            if not location_chosen:
                location_chosen = True

        self._agent_location = self._agent_start_location.copy()
        
        # Reset maps
        self._reset_knowledge_maps()
        self._update_map_agent_location()


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _get_random_location(self):
        return self.np_random.integers([0,0],[self._map_rows-1,self._map_cols-1],size=(2,),dtype=int)
    



    # Step and Helpers

    def step(self,action):
        """Take an action: update the environment and return the observation"""
        self._current_step += 1
        reward = 0 # updated at each phase

        # Phase 1: Move
        proposed_location = np.clip(self._agent_location + self._action_to_position_change[action],[0,0],[self._map_rows-1,self._map_cols-1],dtype=int)

        if self._hidden_map[tuple(proposed_location)] == self.TILES.CLEAR:
            self._agent_location = proposed_location

        self._update_map_agent_location()

        reward += self.REWARD_STEP # Penalty for each step


        # Phase 2: Scan
        newtiles = self._update_map_2dscan()
        reward += self.REWARD_EXPLORE_TILE*newtiles # Reward for exploring new tiles

        if action == self.ACTIONS.SCAN:
            new3dtiles, newfaces = self._update_map_3dscan()
            reward += self.REWARD_SCAN + self.REWARD_NEW_3D*new3dtiles + self.REWARD_NEW_FACE*newfaces # Reward for 3D exploration and scanning tree faces
        

        # Phase 3: Terminate
        terminated = True if action == self.ACTIONS.END else False
        truncated = True if self._current_step >= self._step_limit else False

        if terminated or truncated:
            percent = self._calc_percent_complete()

            if percent < self.FAIL_PERCENT: # Piecewise reward for completion
                reward += self.REWARD_FAIL
            elif percent < self.COMPLETE_PERCENT:
                reward += self.REWARD_PERCENT_MAX*((percent-self.FAIL_PERCENT)/(self.COMPLETE_PERCENT-self.FAIL_PERCENT))**4  # Gives more reward as closer to percent complete, *0 at fail and *1 at complete
            elif percent >= self.COMPLETE_PERCENT:
                reward += self.REWARD_COMPLETE

            if np.linalg.norm(self._agent_location-self._agent_start_location) > self.END_DIST_THRESHOLD: # If too far at end, give penalty
                reward += self.REWARD_FAR_END


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    

    def _calc_percent_complete(self):
        """Calculate the percent (fraction) of tree sides scanned"""
        num_sides = 0
        num_scans = 0
        for row in range(self._map_rows):
            for col in range(self._map_cols):
                if self._hidden_map[row,col] == self.TILES.TREE:
                    num_sides += 4
                    num_scans += self._map_scanned_left[row,col] + self._map_scanned_right[row,col] + self._map_scanned_up[row,col] + self._map_scanned_down[row,col]

        return num_scans/num_sides








    # Render and Helpers

    def render(self):
        """Return render results of the treeworld"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        """Render one step of the treeworld"""
        # Note: pygame is (x,y) not (r,c) but origin is top left
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill(color=self.BACKGROUND_COLOR)
        

        # =========== Render Hidden State ============== # 

        # Draw Grass
        pygame.draw.rect(
            canvas,
            self.GRASS_COLOR,
            pygame.Rect(
                [0,0],
                [self.pix_square_size*self._map_cols, self.pix_square_size*self._map_rows]
            )
        )

        # Draw Obstacles
        for r,row in enumerate(self._hidden_map):
            for c,tile in enumerate(row):

                if tile == self.TILES.TREE:
                    pygame.draw.rect(
                        canvas,
                        self.TREE_COLOR,
                        pygame.Rect(
                            self.pix_square_size * np.array([c,r]),
                            (self.pix_square_size, self.pix_square_size),
                        ),
                    )

                if tile == self.TILES.ROCK:
                    pygame.draw.rect(
                        canvas,
                        self.ROCK_COLOR,
                        pygame.Rect(
                            self.pix_square_size * np.array([c,r]),
                            (self.pix_square_size, self.pix_square_size),
                        ),
                    )

        # Draw agent
        pygame.draw.circle(
            canvas,
            self.AGENT_COLOR,
            (self._agent_location[::-1] + 0.5) * self.pix_square_size,
            self.pix_square_size / 3,
        )

        # Draw gridlines
        for col in range(self._map_cols+1):
            pygame.draw.line(
                canvas,
                self.GRID_COLOR,
                (self.pix_square_size * col, 0),
                (self.pix_square_size * col, self.pix_square_size * self._map_rows),
                width=3,
            )
        for row in range(self._map_rows+1):
            pygame.draw.line(
                canvas,
                self.GRID_COLOR,
                (0, self.pix_square_size * row),
                (self.pix_square_size * self._map_cols, self.pix_square_size * row),
                width=3,
            )


         # =========== Render Info State ============== # 

        # # Draw Grass
        # pygame.draw.rect(
        #     canvas,
        #     self.GRASS_COLOR,
        #     pygame.Rect(
        #         [0,0],
        #         [self.pix_square_size*self._map_cols, self.pix_square_size*self._map_rows]
        #     )
        # )

        # Draw Tiles
        for r in range(self._map_rows):
            for c in range(self._map_cols):
                
                if self._map_2dknown[r,c] == 0:
                    pygame.draw.rect(
                        canvas,
                        self.UNKNOWN_COLOR,
                        pygame.Rect(
                            np.array([self.info_width_start,0]) + self.pix_square_size * np.array([c,r]),
                            (self.pix_square_size, self.pix_square_size),
                        ),
                    )
                else:
                    if self._map_3dknown[r,c] == 1:
                        
                        if self._map_occupied[r,c] == 0:
                            pygame.draw.rect(
                                canvas,
                                self.GRASS_COLOR,
                                pygame.Rect(
                                    np.array([self.info_width_start,0]) + self.pix_square_size * np.array([c,r]),
                                    (self.pix_square_size, self.pix_square_size),
                                ),
                            )
                        
                        elif self._map_rocks[r,c] == 1:
                            pygame.draw.rect(
                                canvas,
                                self.ROCK_COLOR,
                                pygame.Rect(
                                    np.array([self.info_width_start,0]) + self.pix_square_size * np.array([c,r]),
                                    (self.pix_square_size, self.pix_square_size),
                                ),
                            )

                        elif self._map_trees[r,c] == 1:

                            pygame.draw.rect(
                                canvas,
                                self.TREE_COLOR,
                                pygame.Rect(
                                    np.array([self.info_width_start,0]) + self.pix_square_size*np.array([c,r]),
                                    (self.pix_square_size, self.pix_square_size),
                                ),
                            )

                            if self._map_scanned_up[r,c] == 1:
                                pygame.draw.circle(
                                    canvas,
                                    self.SCANNED_COLOR,
                                    np.array([self.info_width_start,-0.25* self.pix_square_size]) + (np.array([c,r]) + 0.5) * self.pix_square_size,
                                    self.pix_square_size / 8,
                                )
                            
                            if self._map_scanned_down[r,c] == 1:
                                pygame.draw.circle(
                                    canvas,
                                    self.SCANNED_COLOR,
                                    np.array([self.info_width_start,0.25* self.pix_square_size]) + (np.array([c,r]) + 0.5) * self.pix_square_size,
                                    self.pix_square_size / 8,
                                )

                            if self._map_scanned_left[r,c] == 1:
                                pygame.draw.circle(
                                    canvas,
                                    self.SCANNED_COLOR,
                                    np.array([self.info_width_start-0.25* self.pix_square_size,0]) + (np.array([c,r]) + 0.5) * self.pix_square_size,
                                    self.pix_square_size / 8,
                                )
                                
                            if self._map_scanned_right[r,c] == 1:
                                pygame.draw.circle(
                                    canvas,
                                    self.SCANNED_COLOR,
                                    np.array([self.info_width_start+0.25* self.pix_square_size,0]) + (np.array([c,r]) + 0.5) * self.pix_square_size,
                                    self.pix_square_size / 8,
                                )



                    else:

                        if self._map_occupied[r,c] == 1:
                            pygame.draw.rect(
                                canvas,
                                self.OCCUPIED_COLOR,
                                pygame.Rect(
                                    np.array([self.info_width_start,0]) + self.pix_square_size * np.array([c,r]),
                                    (self.pix_square_size, self.pix_square_size),
                                ),
                            )

                        else:
                            pygame.draw.rect(
                                canvas,
                                self.ONLY2D_COLOR,
                                pygame.Rect(
                                    np.array([self.info_width_start,0]) + self.pix_square_size * np.array([c,r]),
                                    (self.pix_square_size, self.pix_square_size),
                                ),
                            )

                # if tile == self.TILES.TREE:
                #     pygame.draw.rect(
                #         canvas,
                #         self.TREE_COLOR,
                #         pygame.Rect(
                #             self.pix_square_size * np.array([c,r]),
                #             (self.pix_square_size, self.pix_square_size),
                #         ),
                #     )

                # if tile == self.TILES.ROCK:
                #     pygame.draw.rect(
                #         canvas,
                #         self.ROCK_COLOR,
                #         pygame.Rect(
                #             self.pix_square_size * np.array([c,r]),
                #             (self.pix_square_size, self.pix_square_size),
                #         ),
                #     )

        # Draw agent
        pygame.draw.circle(
            canvas,
            self.AGENT_COLOR,
            np.array([self.info_width_start,0]) + (self._agent_location[::-1] + 0.5) * self.pix_square_size,
            self.pix_square_size / 3,
        )

        # Draw gridlines
        for col in range(self._map_cols+1):
            pygame.draw.line(
                canvas,
                self.GRID_COLOR,
                (self.info_width_start + self.pix_square_size * col, 0),
                (self.info_width_start + self.pix_square_size * col, self.pix_square_size * self._map_rows),
                width=3,
            )
        for row in range(self._map_rows+1):
            pygame.draw.line(
                canvas,
                self.GRID_COLOR,
                (self.info_width_start, self.pix_square_size * row),
                (self.info_width_start + self.pix_square_size * self._map_cols, self.pix_square_size * row),
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