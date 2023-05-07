from collections import OrderedDict

import gym
from gym import (spaces, logger)
import numpy as np

from protein_folding_environment.base_environment_2d import ProteinFolding2DEnv
from protein_folding_environment.movement_utils import move_to_new_state_2d, move_to_new_state_lrf



class ProteinFoldingLRF2DEnv(ProteinFolding2DEnv):
    def __init__(self, seq):
        super().__init__(seq)
        self.action_space = spaces.Discrete(start=1, n=3)
        self.observation_space = spaces.Box(low=0, high=3,
                                            shape=(len(self.seq) - 2,),
                                            dtype=int)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))

        previous = list(self.state.keys())[-1]
        new_move_direction, next_state = move_to_new_state_lrf(self.move_direction,previous,action)
        self.move_direction = new_move_direction
        idx = len(self.state)
        if next_state is None or next_state in self.state:
            return (None, None, False, False, {})
        self.actions.append(action)
        try:
            self.state.update({next_state: self.seq[idx]})
        except IndexError:
            logger.error('All molecules have been placed! Nothing can be added to the protein chain.')
            raise

        return self.get_observation_info(next_state)

    def reset(self):
        obs = super().reset()
        self.move_direction = (0, 1)
        return obs

    def _get_adjacent_coords(self, coords):
        x, y = coords
        adjacent_coords = {
            0: (x - 1, y),
            1: (x, y - 1),
            2: (x, y + 1),
            3: (x + 1, y),
        }
        return adjacent_coords