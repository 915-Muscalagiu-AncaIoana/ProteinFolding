from collections import OrderedDict
import gym
from gym import (spaces,logger)
import numpy as np

from protein_folding_environment.base_environment import ProteinFoldingBaseEnv
from protein_folding_environment.movement_utils import move_to_new_state_3d
from utilities.plotting_utils import plot_3D_foleded_protein

class ProteinFolding3DEnv(ProteinFoldingBaseEnv):
    def __init__(self, seq):
        super().__init__(seq)
        self.reset()
        self.action_space = spaces.Discrete(start=1, n=6)
        self.observation_space = spaces.Box(low=0, high=5,
                                            shape=(len(self.seq) - 2,),
                                            dtype=int)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))

        self.last_action = action

        previous = list(self.state.keys())[-1]
        previous2 = list(self.state.keys())[-1]
        next_state = move_to_new_state_3d(
            previous,
            previous2,
            action
        )
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

        self.actions = []
        self.state = OrderedDict(
            {
                (0, 0, 0): self.seq[0],
                (1, 0, 0): self.seq[1],
            }
        )
        self.done = len(self.seq) == 2
        obs = self.observe()
        return obs

    def render(self, mode='human', display_mode="draw",
               pause_t=0.0, save_fig=False, save_path="",
               score=2022, optima_idx=0):

        if mode == "human":
            plot_3D_foleded_protein(
                list(self.state.items()),
            )

    def _get_adjacent_coords(self, coords):
        x, y, z = coords
        adjacent_coords = {
            0: (x - 1, y, z),
            1: (x, y - 1, z),
            2: (x, y + 1, z),
            3: (x + 1, y, z),
            4: (x, y, z - 1),
            5: (x, y, z + 1)
        }
        return adjacent_coords

    def _compute_free_energy(self, chain):
        path = list(chain.items())
        total_energy = 0
        for index in range(0, len(path)):
            for jndex in range(index, len(path)):
                if abs(index - jndex) >= 2:
                    current_amino_acid_i = path[index][1]
                    current_amino_acid_j = path[jndex][1]
                    current_place_i = path[index][0]
                    current_place_j = path[jndex][0]
                    x_i = current_place_i[0]
                    y_i = current_place_i[1]
                    z_i = current_place_i[2]
                    x_j = current_place_j[0]
                    y_j = current_place_j[1]
                    z_j = current_place_j[2]
                    if current_amino_acid_i == 'H' and current_amino_acid_j == 'H' and (
                            abs(x_i - x_j) + abs(y_i - y_j) + abs(z_i - z_j) == 1):
                        total_energy += 1
        return total_energy
