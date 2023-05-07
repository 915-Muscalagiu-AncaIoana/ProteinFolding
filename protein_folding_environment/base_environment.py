from collections import OrderedDict

import gym
from gym import (spaces, utils, logger)
import numpy as np
from protein_folding_environment.movement_utils import move_to_new_state_2d

class ProteinFoldingBaseEnv(gym.Env):
    def __init__(self,
                 seq,
                 ):
        self.seq = seq.upper()
        self.is_trapped = False
        self.done = False
        if len(self.seq) <= 2:
            return

    def observe(self):
        action_chain = self.actions
        native_obs = np.zeros(shape=(len(self.seq) - 2,), dtype=int)
        for i, item in enumerate(action_chain):
            native_obs[i] = item
        return native_obs

    def _compute_reward(self):
        curr_reward = self._compute_free_energy(self.state)
        if self.is_trapped:
            return -0.01
        elif self.done:
            return curr_reward
        else:
            return 0

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
                    x_j = current_place_j[0]
                    y_j = current_place_j[1]
                    if current_amino_acid_i == 'H' and current_amino_acid_j == 'H' and (
                            abs(x_i - x_j) + abs(y_i - y_j) == 1):
                        total_energy += 1
        return total_energy

    def get_observation_info(self, next_state):
        is_trapped = False
        if len(self.state) < len(self.seq):
            if set(self._get_adjacent_coords(next_state).values()).issubset(self.state.keys()):
                is_trapped = True
        obs = self.observe()
        self.is_trapped = is_trapped
        self.done = len(self.state) == len(self.seq) or is_trapped
        reward = self._compute_reward()
        info = {
            'chain_length': len(self.state),
            'seq_length': len(self.seq),
            'actions': [str(i) for i in self.actions],
            'is_trapped': is_trapped,
            'state_chain': self.state,
        }
        return obs, reward, self.done, False, info

    def seed(self, seed=None):
        """
        seed the gym env
        NOTE: this umbrella seed() will seed the end globally
        it will seed:
            action_space
            np random (for uniform to_exploit)
        """
        self.np_random, seed = utils.seeding.np_random(seed)

        # NOTE: spaces sample use separate random number generator
        # that lives in gym.spaces.prng. If you want action / observation
        # space to sample deterministically you will need to seed separately
        self.action_space.seed(seed)
        # NOTE: agent also uses randomness, need to seed that separately
        np.random.seed(seed)

        return [seed]