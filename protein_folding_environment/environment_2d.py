# Import gym modules
import sys
from collections import OrderedDict

import gym
from gym import (spaces, utils, logger)
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from protein_folding_environment.movement_utils import move_to_new_state_2d
from utils.plotting_utils import plot_HPSandbox_conf_2D

sys.path.append('../code')


# Over-write the miranda's default action to string dict


class ProteinFolding2DEnv(gym.Env):

    def __init__(self,
                 seq,
                 ):

        self.seq = seq.upper()

        self.reset()

        if len(self.seq) <= 2:
            return

        self.action_space = spaces.Discrete(start=1,n=4)

        self.observation_space = spaces.Box(low=0, high=3,
                                            # quaternary tuple len is N-2
                                            shape=(len(self.seq) - 2,),
                                            dtype=int)
        self.is_trapped = False

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))

        self.last_action = action
        is_trapped = False  # Trap signal

        previous = list(self.state.keys())[-1]
        previous2 = list(self.state.keys())[-2]
        # p3 is to-be-moved point == next_move
        next_state = move_to_new_state_2d(
            previous,
            previous2,
            action
        )
        if next_state is None:
            return (None, None, False, False, {})

        idx = len(self.state)
        if next_state in self.state or next_state is None:
            return (None, None, False, False, {})
        # only append valid actions to action chain
        self.actions.append(action)
        try:
            self.state.update({next_state: self.seq[idx]})
        except IndexError:
            logger.error('All molecules have been placed! Nothing can be added to the protein chain.')
            raise

        # NOTE: agent is only trapped WHEN THERE ARE STILL STEPS TO BE DONE!
        if len(self.state) < len(self.seq):
            if set(self._get_adjacent_coords(next_state).values()).issubset(self.state.keys()):
                # logger.warn('Your agent was trapped! Ending the episode.')
                is_trapped = True

        # Set-up return values
        obs = self.observe()
        # print("\n***********step's obs*********")
        # print(obs)
        self.is_trapped = is_trapped
        self.done = True if (len(self.state) == len(self.seq) or is_trapped) else False
        reward = self._compute_reward()
        info = {
            'chain_length': len(self.state),
            'seq_length': len(self.seq),
            'actions': [str(i) for i in self.actions],
            'is_trapped': is_trapped,
            'state_chain': self.state,

        }

        return (obs, reward, self.done, False, info)

    def observe(self):

        # self.actions is list of 012 integers
        action_chain = self.actions

        # native obs space is [0,0,0,...,0]
        # len of quaternary tuple is N-2
        native_obs = np.zeros(shape=(len(self.seq) - 2,), dtype=int)

        # transfer the actions in action chain to
        # the obs array (each action+1)
        for i, item in enumerate(action_chain):
            native_obs[i] = item

        # for NN input, preserve the np array instead of tuple
        quaternary_tuple = native_obs

        return quaternary_tuple

    def reset(self):

        self.actions = []

        self.last_action = None
        self.prev_reward = 0

        # customized reset
        # in miranda Jul2020 baseEnv and in 4actionState,
        # the initial polymer is placed at origin
        # for 3actionState, place the next polyer at (0,1)
        self.state = OrderedDict(
            {
                (0, 0): self.seq[0],
                (1, 0): self.seq[1],
            }
        )
        self.done = len(self.seq) == 2
        obs = self.observe()

        return obs

    def render(self, mode='human', display_mode="draw",
               pause_t=0.0, save_fig=False, save_path="",
               score=2022, optima_idx=0):

        if mode == "human":
            # matplotlib plot the conf
            plot_HPSandbox_conf_2D(
                list(self.state.items()),
                display_mode=display_mode,
                pause_t=pause_t,
                save_fig=save_fig,
                save_path=save_path,
                score=score,
                optima_idx=optima_idx,
                info={
                    'chain_length': len(self.state),
                    'seq_length': len(self.seq),
                    'actions': [str(i) for i in self.actions],
                },
            )

    def _get_adjacent_coords(self, coords):

        x, y = coords
        adjacent_coords = {
            0: (x - 1, y),
            1: (x, y - 1),
            2: (x, y + 1),
            3: (x + 1, y),
        }

        return adjacent_coords

    def _compute_reward(self):
        # new Sep19 reward in tuple (state_E, step_E, reward)
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
