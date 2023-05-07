from collections import OrderedDict
from gym import (spaces)

from protein_folding_environment.base_environment import ProteinFoldingBaseEnv
from utilities.plotting_utils import plot_2D_foleded_protein

class ProteinFolding2DEnv(ProteinFoldingBaseEnv):
    def __init__(self, seq):
        super().__init__(seq)
        self.reset()

    def reset(self):
        self.actions = []
        self.state = OrderedDict(
            {
                (0, 0): self.seq[0],
                (1, 0): self.seq[1],
            }
        )
        self.done = len(self.seq) == 2
        obs = self.observe()
        return obs

    def render(self, mode='human'):
        if mode == "human":
            plot_2D_foleded_protein(
                list(self.state.items()),
            )