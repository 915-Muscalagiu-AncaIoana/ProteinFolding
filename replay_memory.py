
# imports
from collections import deque
# Note: deque is pronounced as “deck.” The name stands for double-ended queue.
import random
import pickle
# pytorch deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    """
    for DQN (off-policy RL), big buffer of experience
    you don't update weights of the NN as you run
    through the environment, instead you save
    your experience of the environment to this ReplayBuffer
    It has a max-size to fit in certain examples
    """

    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            # a tuple that tells us what the state was
            # at a particular point in time
            # we store the current state, the action we chose,
            # the state we ended up in, and whether finished or not
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # converting the list to a single numpy.ndarray with numpy.array()
        # before converting to a tensor
        s_lst = np.array(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_prime_lst = np.array(s_prime_lst)
        done_mask_lst = np.array(done_mask_lst)


        return torch.tensor(s_lst, device=device, dtype=torch.float), torch.tensor(a_lst, device=device), \
               torch.tensor(r_lst, device=device), torch.tensor(s_prime_lst, device=device, dtype=torch.float), \
               torch.tensor(done_mask_lst, device=device)

    def size(self):
        return len(self.buffer)

    def save(self, save_path):
        """save in .pkl file"""
        with open(save_path, 'wb') as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        """load a .pkl file"""
        with open(file_path, 'rb') as handle:
            self.buffer = pickle.load(handle)
