import pickle

import numpy as np
import torch


def ExponentialDecay(episode, num_episodes,
                min_exploration_rate, max_exploration_rate,
                exploration_decay_rate=5,
                start_decay=0):
    decay_duration = num_episodes - start_decay
    exploration_rate = max_exploration_rate
    if episode > start_decay:
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*(episode-start_decay)/decay_duration)
    return exploration_rate

def LinearDecay(episode, num_episodes,
                min_exploration_rate, max_exploration_rate,
                start_decay=0):
    decay_duration = num_episodes - start_decay
    exploration_rate = max_exploration_rate
    if episode > start_decay:
        exploration_rate = min_exploration_rate + \
                            (decay_duration-(episode-start_decay))/decay_duration
        # print("(decay_duration-(episode-start_decay))/decay_duration = ", (decay_duration-(episode-start_decay))/decay_duration)
    # print("exploration_rate = ", exploration_rate)
    return exploration_rate
