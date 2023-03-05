import argparse
# import csv
import gym
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from prettytable import PrettyTable
import os  # for creating directories
import sys

from protein_folding_environment.movement_utils import sample_action_from_q_table
from q_learning_utils import batch_size, gamma, train_times, ReplayBuffer, device, train

# import the hpsandbox util functions
sys.path.append('../code')
from utils.plotting_utils import (
    plot_print_rewards_stats,
)
from annealling import (
    ExponentialDecay,
    LinearDecay,
)

parser = argparse.ArgumentParser(
    usage="%(prog)s [seq] [seed] [num_episodes]...",
    description="Train with "
)
parser.add_argument(
    "seq",
)
parser.add_argument(
    "seed",
    type=int,
)
parser.add_argument(
    "num_episodes",
    type=int,
)

args = parser.parse_args()

seq = args.seq.upper()  # Our input sequence
seed = args.seed  # read the seed from CMD
num_episodes = args.num_episodes  # number of episodes

max_steps_per_episode = len(seq)

learning_rate = 1
discount_rate = 0.9
# Exploration parameters
max_exploration_rate = 1
min_exploration_rate = 0.01

# render settings
show_every = num_episodes // 1000  # for plot_print_rewards_stats

rewards_all_episodes = np.zeros(
    (num_episodes,),
    # dtype=np.int32
)
reward_max = 0
num_trapped = 0

decay_mode = "exponential"  # exponential, cosine, linear
exploration_decay_rate = 5  # for exponential decay
start_decay = 0  # for exponential and linear

env = gym.make(
    id="protein_folding_environment:ProteinFolding2DEnv",
    seq=seq,
)

initial_state = env.reset()

print("initial state/obs:")
print(initial_state)

# Get number of actions from gym action space
n_actions = env.action_space.n
print("n_actions = ", n_actions)

# Initialize the Q-table to 0
Q_table = {}

for n_episode in range(num_episodes):
    # print("\nEpisode: ", n_episode)
    if (n_episode == 999):
        print(n_episode)
    x = PrettyTable()
    x.field_names = ["State", "Action", "Q_value"]
    # only render the game every once a while

    # = epsilon = max(min_exploration_rate, max_exploration_rate - exploration_decay_rate*(n_episode/200)) # linear annealing
    if decay_mode == "exponential":
        epsilon = ExponentialDecay(
            n_episode,
            num_episodes,
            min_exploration_rate,
            max_exploration_rate,
            exploration_decay_rate=exploration_decay_rate,
            start_decay=start_decay,
        )
    elif decay_mode == "linear":
        epsilon = LinearDecay(
            n_episode,
            num_episodes,
            min_exploration_rate,
            max_exploration_rate,
            start_decay=start_decay,
        )

    # reset the environment
    # Initialize the environment and state
    s = env.reset()

    done = False
    score = 0.0

    for step in range(max_steps_per_episode):

        a = sample_action_from_q_table(env, Q_table, s, epsilon)

        s_prime, r, done, truncated, info = env.step(a)

        while s_prime is None:
            # retry until action is not colliding
            # print("retry sample another action...")
            a = sample_action_from_q_table(env, Q_table, s, epsilon)
            # print("retried action = ", a)
            # Take the action (a) and observe the outcome state(s') and reward (r)
            s_prime, r, done, truncated, info = env.step(a)
        q_value = 0
        s = tuple(s)
        s_prime = tuple(s_prime)
        max_value = 0
        action = a
        if s_prime not in Q_table:
            max_value = 0
        else:
            action_value_table = Q_table[s_prime]
            for a in action_value_table:
                if action_value_table[a] > max_value:
                    max_value = action_value_table[a]

        prev_value = 0
        if s not in Q_table.keys():
            Q_table[s] = {}
            Q_table[s][action] = 0
        else:
            if action in Q_table[s]:
                prev_value = Q_table[s][action]
            else:
                Q_table[s][action] = 0

        Q_table[s][action] = Q_table[s][action] + learning_rate * (r + discount_rate * (max_value - prev_value))

        s = s_prime
        score = r

        if done:
            if len(info['actions']) == (len(seq) - 2):
                pass
            else:
                num_trapped += 1
            break

    rewards_all_episodes[n_episode] = score
    # update max reward found so far
    for state in Q_table.keys():
        for action in Q_table[state].keys():
            x.add_row([state, action, Q_table[state][action]])

    print("Episode {}, score: {:.1f}, epsilon: {:.2f}, reward_max: {}".format(
        n_episode,
        score,
        epsilon,
        reward_max,
    ))
    print(f"\ts_prime: {s_prime}, reward: {r}, done: {done}, info: {info}")
    if score > reward_max:
        print("found new highest reward = ", score)
        reward_max = score
        print(x)
        env.render()

print('Complete')

# ***** plot the stats and save in save_path *****

plot_print_rewards_stats(
    rewards_all_episodes,
    1,
    args,
    mode="show",
)

env.close()

print("\nnum_trapped = ", num_trapped)

# last line of the output is the max reward
print("\nreward_max = ", reward_max)

f = open("Qtable.txt", "w")
f.write(str(x))
f.close()
