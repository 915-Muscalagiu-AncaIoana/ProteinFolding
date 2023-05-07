import argparse
# import csv
import gym
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys

from ANN.ANN import DQN, FCN_QNet
from RNN.RNN import RNN_LSTM_onlyLastHidden
from utilities.training_utils import BATCH_SIZE,discount_rate, train_times,  device, train, sample_action_from_q_table, sample_action_from_ann
from replay_memory import ReplayBuffer


# import the hpsandbox util functions
sys.path.append('../code')
from utilities.plotting_utils import (
    plot_print_rewards_stats, plot_loss, plot_moving_avg,
)
from annealling import (
    ExponentialDecay,
    LinearDecay,
)


def one_hot_state(state_arr, action_depth):
    # print("after catting first_two_actions, state_arr = ", state_arr, state_arr.dtype, state_arr.shape)
    state_arr = F.one_hot(torch.from_numpy(state_arr), num_classes=action_depth)
    state_arr = state_arr.numpy()  # q.sample_action expects numpy arr
    return state_arr


seq = "HHPPHH"  # Our input sequence
seed = 22  # read the seed from CMD
num_episodes = 20000  # number of episodes

max_steps_per_episode = len(seq) - 2

learning_rate = 0.0005
discount_rate = 0.9
# Exploration parameters
max_exploration_rate = 1
min_exploration_rate = 0

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
    id="protein_folding_environment:ProteinFoldingSquareEnv",
    seq=seq,
)

torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)
initial_state = env.reset()

print("initial state/obs:")
print(initial_state)

# Get number of actions from gym action space
n_observations = len(seq) - 2

col_length = env.observation_space.shape[0]

n_actions = env.action_space.n + 1
hidden_size, num_layers, = 256, 2
print("n_actions = ", n_actions)
memory = ReplayBuffer(60000)
policy_net = FCN_QNet(col_length * n_actions, n_actions - 1).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
target_net = FCN_QNet(col_length * n_actions, n_actions - 1).to(device)
target_net.load_state_dict(policy_net.state_dict())
losses = []
eps = []
for n_episode in range(num_episodes):
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

    s = one_hot_state(s, n_actions)
    done = False
    score = 0.0

    for step in range(max_steps_per_episode):

        a = policy_net.sample_action(torch.from_numpy(s).float().unsqueeze(0), epsilon)

        s_prime, r, done, truncated, info = env.step(a)

        while s_prime is None:
            # retry until action is not colliding
            # print("retry sample another action...")
            a = policy_net.sample_action(torch.from_numpy(s).float().unsqueeze(0), epsilon)
            # print("retried action = ", a)
            # Take the action (a) and observe the outcome state(s') and reward (r)
            s_prime, r, done, truncated, info = env.step(a)
        s_prime = one_hot_state(s_prime, n_actions)
        q_value = 0
        max_value = 0
        action = a
        done_mask = 0.0 if done else 1.0
        memory.put((s, a, r, s_prime, done_mask))
        if memory.size() >= BATCH_SIZE:
            eps.append(n_episode + 0.01 * step)

        if done:
            if len(info['actions']) == (len(seq) - 2):
                pass
            else:
                num_trapped += 1
            break
        s = s_prime

    losses = train(policy_net, target_net, memory, optimizer, losses)

    # Update the target network, copying all weights and biases in DQN
    if n_episode % 100 == 0:
        target_net.load_state_dict(policy_net.state_dict())
    score = r
    rewards_all_episodes[n_episode] = score
    # update max reward found so far

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
        env.render()

print('Complete')

# ***** plot the stats and save in save_path *****

plot_print_rewards_stats(
    rewards_all_episodes,
    1,
    (seq,seed,num_episodes),
    "show",
    "fe"
)
plot_moving_avg(rewards_all_episodes, mode="show", save_path="v")
save_path = "./models/square/"+"ANN_"+seq+".pth"
torch.save(policy_net.state_dict(), f'{save_path}-state_dict.pth')
plot_loss(
    eps,
    losses
)
env.close()

print("\nnum_trapped = ", num_trapped)

# last line of the output is the max reward
print("\nreward_max = ", reward_max)

