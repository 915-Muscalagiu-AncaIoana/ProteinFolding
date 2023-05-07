import argparse
# import csv
import gym
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import os  # for creating directories
import sys
import datetime

# time the program
from time import time

from RNN.RNN import BRNN
from q_learning_utils import batch_size, gamma, train_times, ReplayBuffer, device, train

# import the hpsandbox util functions
sys.path.append('../code')
from utilities.plotting_utils import (
    plot_print_rewards_stats,
)
from annealling import (
    ExponentialDecay,
    LinearDecay,
)


def sample_action_from_q_table(Q_table, current_state, epsilon):
    # print("Sample Action called+++")
    """
    greedy epsilon choose
    """
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_table[current_state, :])
    return action



parser = argparse.ArgumentParser(
    usage="%(prog)s [seq] [seed] [algo] [num_episodes]...",
    description="DQN learning for Lattice 2D HP"
)
parser.add_argument(
    "seq",
)
parser.add_argument(
    "seed",
    type=int,
)
parser.add_argument(
    "algo",
)
parser.add_argument(
    "num_episodes",
    type=int,
)

args = parser.parse_args()

seq = args.seq.upper()  # Our input sequence
seed = args.seed  # read the seed from CMD
algo = args.algo  # path to save the experiments
num_episodes = args.num_episodes  # number of episodes


max_steps_per_episode = len(seq)

learning_rate = 0.0005

mem_start_train = max_steps_per_episode * 50
TARGET_UPDATE = 100  # fix to 100

# capped at 50,000 for <=48mer
buffer_limit = int(min(50000, num_episodes // 10))  # replay-buffer size

# Exploration parameters
max_exploration_rate = 1
min_exploration_rate = 0.5

# render settings
show_every = num_episodes // 1000  # for plot_print_rewards_stats
pause_t = 0.0
# metric for evaluation
rewards_all_episodes = np.zeros(
    (num_episodes,),
    # dtype=np.int32
)
reward_max = 0
# keep track of trapped SAW
num_trapped = 0

decay_mode = "exponential"  # exponential, cosine, linear
num_restarts = 1  # for cosine decay warmRestart=True
exploration_decay_rate = 5  # for exponential decay
start_decay = 0  # for exponential and linear

print(f"num_restarts={num_restarts} exploration_decay_rate={exploration_decay_rate} start_decay={start_decay}")
# visualize the annealing schedule
# for early stop schemes


# Nov30 2021 add one more column of step_E
hp_depth = 2  # {H,P} binary alphabet
action_depth = 7  # 0,1,2,3 in observation_box
energy_depth = 0  # state_E and step_E
# one hot the HP seq
seq_bin_arr = np.asarray([1 if x == 'H' else 0 for x in seq])
seq_one_hot = F.one_hot(torch.from_numpy(seq_bin_arr), num_classes=hp_depth)
seq_one_hot = seq_one_hot.numpy()
# print(f"seq({seq})'s one_hot = ")
# print(seq_one_hot)
init_HP_len = 2  # initial two HP units placed
first_two_actions = np.zeros((init_HP_len,), dtype=int)


def one_hot_state(state_arr, seq_one_hot, action_depth):
    # state_E_col, step_E_col):
    state_arr = np.concatenate((first_two_actions, state_arr))
    # print("after catting first_two_actions, state_arr = ", state_arr, state_arr.dtype, state_arr.shape)
    state_arr = F.one_hot(torch.from_numpy(state_arr), num_classes=action_depth)
    state_arr = state_arr.numpy()  # q.sample_action expects numpy arr
    # print("one-hot first_two_actions catted state = ")
    # print(state_arr)
    state_arr = np.concatenate((
        # state_E_col,
        # step_E_col,
        state_arr,
        seq_one_hot), axis=1)
    # print("state_arr concat with seq_one_hot, state_E_col, step_E_col =")
    # print(state_arr)
    return state_arr


# NOTE: partial_reward Sep15 changed to delta of curr-prev rewards

# env = gym.make(id="gym_lattice:Lattice2D-miranda2020Jul-v1", seq=seq)
env = gym.make(
    id="protein_folding_environment:ProteinFoldingSquareEnv",
    seq=seq,
)

# reproducible environment and action spaces, do not change lines 6-11 here (tools > settings > editor > show line numbers)



initial_state = env.reset()

print("initial state/obs:")
print(initial_state)

# Get number of actions from gym action space
n_actions = env.action_space.n
print("n_actions = ", n_actions)

row_width = action_depth + hp_depth + energy_depth
col_length = env.observation_space.shape[0] + init_HP_len

# config for RNN
input_size = row_width
# number of nodes in the hidden layers
hidden_size = 512
num_layers = 3

print("RNN_LSTM_onlyLastHidden with:")
print(f"inputs_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_classes={n_actions}")
# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)


q = BRNN(input_size, hidden_size, num_layers, n_actions).to(device)
print(q)
q_target = BRNN(input_size, hidden_size, num_layers, n_actions).to(device)
q_target.load_state_dict(q.state_dict())
optimizer = optim.Adam(q.parameters(), lr=learning_rate)
# Inspect NN state_dict in pytorch
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in q.state_dict():
    print(param_tensor, "\t", q.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


# display the model params


memory = ReplayBuffer(buffer_limit)

# monitor GPU usage
print("torch.cuda.is_available() = ", torch.cuda.is_available())
print("device = ", device)
# Additional Info when using cuda
# https://newbedev.com/how-to-check-if-pytorch-is-using-the-gpu
if device.type == 'cuda':
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

for n_episode in range(num_episodes):
    # print("\nEpisode: ", n_episode)

    # only render the game every once a while
    if (n_episode == 0) or ((n_episode + 1) % show_every == 0):
        render = True
    else:
        render = False

    # epsilon = max(min_exploration_rate, max_exploration_rate - exploration_decay_rate*(n_episode/200)) # linear annealing
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

    """
    for NN:
    "one-hot" --> return the one-hot version of the quaternary tuple
    """
    s_table = s
    s = one_hot_state(s, seq_one_hot, action_depth)
    # state_E_col, step_E_col)

    done = False
    score = 0.0



    for step in range(max_steps_per_episode):



        a = q.sample_action(torch.from_numpy(s).float().unsqueeze(0), epsilon)

        # print('---> action = ', a)

        # take the step and get the returned observation s_prime
        s_prime, r, done, truncated, info = env.step(a)


        # todo give bad reqard if it is trapped or overlaps
        while s_prime is None:
            # retry until action is not colliding
            # print("retry sample another action...")
            a = ((a + 1) % 6)
            # print("retried action = ", a)
            # Take the action (a) and observe the outcome state(s') and reward (r)
            s_prime, r, done, truncated, info = env.step(a)
            # print(f"s_prime: {s_prime}, reward: {r}, done: {done}, info: {info}")

        # Only keep first turn of Left
        # internal 3actionStateEnv self.last_action updated
        a = env.last_action

        (state_E, step_E, reward) = r

        """
        for NN:
            "one-hot" --> return the one-hot version of the quaternary tuple
        """
        s_prime_table = s_prime
        s_prime = one_hot_state(s_prime, seq_one_hot, action_depth)
        # state_E_col, step_E_col)
        # print("one-hot s_prime = ")
        # print(s_prime)

        if info["is_trapped"]:

            reward = state_E
            # print("adjusted trapped reward = ", reward)

        # NOTE: MUST ENSURE THE REWARD IS FINALIZED BEFORE FEEDING TO RL ALGO!!

        r = reward

        # NOTE: done_mask is for when you get the end of a run,
        # then is no future reward, so we mask it with done_mask
        done_mask = 0.0 if done else 1.0

        memory.put((s, a, r, s_prime, done_mask))
        s = s_prime

        score = r


        if done:

            if len(info['actions']) == (len(seq) - 2):
                # print("Complete: used up all actions!")
                pass
            else:
                num_trapped += 1
            break


    if memory.size() > mem_start_train:
        train(q, q_target, memory, optimizer)

    # Update the target network, copying all weights and biases in DQN
    if n_episode % TARGET_UPDATE == 0:
        q_target.load_state_dict(q.state_dict())

    # Add current episode reward to total rewards list
    rewards_all_episodes[n_episode] = score
    # update max reward found so far
    if score > reward_max:
        print("found new highest reward = ", score)
        reward_max = score
        env.render()

    if (n_episode == 0) or ((n_episode + 1) % show_every == 0):
        print("Episode {}, score: {:.1f}, epsilon: {:.2f}, reward_max: {}".format(
            n_episode,
            score,
            epsilon,
            reward_max,
        ))
        print(f"\ts_prime: {s_prime[:3], s_prime.shape}, reward: {r}, done: {done}, info: {info}")
    # move on to the next episode

print('Complete')


# ***** plot the stats and save in save_path *****

plot_print_rewards_stats(
    rewards_all_episodes,
    show_every,
    args,
    mode="show",
)

env.close()

print("\nnum_trapped = ", num_trapped)

# last line of the output is the max reward
print("\nreward_max = ", reward_max)
