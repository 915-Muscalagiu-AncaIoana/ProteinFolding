# Select the number of qubits
import argparse
from collections import deque

import qiskit as qk
import gym
import numpy as np
import torch
from qiskit.utils import QuantumInstance
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import Adam

from QNN.PQC import parametrized_circuit
from QNN.encoder import encoding_layer, exp_val_layer
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector

import matplotlib.pyplot as plt
import seaborn as sns

from utilities.plotting_utils import plot_moving_avg

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
max_steps_per_episode = len(seq) - 2
num_qubits = len(seq) - 2
rewards_all_episodes = np.zeros(
    (num_episodes,),
    # dtype=np.int32
)
# Generate the Parametrized Quantum Circuit (note the flags reuploading and reps)
qc = parametrized_circuit(num_qubits=num_qubits,
                          reuploading=True,
                          reps=6)

# Fetch the parameters from the circuit and divide them in Inputs (X) and Trainable Parameters (params)
# The first four parameters are for the inputs
X = list(qc.parameters)[: num_qubits]

# The remaining ones are the trainable weights of the quantum neural network
params = list(qc.parameters)[num_qubits:]
qc.draw(output='mpl', filename='my_circuit.png')

# Select a quantum backend to run the simulation of the quantum circuit
qi = QuantumInstance(qk.Aer.get_backend('statevector_simulator'))

# Create a Quantum Neural Network object starting from the quantum circuit defined above
qnn = CircuitQNN(qc, input_params=X, weight_params=params,
                 quantum_instance=qi)

initial_weights = (2 * np.random.rand(qnn.num_weights) - 1)
quantum_nn = TorchConnector(qnn, initial_weights)

encoding = encoding_layer(num_qubits)

# Classical trainable postprocessing
exp_val = exp_val_layer(action_space=4, state_length=num_qubits)

# Stack the classical and quantum layers together
model = torch.nn.Sequential(encoding,
                            quantum_nn,
                            exp_val)

model.state_dict()

env = gym.make(
    id="protein_folding_environment:ProteinFoldingLRF2DEnv",
    seq=seq,
)
input_shape = env.observation_space.shape  # == env.observation_space.shape
n_outputs = env.action_space.n  # == env.action_space.n

replay_memory = deque(maxlen=2000)
batch_size = 4
discount_rate = 0.99
optimizer = Adam(model.parameters(), lr=1e-2)


def epsilon_greedy_policy(state, epsilon=0):
    """Manages the transition from the *exploration* to *exploitation* phase"""
    if np.random.rand() < epsilon:
        return np.random.randint(1, n_outputs)
    else:
        with torch.no_grad():
            Q_values = model(Tensor(state))
            Q_values = Q_values.numpy()
        return np.argmax(Q_values[0])+1


def sample_experiences(batch_size):
    """Sample some past experiences from the replay memory"""
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states_arrays, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    #next_states = [next_state[0] if next_state != None else next_state for next_state in next_states_arrays]
    return states, actions, rewards, next_states_arrays, dones


def play_one_step(env, state, epsilon):
    """Perform one action in the environment and register the state of the system"""
    action = epsilon_greedy_policy(state, epsilon)
    next_state = None
    while next_state is None:
        next_state, reward, done, truncated, info = env.step(action)
        action = epsilon_greedy_policy(state, epsilon)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


def sequential_training_step(batch_size):
    """
    Actual training routine. Implements the Deep Q-Learning algorithm.

    This implementation evaluates individual losses sequentially instead of using batches.
    This is due to an issue in the TorchConnector, which yields vanishing gradients if it
    is called with a batch of data (see https://github.com/Qiskit/qiskit-machine-learning/issues/100).

    Use this training for the quantum model. If using the classical model, you can use indifferently
    this implementation or the batched one below.
    """

    # Sample past experiences
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences

    # Evaluates Target Q-values
    with torch.no_grad():
        next_Q_values = model(Tensor(next_states)).numpy()
    max_next_Q_values = np.max(next_Q_values, axis=0)
    target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)

    # Accumulate Loss sequentially (if batching data, gradients of the parameters are vanishing)
    loss = 0.
    for j, state in enumerate(states):
        single_Q_value = model(Tensor(state))
        Q_value = single_Q_value[actions[j]]
        loss += (target_Q_values[j] - Q_value) ** 2

    # Evaluate the gradients and update the parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


best_score = 0
rewards = []
# We let the agent train for 2000 episodes
for episode in range(num_episodes):

    # Run enviroment simulation
    obs = env.reset()

    # 200 is the target score for considering the environment solved
    for step in range(max_steps_per_episode):

        # Manages the transition from exploration to exploitation
        epsilon = max(1 - episode / num_episodes, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    rewards.append((episode,reward))
    rewards_all_episodes[episode] = reward
    # Saving best agent
    if reward > best_score:
        # torch.save(model.state_dict(), './new_model_best_weights.pth') # Save best weights
        best_score = reward
        best_episode = episode
        env.render()
    print("\rEpisode: {}, Steps : {}, eps: {:.3f}, Score: {}".format(episode, step + 1, epsilon,reward), end="")

    # Start training only after some exploration experiences
    if episode > 20:
        sequential_training_step(batch_size)

model.state_dict()


sns.set_theme()

plot_moving_avg(rewards_all_episodes, mode="show", save_path="v")
cmap = plt.get_cmap('tab20c')

fig = plt.figure(figsize=(8, 5))
plt.axhline([0], ls='dashed', c=cmap(9))
plt.xlim(0,num_episodes)
plt.ylim(-1, best_score+1)
plt.text(x = best_episode , y = best_score, s='Max reward', c=cmap(8))
plt.text(x = num_episodes/4 , y = best_score/2, s='Exploration Phase', c=cmap(8))
plt.text(x = num_episodes*3/4  , y = best_score/2, s='Exploitation Phase', c=cmap(8))
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Final reward")

plt.show()
