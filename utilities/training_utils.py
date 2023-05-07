import numpy as np
import torch
from torch import nn
from torch.optim import optimizer
import torch.nn.functional as F

# hyperparameters
discount_rate = 0.99  # discount rate
BATCH_SIZE = 128
RELATIVE_MODELS_PATH = "./models/"
train_times = 10  # number of times train was run in a loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_action_from_q_table(env, Q_table, current_state, epsilon):
    """
    greedy epsilon choose
    """
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        current_state = tuple(current_state)
        if current_state not in Q_table.keys():
            return env.action_space.sample()
        action_value_table = Q_table[current_state]
        max_value = 0
        action = env.action_space.sample()
        for a in action_value_table:
            if action_value_table[a] >= max_value:
                max_value = action_value_table[a]
                action = a

    return action


def sample_action_from_ann(env, model, current_state, epsilon):
    """
    greedy epsilon choose
    """
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = model(torch.from_numpy(current_state).float().unsqueeze(0))
        action = action.argmax().item() + 1

    return action


def train(policy_net, target_net, memory, optimizer, losses):
    """
    core algorithm of Deep policy_net-learning

    do this training once per evaluation of the environment
    run evaluation once and train X times
    """
    if memory.size() < BATCH_SIZE:
        return losses

    for i in range(train_times):
        s, a, r, s_prime, done = memory.sample(BATCH_SIZE)

        state_values = policy_net(s)
        q_values_state = state_values.gather(1, a - 1)
        max_q_prime = target_net(s_prime).max(1)[0].unsqueeze(1)
        q_values_expected = r + discount_rate * max_q_prime * done
        loss = F.smooth_l1_loss(q_values_state, q_values_expected)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses


def save_model(model, environment, sequence, model_type):
    file_name = environment + "/" + model_type + "_" + sequence
    path = RELATIVE_MODELS_PATH + file_name + ".pth"
    torch.save(model.state_dict(), path)
    print("Successfully saved model: " + file_name)

