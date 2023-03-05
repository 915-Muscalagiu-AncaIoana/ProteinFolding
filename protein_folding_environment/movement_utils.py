import numpy as np

ACTION_TO_MEANING_2D = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

ACTION_TO_MEANING_3D = {1: (-1, 0, 0), 2: (1, 0, 0), 3: (0, -1, 0), 4: (0, 1, 0), 5: (0, 0, -1), 6: (0, 0, 1)}


def move_to_new_state_3d(p1, p2, move_direction):
    if move_direction not in {1, 2, 3, 4, 5, 6}:
        return

    # print("p1, p2: ", p1, p2)
    x1, y1, z1 = p1

    new_state = (x1 + (ACTION_TO_MEANING_3D[move_direction])[0], y1 + (ACTION_TO_MEANING_3D[move_direction])[1],
                 z1 + (ACTION_TO_MEANING_3D[move_direction])[2])
    if p2 == new_state:
        return
    return new_state


def move_to_new_state_2d(p1, p2, move_direction):
    if move_direction not in {1, 2, 3, 4}:
        return

    # print("p1, p2: ", p1, p2)
    x1, y1 = p1

    new_state = (x1 + (ACTION_TO_MEANING_2D[move_direction])[0], y1 + (ACTION_TO_MEANING_2D[move_direction])[1])
    if p2 == new_state:
        return
    return new_state

def sample_action_from_q_table(env, Q_table, current_state, epsilon):
    # print("Sample Action called+++")
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

def sample_action_from_ann(env,model, current_state, epsilon):
    # print("Sample Action called+++")
    """
    greedy epsilon choose
    """
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(
            model.predict(np.identity(env.observation_space.n)[current_state:current_state + 1]))

    return action