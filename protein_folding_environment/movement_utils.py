import numpy as np

ACTION_TO_MEANING_2D = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
ACTION_TO_MEANING_3D = {1: (-1, 0, 0), 2: (1, 0, 0), 3: (0, -1, 0), 4: (0, 1, 0), 5: (0, 0, -1), 6: (0, 0, 1)}
ACTION_TO_DIRECTION = {
    (-1, 0): {'L': (0, -1), 'R': (0, 1), 'F': (-1, 0)},
    (1, 0): {'L': (0, 1), 'R': (0, -1), 'F': (1, 0)},
    (0, -1): {'L': (1, 0), 'R': (-1, 0), 'F': (0, -1)},
    (0, 1): {'L': (0, -1), 'R': (1, 0), 'F': (0, 1)},
}
ACTION_TO_LRF = {
    1: 'L',
    2: 'R',
    3: 'F'
}
LRF_TO_ACTION = {
    'L': 1,
    'R': 2,
    'F': 3
}
ACTION_TO_MEANING_TRIANGULAR = {
    1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1), 5: (1, 1), 6: (-1, -1)
}


def move_to_new_state_lrf(previous_move_direction, state, action):
    if action not in {1, 2, 3}:
        return

    action_lrf = ACTION_TO_LRF[action]
    current_move_direction = ACTION_TO_DIRECTION[previous_move_direction][action_lrf]
    x1, y1 = state

    new_state = (x1 + current_move_direction[0], y1 + current_move_direction[1])

    return current_move_direction, new_state


def move_to_new_state_3d(p1, p2, move_direction):
    if move_direction not in {1, 2, 3, 4, 5, 6}:
        return
    x1, y1, z1 = p1

    new_state = (x1 + (ACTION_TO_MEANING_3D[move_direction])[0], y1 + (ACTION_TO_MEANING_3D[move_direction])[1],
                 z1 + (ACTION_TO_MEANING_3D[move_direction])[2])
    if p2 == new_state:
        return
    return new_state


def move_to_new_state_2d(p1, p2, move_direction):
    if move_direction not in {1, 2, 3, 4}:
        return

    x1, y1 = p1

    new_state = (x1 + (ACTION_TO_MEANING_2D[move_direction])[0], y1 + (ACTION_TO_MEANING_2D[move_direction])[1])
    if p2 == new_state:
        return
    return new_state


def move_to_new_state_triangular(p1, p2, move_direction):
    if move_direction not in {1, 2, 3, 4, 5, 6}:
        return

    x1, y1 = p1
    new_state = (
        x1 + (ACTION_TO_MEANING_TRIANGULAR[move_direction])[0], y1 + (ACTION_TO_MEANING_TRIANGULAR[move_direction])[1])
    if p2 == new_state:
        return
    return new_state
