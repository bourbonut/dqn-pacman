from .parameters import PACMAN_COLOR, WALL_COLOR, BACKGROUND
import numpy as np
import cv2
import torch

from collections import Counter
from time import perf_counter as pf


def extend_walls(img):
    extension = np.array([[WALL_COLOR for x in range(160)] for y in range(3)])
    return np.concatenate([extension, img, extension])


def shrink_cell(cell):
    reshaped_cell = cell.reshape(-1, 3).tolist()
    counter = Counter(map(tuple, reshaped_cell))
    max_value = max(counter.values())
    colors = [key for key in counter if counter[key] == max_value]
    result = max(colors, key=lambda x: sum(x))
    return result


def change_color(array, color, value, diff=True):
    r, g, b = color
    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    if diff:
        mask = (red != r) & (green != g) & (blue != b)
    else:
        mask = (red == r) | (green == g) | (blue == b)
    array[:, :, :3][mask] = value


def preprocess_observation(obs):
    trimmed_img = obs[1:171]
    extended_img = extend_walls(trimmed_img)
    # shrink = lambda i, j: shrink_cell(extended_img[i : i + 4, j : j + 4])
    # shrunk_img = np.array([[shrink(4 * i, 4 * j) for j in range(40)] for i in range(44)])
    canvas = extended_img[1::4, 1::4]  # shrunk_img.astype(np.uint8)
    pills_walls = canvas.copy()
    change_color(pills_walls, WALL_COLOR, [0, 0, 0])
    pacman_monsters = canvas.copy()
    change_color(pacman_monsters, WALL_COLOR, [0, 0, 0], False)
    change_color(pacman_monsters, BACKGROUND, [0, 0, 0], False)
    pacman = pacman_monsters.copy()
    monsters = pacman_monsters.copy()
    change_color(pacman, PACMAN_COLOR, [0, 0, 0])
    change_color(monsters, PACMAN_COLOR, [0, 0, 0], False)
    # return [shrunk_img(x) for x in (pills_walls, pacman, monsters)]
    return [pills_walls, pacman, monsters]


def identify_movement(old, current):
    unique = lambda x: set(map(tuple, x.reshape(-1, 3).tolist())) - {(0, 0, 0)}
    get_positions = lambda color, img: np.argwhere(img == color)
    old_colors = unique(old)
    current_colors = unique(current)
    common_colors = old_colors & current_colors
    old_positions = {color: get_positions(color, old) for color in old_colors}
    current_positions = {color: get_positions(color, current) for color in current_colors}
    bc = lambda positions: np.mean(positions[::3, :2], axis=0)  # barycenter
    get_speed = lambda color: bc(current_positions[color]) - bc(old_positions[color])
    velocities = {color: get_speed(color) for color in common_colors}
    velocity_map_x = np.zeros(old.shape[:2])
    velocity_map_y = np.zeros(old.shape[:2])
    for color in common_colors:
        locations = np.where(current == color)[:2]
        velocity_map_x[locations] = velocities[color][0]
        velocity_map_y[locations] = velocities[color][1]
    return [velocity_map_x, velocity_map_y]


# def identify_movement_gray(old, current):
#     unique = lambda x: set(x.reshape(-1).tolist()) - {0}
#     get_positions = lambda color, img: np.argwhere(img == color)
#     old_colors = unique(old)
#     current_colors = unique(current)
#     common_colors = old_colors & current_colors
#     old_positions = {color: get_positions(color, old) for color in old_colors}
#     current_positions = {color: get_positions(color, current) for color in current_colors}
#     print(old_positions)
#     print(current_positions)
#     bc = lambda positions: np.mean(positions, axis=0)  # barycenter
#     get_speed = lambda color: bc(current_positions[color]) - bc(old_positions[color])
#     velocities = {color: get_speed(color) for color in common_colors}
#     velocity_map_x = np.zeros(old.shape[:2])
#     velocity_map_y = np.zeros(old.shape[:2])
#     for color in common_colors:
#         locations = np.where(current == color)[:2]
#         velocity_map_x[locations] = velocities[color][0]
#         velocity_map_y[locations] = velocities[color][1]
#     return [velocity_map_x, velocity_map_y]


def preprocess_state(old_state, current_state):
    # v_pacman = identify_movement(old_state[1], current_state[1])
    # v_monsters = identify_movement(old_state[2], current_state[2])
    # current_state[1] = np.where(current_state[1] > 0.5, 1, 0)
    gray = lambda state: cv2.cvtColor(state.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    observation = np.stack([[gray(s) for s in current_state]])  #  + v_pacman + v_monsters
    screen = torch.from_numpy(observation[0].astype(np.float32))
    return screen.unsqueeze(0)
