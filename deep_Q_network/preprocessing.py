from .parameters import PACMAN_COLOR, WALL_COLOR, BACKGROUND
import numpy as np
import cv2
import torch

from collections import Counter


def extend_walls(img):
    extension = np.array([[WALL_COLOR for x in range(160)] for y in range(3)])
    return np.concatenate([extension, img, extension])


def shrink_cell(cell):
    reshaped_cell = cell.reshape(-1, 3).tolist()
    counter = Counter(map(tuple, reshaped_cell))
    max_value = max(counter.values())
    colors = [key for key in counter if counter[key] == max_value]
    return max(colors, key=lambda x: np.sum(x))


def preprocess_observation(obs):
    trimmed_img = obs[1:171]
    extended_img = extend_walls(trimmed_img)
    shrink = lambda i, j: shrink_cell(extended_img[i : i + 4, j : j + 4])
    shrunk_img = np.array([[shrink(4 * i, 4 * j) for j in range(40)] for i in range(44)])
    canvas = shrunk_img
    pills_walls = canvas.copy()
    pills_walls[canvas != WALL_COLOR] = 0
    pacman_monsters = canvas.copy()
    pacman_monsters[(canvas == WALL_COLOR) | (canvas == BACKGROUND)] = 0
    pacman = pacman_monsters.copy()
    monsters = pacman_monsters.copy()
    pacman[pacman_monsters != PACMAN_COLOR] = 0
    monsters[pacman_monsters == PACMAN_COLOR] = 0
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


def preprocess_state(old_state, current_state):
    v_pacman = identify_movement(old_state[1], current_state[1])
    v_monsters = identify_movement(old_state[2], current_state[2])
    gray = lambda state: cv2.cvtColor(state.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return np.stack([[gray(s) for s in current_state] + v_pacman + v_monsters])
