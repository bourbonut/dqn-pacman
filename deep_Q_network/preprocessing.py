from .parameters import PACMAN_COLOR_GRAY, WALL_COLOR_GRAY, BACKGROUND_GRAY
import numpy as np
import cv2
import torch

from collections import Counter
from time import perf_counter as pf


def extend_walls(img):
    extension = np.array([[WALL_COLOR_GRAY for x in range(160)] for y in range(3)])
    return np.concatenate([extension, img, extension])


def unit_prepr_obs(obs):
    gray_img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    trimmed_img = gray_img[1:171]
    extended_img = extend_walls(trimmed_img)
    canvas = extended_img[1::4, 1::4]
    pills_walls = canvas.copy()
    pills_walls[canvas != WALL_COLOR_GRAY] = 0
    pacman_monsters = canvas.copy()
    pacman_monsters[(canvas == WALL_COLOR_GRAY) | (canvas == BACKGROUND_GRAY)] = 0
    pacman = pacman_monsters.copy()
    monsters = pacman_monsters.copy()
    pacman[pacman_monsters != PACMAN_COLOR_GRAY] = 0
    monsters[pacman_monsters == PACMAN_COLOR_GRAY] = 0
    return np.stack(
        [
            pills_walls.astype(np.float32),
            pacman.astype(np.float32),
            monsters.astype(np.float32),
        ]
    )


def preprocess_observation(observations, new_obs):
    for i in range(3):
        observations[3 - i] = observations[2 - i]
    observations[0] = unit_prepr_obs(new_obs)
    state = np.concatenate(observations)
    screen = torch.from_numpy(state)
    return screen.unsqueeze(0)


def init_obs(env):
    return [unit_prepr_obs(env.step(0)[0]) for i_step in range(4)][::-1]


# def unit_prepr_obs(obs):
#     cropped_img = obs[1:176:2, ::2]
#     gray_truncated_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
#     gray_truncated_img[gray_truncated_img == PACMAN_COLOR_GRAY] = 0
#     return gray_truncated_img.astype(np.float32)
#
#
# def preprocess_observation(observations, new_obs):
#     for i in range(3):
#         observations[3 - i, :, :] = observations[2 - i, :, :]
#     observations[0, :, :] = unit_prepr_obs(new_obs)
#     screen = torch.from_numpy(observations)
#     return screen.unsqueeze(0)
#
#
# def init_obs(env):
#     return np.stack([unit_prepr_obs(env.step(0)[0]) for i_step in range(4)][::-1])
