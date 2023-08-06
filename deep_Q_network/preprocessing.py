from .parameters import PACMAN_COLOR_GRAY, WALL_COLOR_GRAY, BACKGROUND_GRAY
from ale_py import ALEInterface
from typing import List
import numpy as np
import cv2
import torch

from collections import Counter
from time import perf_counter as pf

def extend_walls(img: np.array):
    extension = np.array([[WALL_COLOR_GRAY for x in range(160)] for y in range(3)])
    return np.concatenate([extension, img, extension])

def unit_prepr_obs(obs: np.array):
    gray_img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    trimmed_img = gray_img[1:171]
    extended_img = extend_walls(trimmed_img)
    final_img = extended_img[1::4, 1::4]
    return np.stack([final_img.astype(np.float32)])

def preprocess_observation(observations: List[np.array], new_obs: np.array):
    for i in range(3):
        observations[3 - i] = observations[2 - i]
    observations[0] = unit_prepr_obs(new_obs)
    state = np.concatenate(observations)
    screen = torch.from_numpy(state)
    return screen.unsqueeze(0)

def init_obs(env: ALEInterface):
    return [unit_prepr_obs(env.step(0)[0]) for i_step in range(4)][::-1]
