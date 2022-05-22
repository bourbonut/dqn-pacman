import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing constant
WALL_COLOR = [111, 111, 228]
BACKGROUND = [136, 28, 0]
PACMAN_COLOR = [74, 164, 210]


# Reinforcement learning constants
BATCH_SIZE = 32
DISCOUNT_RATE = 0.95
EPS_MAX = 1.0
EPS_MIN = 0.1
EPS_DECAY = 1_000_000
TARGET_UPDATE = 9_000
REPLAY_MEMORY_SIZE = 20_000

# Environment constants
HEIGHT = 88
WIDTH = 80
N_ACTIONS = 9  # env.action_space.n
AVOIDED_STEPS = 88  # At the beginning, there is a period of time where the game doesn't allowed the player to move Pacman
DEAD_STEPS = 36  # frames to avoid when the agent dies
K_FRAME = 4
TRAINING_START = 0
TAU = 2.5e-4

# Optimizer parameters
LEARNING_RATE = 2.5e-4
# DECAY_RATE = 0.99
MOMENTUM = 0.95

# Algorithm constant
MAX_FRAMES = 2_000_000
SAVE_MODEL = 20
