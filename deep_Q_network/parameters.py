import numpy as np
import torch, cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing constant
WALL_COLOR = [228, 111, 111]
BACKGROUND = [0, 28, 136]
PACMAN_COLOR = [210, 164, 74]
tonumpy = lambda color: np.array([[color]], dtype=np.uint8)
togray = lambda color: cv2.cvtColor(tonumpy(color), cv2.COLOR_RGB2GRAY)[0][0]
WALL_COLOR_GRAY = togray(WALL_COLOR)
BACKGROUND_GRAY = togray(BACKGROUND)
PACMAN_COLOR_GRAY = togray(PACMAN_COLOR)

# Reinforcement learning constants
BATCH_SIZE = 256
DISCOUNT_RATE = 0.99
EPS_MAX = 1.0
EPS_MIN = 0.1
EPS_DECAY = 1_000_000
TARGET_UPDATE = 9_000
REPLAY_MEMORY_SIZE = 7_500

# Environment constants
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
