from .parameters import *
from .preprocessing import preprocess_observation, init_obs

from .model import optimize_model, DQN
from .decision import DecisionMaker
from .memory import ReplayMemory

from ale_py import ALEInterface
from ale_py.roms import Pacman
import gym, torch  # cv2
from torch import optim

# Set environment
ale = ALEInterface()
ale.loadROM(Pacman)

env = gym.make("MsPacman-v0")

# Set neural networks
policy_DQN = DQN(N_ACTIONS).to(device)
target_DQN = DQN(N_ACTIONS).to(device)
target_DQN.load_state_dict(policy_DQN.state_dict())

# Set optimizer
# optimizer = optim.Adam(policy_DQN.parameters(), lr=LEARNING_RATE)
optimizer = optim.SGD(
    policy_DQN.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True
)

# Set memory
memory = ReplayMemory(REPLAY_MEMORY_SIZE, BATCH_SIZE)

# Set decision maker
dmaker = DecisionMaker(0, policy_DQN)
