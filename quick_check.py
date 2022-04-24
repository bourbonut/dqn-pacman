from matplotlib import pyplot as plt
from agent import *
import torch
import numpy as np
from itertools import count
from pathlib import Path

PATH = Path().absolute()

device = "cpu"
agent = DQN(HEIGHT, WIDTH, N_ACTIONS).to(device)

ep = int(input("Episode="))
agent.load_state_dict(torch.load(str(PATH / "models" / f"policy-model-{ep}.pt")))
agent.eval()

obs = env.reset()

for i_step in range(AVOIDED_STEPS - 5):
    obs, reward, done, info = env.step(0)

observations = np.stack([unit_prepr_obs(env.step(0)[0]) for i_step in range(4)])


for t in count():
    state = preprocessing_observation(observations, obs)
    with torch.no_grad():
        action = agent(state).max(1)[1].view(1, 1)
    obs, reward, done, info = env.step(action.item())
    plt.ion()
    plt.imshow(obs)
    plt.draw()
    plt.pause(0.0001)
    if done:
        break
