from math import log

REWARDS = {
    "default": -0.2,
    200: 20,
    50: 15,
    10: 10,
    0: 0,
    "lose": -log(20, 1000),
    "win": 10,
    "reverse": -2,
}

def transform_reward(reward: int):
    return log(reward, 1000) if reward > 0 else reward
