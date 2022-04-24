from itertools import count
from agent import *
from matplotlib import pyplot as plt
import numpy as np
from time import perf_counter as pf
from pathlib import Path
import argparse

import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dynamic",
    action="store_true",
    dest="dynamic",
    help="Display a dynamic graph (no save during execution)",
)
parser.add_argument(
    "--image",
    action="store_true",
    dest="image",
    help="Save data and images",
)
args = parser.parse_args()
DYNAMIC = args.dynamic
IMAGE = args.image

if DYNAMIC:
    print("Dynamic display (no save during execution)")
else:
    print("Saves during execution in `results` folder and `evolution` folder")

PATH_MODELS = Path().absolute() / "models"
PATH_FINAL = Path().absolute() / "final"


policy_DQN = DQN(HEIGHT, WIDTH, N_ACTIONS).to(device)
target_DQN = DQN(HEIGHT, WIDTH, N_ACTIONS).to(device)
target_DQN.load_state_dict(policy_DQN.state_dict())

# optimizer = optim.Adam(policy_DQN.parameters(), lr=LEARNING_RATE)
optimizer = optim.SGD(
    policy_DQN.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True
)

memory = ReplayMemory(REPLAY_MEMORY_SIZE, BATCH_SIZE)


display = Display(DYNAMIC, IMAGE)
# show = (lambda: display.show()) if display.dynamic else (lambda: None)
show = lambda: None
save = (lambda: None) if display.dynamic else (lambda: display.save())

optimization = lambda it, r: it % K_FRAME and r  # or r in (-10, 50, 200)

dmaker = DecisionMaker(0, policy_DQN)
episodes = 0
learn_counter = 0
last_action = 0

REWARDS = {
    "default": -0.2,
    200: 20,
    50: 15,
    10: 10,
    0: 0,
    "lose": -1,
    "win": 5,
    "reverse": -6,
}
REVERSED = {0: 9, 1: 4, 2: 3, 3: 2, 4: 1, 5: 9, 6: 7, 7: 6, 8: 5}
isreversed = (
    lambda last_action, action: "default" if REVERSED[action] - last_action else "reverse"
)

one_game = []
best_score = 0

# Main loop
while True:
    if dmaker.steps_done > MAX_FRAMES:
        break
    episodes += 1

    obs = env.reset()
    lives = 3
    jump_dead_step = False

    # Avoid beginning steps of the game
    for i_step in range(AVOIDED_STEPS - 5):
        obs, reward, done, info = env.step(0)

    observations = np.stack([unit_prepr_obs(env.step(0)[0]) for i_step in range(4)][::-1])

    obs, reward, done, info = env.step(0)
    state = preprocessing_observation(observations, obs)

    got_reward = False

    one_game.clear()

    no_move_count = 0
    for t in count():
        if dmaker.steps_done > MAX_FRAMES:
            break
        # epsilon greedy decision maker
        action = dmaker.select_action(state, policy_DQN, display, learn_counter)
        action_ = action.item()

        obs, reward_, done, info = env.step(action_)
        display.update_obs(obs)
        one_game.append(obs)

        reward = min(reward_,1)
        # reward = REWARDS["default"]  # isreversed(last_action, action_)
        # if reward_ not in REWARDS:
        #     print(reward_)
        # reward += REWARDS[reward_] if reward_ in REWARDS else reward_ / 10

        if info["lives"] < lives:
            lives -= 1
            jump_dead_step = True
            got_reward = False
            reward += REWARDS["lose"]

        if done and lives > 0:
            reward += REWARDS["win"]

        last_action = action_
        got_reward = got_reward or reward != 0
        display.add_reward(reward)
        reward = torch.tensor([reward], device=device)

        next_state = preprocessing_observation(observations, obs)

        if got_reward:
            memory.push(state, action, reward, next_state, done)

        state = next_state
        if optimization(dmaker.steps_done, got_reward):
            learn_counter = optimize_model(
                policy_DQN, target_DQN, memory, optimizer, display, learn_counter, device
            )

        show()
        if done:
            display.successes += info["lives"] > 0
            break
        if jump_dead_step:
            for i_dead in range(DEAD_STEPS):
                obs, reward, done, info = env.step(0)
            jump_dead_step = False

    if best_score < sum(display.rewards):
        best_score = sum(display.rewards)
        frameSize = (160, 210)
        out = cv2.VideoWriter(
            str(PATH_FINAL / "best_video.avi"), cv2.VideoWriter_fourcc(*"DIVX"), 15, frameSize
        )
        for img in one_game:
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.release()

    if dmaker.steps_done % TARGET_UPDATE == 0:
        target_DQN.load_state_dict(policy_DQN.state_dict())

    if episodes % SAVE_MODEL == 0:
        torch.save(policy_DQN.state_dict(), PATH_MODELS / f"policy-model-{episodes}.pt")
        torch.save(target_DQN.state_dict(), PATH_MODELS / f"target-model-{episodes}.pt")
        save()

    display.new_episode()

torch.save(policy_DQN.state_dict(), PATH_MODELS / f"policy-model-final.pt")
torch.save(target_DQN.state_dict(), PATH_MODELS / f"target-model-final.pt")
print("Complete")
