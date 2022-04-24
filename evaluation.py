from pathlib import Path
import argparse
from matplotlib import pyplot as plt
import pickle
from agent import *
import torch
import cv2
from itertools import count
import numpy as np
import random
import re

PATH = Path().absolute()
NUM = re.compile(r"episode-(\d+).pkl")


def f(values, n):
    offset = (n - 1) // 2
    return [
        sum(values[i - offset : i + offset + 1]) / n
        for i in range(offset, len(values) - offset)
    ]


def only_rewards(ep):
    Y_LABELS = (
        "Average of rewards per episode",
        "Total of rewards per episode",
    )

    with open(PATH / "results" / f"episode-{ep}.pkl", "rb") as file:
        data = pickle.load(file)

    iterations = list(map(range, map(len, data[:-1])))

    fig, axis = plt.subplots(2, 1, figsize=(16, 10))
    successes = data[-1]

    for i, it in enumerate((1, 4)):
        axis[i].plot(iterations[it][4:], data[it][4:])
        axis[i].plot(iterations[it][4:-16], f(data[it][4:], 17))
    for label, axis in zip(Y_LABELS, axis):
        axis.set_ylabel(label)
    fig.suptitle(f"Episode {ep} | Total of successes = {successes}")
    fig.tight_layout()
    plt.savefig(PATH / "final" / "rewards.png")


def only_q_values(ep):
    Y_LABELS = (
        "Average of max predicted Q value",
        "Total of max predicted Q value",
    )

    with open(PATH / "results" / f"episode-{ep}.pkl", "rb") as file:
        data = pickle.load(file)

    iterations = list(map(range, map(len, data[:-1])))

    fig, axis = plt.subplots(2, 1, figsize=(16, 10))
    successes = data[-1]

    for i, it in enumerate((2, 5)):
        axis[i].plot(iterations[it][4:], data[it][4:])
        axis[i].plot(iterations[it][4:-16], f(data[it][4:], 17))
    for label, axis in zip(Y_LABELS, axis):
        axis.set_ylabel(label)
    fig.suptitle(f"Episode {ep} | Total of successes = {successes}")
    fig.tight_layout()
    plt.savefig(PATH / "final" / "q_values.png")


def load_save_result(ep):
    Y_LABELS = (
        "Loss per optimization",
        "Average of rewards per episode",
        "Average of max predicted Q value",
        "Rewards per action",
        "Total of rewards per episode",
        "Total of max predicted Q value",
    )

    with open(PATH / "results" / f"episode-{ep}.pkl", "rb") as file:
        data = pickle.load(file)

    iterations = list(map(range, map(len, data[:-1])))

    fig, axis = plt.subplots(2, 3, figsize=(16, 10))
    axis = axis.flatten()

    successes = data[-1]

    for i in range(len(iterations)):
        if i in (1, 2, 4, 5):
            axis[i].plot(iterations[i][4:], data[i][4:])
        else:
            axis[i].plot(iterations[i], data[i])
    for label, axis in zip(Y_LABELS, axis):
        axis.set_ylabel(label)
    fig.suptitle(f"Episode {ep} | Total of successes = {successes}")
    fig.tight_layout()
    plt.savefig(PATH / "final" / "result.png")


def record(ep):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQN(HEIGHT, WIDTH, N_ACTIONS).to(device)

    agent.load_state_dict(
        torch.load(str(PATH / "models" / f"policy-model-{ep}.pt"), map_location=device)
    )

    dmaker = DecisionMaker(0, agent)
    obs = env.reset()

    frameSize = (160, 210)
    out = cv2.VideoWriter(
        str(PATH / "final" / "output_video.avi"), cv2.VideoWriter_fourcc(*"DIVX"), 15, frameSize
    )

    # Avoid beginning steps of the game
    for i_step in range(AVOIDED_STEPS - 5):
        obs, reward, done, info = env.step(0)

    observations = np.stack([unit_prepr_obs(env.step(0)[0]) for i_step in range(4)])

    obs, reward, done, info = env.step(0)

    out.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    for t in count():
        state = preprocessing_observation(observations, obs)
        sample = random.random()
        eps_threshold = EPS_MIN
        with torch.no_grad():
            q_values = agent(state)
        if sample > eps_threshold:
            action = q_values.max(1)[1].view(1, 1)
        else:
            random_action = [[random.randrange(N_ACTIONS)]]
            action = torch.tensor(random_action, device=device, dtype=torch.long)

        obs, reward, done, info = env.step(action)
        out.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        if done:
            break

    out.release()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--record",
    action="store_true",
    dest="record",
    help="Record a game of the agent given the `epsisode` value.",
)
parser.add_argument(
    "--reward",
    action="store_true",
    dest="reward",
    help="Load rewards and save graph in `final` folder.",
)
parser.add_argument(
    "--qvalue",
    action="store_true",
    dest="qvalue",
    help="Load Q values and save graph in `final` folder.",
)
parser.add_argument(
    "--all",
    action="store_true",
    dest="all",
    help="Do everything (save results and record the agent movements)",
)
parser.add_argument(
    "episode",
    help="episode value : 20 to maximum episode with a step 20 (`final` to get the final one).",
)
args = parser.parse_args()

if args.episode == "last":
    get_num = lambda s: int(NUM.search(str(s))[1])
    selected_episode = max(map(get_num, (PATH / "results").iterdir()))
else:
    selected_episode = args.episode

print("Episode '{}'".format(selected_episode))

if args.all:
    if selected_episode == "final":
        raise Exception("To get a graph, `final` doesn't exist. Use `last` or a number")
    print("Recording in `final` folder ...")
    record(selected_episode)
    print("Saving rewards in `final` folder ...")
    only_rewards(selected_episode)
    print("Saving Q values in `final` folder ...")
    only_q_values(selected_episode)
    print("Saving result in `final` folder ...")
    load_save_result(selected_episode)
    print("Finished.")
else:
    if args.record:
        print("Recording in `final` folder ...")
        record(selected_episode)
        print("Finished.")
    if args.reward:
        if selected_episode == "final":
            raise Exception("To get a graph, `final` doesn't exist. Use `last` or a number")
        print("Saving rewards in `final` folder ...")
        only_rewards(selected_episode)
        print("Finished.")
    if args.qvalue:
        if selected_episode == "final":
            raise Exception("To get a graph, `final` doesn't exist. Use `last` or a number")
        print("Saving Q values in `final` folder ...")
        only_q_values(selected_episode)
        print("Finished.")
    if not (args.record):
        if selected_episode == "final":
            raise Exception("To get a graph, `final` doesn't exist. Use `last` or a number")
        print("Saving result in `final` folder ...")
        load_save_result(selected_episode)
        print("Finished.")
