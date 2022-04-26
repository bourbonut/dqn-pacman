from .path import WORKING_DIRECTORY
from .parser import args
import pickle, torch, random

if args.record or args.all:
    from deep_Q_network import *


def f(values, n):
    offset = (n - 1) // 2
    return [
        sum(values[i - offset : i + offset + 1]) / n
        for i in range(offset, len(values) - offset)
    ]


def only_rewards(ep):
    from matplotlib import pyplot as plt

    Y_LABELS = (
        "Average of rewards per episode",
        "Total of rewards per episode",
    )

    with open(WORKING_DIRECTORY / "recorded-data" / f"episode-{ep}.pkl", "rb") as file:
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
    plt.savefig(WORKING_DIRECTORY / "rewards.png")
    print('"rewards.png" saved in "{}"'.format(WORKING_DIRECTORY))


def only_q_values(ep):
    from matplotlib import pyplot as plt

    Y_LABELS = (
        "Average of max predicted Q value",
        "Total of max predicted Q value",
    )

    with open(WORKING_DIRECTORY / "recorded-data" / f"episode-{ep}.pkl", "rb") as file:
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
    plt.savefig(WORKING_DIRECTORY / "q_values.png")
    print('"q_values.png" saved in "{}"'.format(WORKING_DIRECTORY))


def load_save_result(ep):
    from matplotlib import pyplot as plt

    Y_LABELS = (
        "Loss per optimization",
        "Average of rewards per episode",
        "Average of max predicted Q value",
        "Rewards per action",
        "Total of rewards per episode",
        "Total of max predicted Q value",
    )

    with open(WORKING_DIRECTORY / "recorded-data" / f"episode-{ep}.pkl", "rb") as file:
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
    plt.savefig(WORKING_DIRECTORY / "result.png")
    print('"result.png" saved in "{}"'.format(WORKING_DIRECTORY))


def record(ep):
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQN(HEIGHT, WIDTH, N_ACTIONS).to(device)

    path_model = WORKING_DIRECTORY / "models" / f"policy-model-{ep}.pt"
    agent.load_state_dict(torch.load(str(path_model), map_location=device))

    dmaker = DecisionMaker(0, agent)
    obs = env.reset()

    frameSize = (160, 210)
    path_video = WORKING_DIRECTORY / "output_video.avi"
    bin_loader = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(str(path_video), bin_loader, 15, frameSize)

    # Avoid beginning steps of the game
    for i_step in range(AVOIDED_STEPS):
        obs, reward, done, info = env.step(0)

    observations = init_obs(env)
    obs, reward, done, info = env.step(0)
    out.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

    while True:
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
    print('"output_video.avi" saved in {}'.format(WORKING_DIRECTORY))
