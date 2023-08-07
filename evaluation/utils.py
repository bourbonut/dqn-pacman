import pickle, torch, random
from utils import ACTIONS
from deep_Q_network import parameters as params
from deep_Q_network import device, init_obs, preprocess_observation
from deep_Q_network import DQN, Buffer, ALEInterface, Pacman
import gym

def moving_average(values, n):
    offset = (n - 1) // 2
    v = [values[0]] * offset + values + [values[-1]] * offset
    return [sum(v[i - offset : i + offset + 1]) / n for i in range(offset, len(v) - offset)]


def only_rewards(ep, path):
    from matplotlib import pyplot as plt

    Y_LABELS = (
        "Average of rewards per episode",
        "Total of rewards per episode",
    )

    with open(path / "recorded-data" / f"episode-{ep}.pkl", "rb") as file:
        data = pickle.load(file)

    iterations = list(map(range, map(len, data[:-1])))

    fig, axis = plt.subplots(2, 1, figsize=(16, 10))
    successes = data[-1]

    for i, it in enumerate((1, 4)):
        axis[i].plot(iterations[it][4:], data[it][4:])
        axis[i].plot(iterations[it][4:], moving_average(data[it][4:], 17))
    for label, axis in zip(Y_LABELS, axis):
        axis.set_ylabel(label)
    fig.suptitle(f"Episode {ep} | Total of successes = {successes}")
    fig.tight_layout()
    plt.savefig(path / "rewards.png")
    print('"rewards.png" saved in "{}"'.format(path))


def only_q_values(ep, path):
    from matplotlib import pyplot as plt

    Y_LABELS = (
        "Average of max predicted Q value",
        "Total of max predicted Q value",
    )

    with open(path / "recorded-data" / f"episode-{ep}.pkl", "rb") as file:
        data = pickle.load(file)

    iterations = list(map(range, map(len, data[:-1])))

    fig, axis = plt.subplots(2, 1, figsize=(16, 10))
    successes = data[-1]

    for i, it in enumerate((2, 5)):
        axis[i].plot(iterations[it][4:], data[it][4:])
        axis[i].plot(iterations[it][4:], moving_average(data[it][4:], 17))
    for label, axis in zip(Y_LABELS, axis):
        axis.set_ylabel(label)
    fig.suptitle(f"Episode {ep} | Total of successes = {successes}")
    fig.tight_layout()
    plt.savefig(path / "q_values.png")
    print('"q_values.png" saved in "{}"'.format(path))


def load_save_result(ep, path):
    from matplotlib import pyplot as plt

    Y_LABELS = (
        "Loss per optimization",
        "Average of rewards per episode",
        "Average of max predicted Q value",
        "Rewards per action",
        "Total of rewards per episode",
        "Total of max predicted Q value",
    )

    with open(path / "recorded-data" / f"episode-{ep}.pkl", "rb") as file:
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
    plt.savefig(path / "result.png")
    print('"result.png" saved in "{}"'.format(path))


def record(ep, path):
    import cv2

    # Set environment
    ale = ALEInterface()
    ale.loadROM(Pacman)
    env = gym.make("MsPacman-v0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQN(params.N_ACTIONS).to(device)

    path_model = path / "models" / f"policy-model-{ep}.pt"
    agent.load_state_dict(torch.load(str(path_model), map_location=device))
    agent.eval()

    obs = env.reset()

    frameSize = (160, 210)
    path_video = path / "output_video.avi"
    bin_loader = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(str(path_video), bin_loader, 30, frameSize)

    # Avoid beginning steps of the game
    for i_step in range(params.AVOIDED_STEPS):
        obs, reward, done, info = env.step(3)

    observations = init_obs(env)
    obs, reward, done, info = env.step(3)
    out.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    old_action = 3

    while True:
        state = preprocess_observation(observations, obs)
        sample = random.random()
        eps_threshold = params.EPS_MIN
        with torch.no_grad():
            q_values = agent(state)
        if sample > eps_threshold:
            action = q_values.max(1)[1].view(1, 1)
        else:
            random_action = [[random.randrange(params.N_ACTIONS)]]
            action = torch.tensor(random_action, device=device, dtype=torch.long)
        # action = agent(state).max(1)[1].view(1, 1)

        action_ = ACTIONS[old_action][action.item()]
        obs, reward, done, info = env.step(action_)
        out.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        old_action = action_
        if done:
            break

    out.release()
    print('"output_video.avi" saved in {}'.format(path))
