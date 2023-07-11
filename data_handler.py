from matplotlib import pyplot as plt
import torch, gym
from torch import optim
import numpy as np
from PIL import Image

from deep_Q_network import parameters as params
from deep_Q_network.parameters import EPS_MAX, EPS_MIN, EPS_DECAY
from deep_Q_network import device, init_obs, preprocess_observation
from deep_Q_network import DQN, DecisionMaker, ReplayMemory, ALEInterface, Pacman
from utils import start, REWARDS, ACTIONS, REVERSED, transform_reward
from utils.parser import args

from dataclasses import dataclass, field
import statistics, random, pickle, json, base64, io

Y_LABELS = (
    "Loss per optimization",
    "Average of rewards per episode",
    "Average of max predicted Q value",
    "Rewards per action",
    "Total of rewards per episode",
    "Total of max predicted Q value",
)

def save_model(model, name, version):
    torch.save(model, PATH_MODELS / f"{name}-model-{version}.pt")
    print(f"Model \"{name}\" (version {version}) saved.")

def save_plot(buffer):
    fig, axis = plt.subplots(2, 3, figsize=(16, 10))
    axis = axis.flatten()
    for ax, data in zip(axis, buffer):
        ax.plot(range(len(data)), data)
    for label, ax in zip(Y_LABELS, axis):
        ax.set_ylabel(label)
    episodes = buffer.episodes
    successes = buffer.successes
    fig.suptitle(f"Episode {episodes} | Total of successes = {successes}")
    fig.tight_layout()
    plt.savefig(PATH_PLOTS / f"episode-{buffer.episodes}.png")
    print(f"Figure \"{buffer.episodes}.png\" saved.")
    for axis in axis:
        axis.cla()

@dataclass
class GeneralData:
    """
    Simple class to hold data and compute mean and total of values.
    """
    raw: list = field(default_factory=list)
    mean: list = field(default_factory=list)
    total: list = field(default_factory=list)

    def compute_mean(self):
        mean = statistics.mean(self.raw) if len(self.raw) > 0 else 0
        self.mean.append(mean)

    def compute_total(self):
        self.total.append(sum(self.raw))

    def clear(self):
        self.raw = []

    def append(self, item):
        self.raw.append(item)

    def moving_avg(self, t):
        values = (
            [0] * (t - len(self.total)) + self.total
            if len(self.total) < t
            else self.total[-t:]
        )
        self.mean.append(statistics.mean(values))

@dataclass
class Buffer:
    """
    Class to store data to communicate between threads
    """
    image: np.array = field(default_factory=lambda: np.array([]))
    rewards: GeneralData = field(default_factory=GeneralData)
    qvalues: GeneralData = field(default_factory=GeneralData)
    losses: GeneralData = field(default_factory=GeneralData)
    episodes: int = field(default_factory=int)
    successes: int = field(default_factory=int)

    def update(self):
        self.episodes += 1

        self.rewards.moving_avg(20)
        self.rewards.compute_total()
        self.qvalues.compute_mean()
        self.qvalues.compute_total()

        self.losses.clear()
        self.rewards.clear()
        self.qvalues.clear()

    def __iter__(self):
        yield self.losses.raw
        yield self.rewards.mean
        yield self.qvalues.mean
        yield self.rewards.raw
        yield self.rewards.total
        yield self.qvalues.total

    def save(self):
        with open(PATH_DATA / f"episode-{self.episodes}.pkl", "wb") as file:
            pickle.dump(list(self) + [self.successes], file)
        print(f"Episode {self.episodes} saved.")

    def parse(self, data):
        return {
            "x": list(range(len(data))),
            "y": data,
            "xmax": len(data),
            "ymin": min(data, default=0),
            "ymax": max(data, default=0)
        }

    def json(self):   
        alpha = np.ones((210, 160, 1), dtype=np.int8) * 255
        img = np.concatenate((self.image, alpha), axis=-1).reshape(210 * 160 * 4)
        data = {
            "image": img.tolist(),
            "losses_raw":    self.parse(self.losses.raw),
            "rewards_mean":  self.parse(self.rewards.mean),
            "qvalues_mean":  self.parse(self.qvalues.mean),
            "rewards_raw":   self.parse(self.rewards.raw),
            "rewards_total": self.parse(self.rewards.total),
            "qvalues_total": self.parse(self.qvalues.total),
        }
        return json.dumps(data)

class DataHandler:

    def __init__(self, env, policy, target, memory, buffer):
        # Arguments
        self.env = env
        self.policy = policy
        self.target = target
        self.memory = memory
        self.buffer = buffer
        self.buffer.episodes = 1
 
        # Common variables
        self.episodes = 0
        self.learn_counter = 0
        self.best_score = 0
        self.lives = 3
        self.jump_dead_step = False
        self.old_action = 3
        self.steps_done = 0

        # Set optimizer
        self.optimizer = optim.SGD(
            self.policy.parameters(),
            lr=params.LEARNING_RATE,
            momentum=params.MOMENTUM,
            nesterov=True,
        )
    
    def optimization(self, reward):
        return self.steps_done % params.K_FRAME == 0 and reward # or reward in (-10, 50, 200)

    def avoid_beginning_steps(self):
        for i_step in range(params.AVOIDED_STEPS):
            obs, reward, done, info = self.env.step(3)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = max(
            EPS_MIN,
            EPS_MAX - (EPS_MAX - EPS_MIN) * self.learn_counter / EPS_DECAY
        )
        self.steps_done += 1
        with torch.no_grad():
            q_values = self.policy(state)
        self.buffer.qvalues.append(q_values.max(1)[0].item())
        if sample > eps_threshold:
            # Optimal action
            return q_values.max(1)[1].view(1, 1)
        else:
            # Random action
            action = random.randrange(params.N_ACTIONS)
            while action == REVERSED[self.old_action]:
                action = random.randrange(params.N_ACTIONS)
            return torch.tensor([[action]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < params.BATCH_SIZE:
            return
        self.learn_counter += 1
        states, actions, rewards, next_states, dones = self.memory.sample()

        predicted_targets = self.policy(states).gather(1, actions)

        target_values = self.target(next_states).detach().max(1)[0]
        labels = rewards + params.DISCOUNT_RATE * (1 - dones.squeeze(1)) * target_values

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(predicted_targets, labels.detach().unsqueeze(1)).to(device)
        self.buffer.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
           param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
     
    def run(self):
        while True:
            if self.steps_done > params.MAX_FRAMES:
                save_model(self.policy.state_dict(), "policy", self.episodes)
                save_model(self.target.state_dict(), "target", self.episodes)
                break
            for _ in self.run_one_episode():
                # yield self.buffer
                yield

    def run_one_episode(self):
        self.episodes += 1
        obs = self.env.reset()
        lives = 3
        jump_dead_step = False

        # Avoid beginning steps of the game
        self.avoid_beginning_steps()
        # Initialization first observations
        observations = init_obs(self.env)
        obs, reward, done, info = self.env.step(3)
        state = preprocess_observation(observations, obs)

        got_reward = False
        old_action = 3
        no_move_count = 0
        while True:
            if self.steps_done > params.MAX_FRAMES:
                break
            # epsilon greedy decision maker
            action = self.select_action(state)
            action_ = ACTIONS[old_action][action.item()]

            obs, reward_, done, info = self.env.step(action_)
            self.buffer.image = obs.copy()
            reward = transform_reward(reward_)

            # update_all = False
            if info["lives"] < lives:
                lives -= 1
                jump_dead_step = True
                got_reward = False
                reward += REWARDS["lose"]
                self.old_action = 3
                # update_all = True

            if done and lives > 0:
                reward += REWARDS["win"]

            got_reward = got_reward or reward != 0
            self.buffer.rewards.append(reward)
            reward = torch.tensor([reward], device=device)

            old_action = action_
            if reward != 0:
                self.old_action = action.item()

            next_state = preprocess_observation(observations, obs)

            if got_reward:
                self.memory.push(state, action, reward, next_state, done)

            state = next_state
            if self.optimization(got_reward):
                self.optimize_model()

            if self.steps_done % params.TARGET_UPDATE == 0:
                self.target.load_state_dict(self.policy.state_dict())

            # display.stream(update_all)
            if done:
                self.buffer.successes += info["lives"] > 0
                break
            if jump_dead_step:
                for i_dead in range(params.DEAD_STEPS):
                    obs, reward, done, info = self.env.step(0)
                jump_dead_step = False
            yield


        if self.episodes % params.SAVE_MODEL == 0:
            save_model(self.policy.state_dict(), "policy", self.episodes)
            save_model(self.target.state_dict(), "target", self.episodes)
            save_plot(self.buffer)
            buffer.save()

        self.buffer.update()
        yield


if __name__ == "__main__":
    PATH_MODELS, PATH_PLOTS, PATH_DATA = start(args)
    # Set environment
    ale = ALEInterface()
    ale.loadROM(Pacman)
    env = gym.make("MsPacman-v0")

    policy = DQN(params.N_ACTIONS).to(device)
    target = DQN(params.N_ACTIONS).to(device)
    memory = ReplayMemory(params.REPLAY_MEMORY_SIZE, params.BATCH_SIZE)

    buffer = Buffer()
    datahandler = DataHandler(env, policy, target, memory, buffer)
    generator = datahandler.run()
    # for i in range(10_000):
    #     next(generator)
    # next_buffer = next(generator)
    # print(buffer)
