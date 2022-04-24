from matplotlib import pyplot as plt
from pathlib import Path
import statistics
import pickle


class Display:

    PATH_PLOT = Path().absolute() / "evolution"
    PATH_RESULTS = Path().absolute() / "results"
    Y_LABELS = (
        "Loss per optimization",
        "Average of rewards per episode",
        "Average of max predicted Q value",
        "Rewards per action",
        "Total of rewards per episode",
        "Total of max predicted Q value",
    )

    def __init__(self, dynamic=True, image=False):
        self.iterations = [[], [], [], []]
        self.losses = []
        self.rewards = []
        self.mean_rewards = []
        self.total_rewards = []
        self.q_values = []
        self.total_q_values = []
        self.mean_q_values = []
        self.ep = 0
        self.successes = 0
        self.dynamic = dynamic
        if dynamic:
            self.fig = plt.figure(figsize=(20, 8))
            self.axis = [
                self.fig.add_subplot(2, 4, 1),
                self.fig.add_subplot(2, 4, 2),
                self.fig.add_subplot(2, 4, 3),
                self.fig.add_subplot(2, 4, 5),
                self.fig.add_subplot(2, 4, 6),
                self.fig.add_subplot(2, 4, 7),
                self.fig.add_subplot(1, 4, 4),
            ]
            self.fig.tight_layout()
        else:
            self.fig, self.axis = plt.subplots(2, 3, figsize=(16, 10))
            self.fig.tight_layout()
            self.axis = self.axis.flatten()
        self.save = self.save_all if image else self.save_data

    def new_episode(self):
        self.ep += 1
        for i in range(2):
            self.iterations[i].clear()

        self.mean_rewards.append(statistics.mean(self.rewards))
        self.total_rewards.append(sum(self.rewards))
        self.iterations[2].append(len(self.total_rewards))

        self.mean_q_values.append(statistics.mean(self.q_values))
        self.total_q_values.append(sum(self.q_values))
        self.iterations[3].append(len(self.mean_q_values))

        self.losses.clear()
        self.rewards.clear()
        self.q_values.clear()

    def add_loss(self, loss):
        self.losses.append(loss)
        self.iterations[0].append(len(self.losses))

    def add_reward(self, reward):
        self.rewards.append(reward)
        self.iterations[1].append(len(self.rewards))

    def add_q_value(self, q_value):
        self.q_values.append(q_value)

    def update_obs(self, obs):
        self.obs = obs

    def show(self):
        plt.ion()
        self.axis[0].plot(self.iterations[0], self.losses)
        self.axis[1].plot(self.iterations[2], self.mean_rewards)
        self.axis[2].plot(self.iterations[3], self.mean_q_values)
        self.axis[3].plot(self.iterations[1], self.rewards)
        self.axis[4].plot(self.iterations[2], self.total_rewards)
        self.axis[5].plot(self.iterations[3], self.total_q_values)
        for label, axis in zip(self.Y_LABELS, self.axis[:-1]):
            axis.set_ylabel(label)
        self.axis[6].imshow(self.obs)
        self.fig.suptitle(f"Episode {self.ep + 1} | Total of successes = {self.successes}")
        plt.draw()
        plt.pause(0.0001)
        for axis in self.axis:
            axis.cla()

    def save_all(self):
        self.axis[0].plot(self.iterations[0], self.losses)
        self.axis[1].plot(self.iterations[2], self.mean_rewards)
        self.axis[2].plot(self.iterations[3], self.mean_q_values)
        self.axis[3].plot(self.iterations[1], self.rewards)
        self.axis[4].plot(self.iterations[2], self.total_rewards)
        self.axis[5].plot(self.iterations[3], self.total_q_values)
        for label, axis in zip(self.Y_LABELS, self.axis):
            axis.set_ylabel(label)
        self.fig.suptitle(f"Episode {self.ep + 1} | Total of successes = {self.successes}")
        self.fig.tight_layout()
        plt.savefig(self.PATH_PLOT / f"episode-{self.ep + 1}.png")
        for axis in self.axis:
            axis.cla()
        data = [
            self.losses,
            self.mean_rewards,
            self.mean_q_values,
            self.rewards,
            self.total_rewards,
            self.total_q_values,
            self.successes,
        ]
        with open(self.PATH_RESULTS / f"episode-{self.ep + 1}.pkl", "wb") as file:
            pickle.dump(data, file)

        print(f"Episode {self.ep + 1} saved")

    def save_data(self):
        data = [
            self.losses,
            self.mean_rewards,
            self.mean_q_values,
            self.rewards,
            self.total_rewards,
            self.total_q_values,
            self.successes,
        ]
        with open(self.PATH_RESULTS / f"episode-{self.ep + 1}.pkl", "wb") as file:
            pickle.dump(data, file)

        print(f"Episode {self.ep + 1} saved")
