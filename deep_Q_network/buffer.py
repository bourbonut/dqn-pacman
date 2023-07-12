import numpy as np

from dataclasses import dataclass, field
import statistics, pickle, json

try:
    from rich import print 
except ImportError:
    import warnings
    warnings.warn("If you want colors, you must install rich (pip install rich)", UserWarning, 2)

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

    def save(self, path):
        with open(path / f"episode-{self.episodes}.pkl", "wb") as file:
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
