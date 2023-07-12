from matplotlib import pyplot as plt
import torch

try:
    from rich import print 
except ImportError:
    import warnings
    warnings.warn("If you want colors, you must install rich (pip install rich)", UserWarning, 2)

Y_LABELS = (
    "Loss per optimization",
    "Average of rewards per episode",
    "Average of max predicted Q value",
    "Rewards per action",
    "Total of rewards per episode",
    "Total of max predicted Q value",
)

def save_model(path, model, name, version):
    torch.save(model, path / f"{name}-model-{version}.pt")
    print(f"Model \"{name}\" (version {version}) saved.")

def save_plot(path, buffer):
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
    plt.savefig(path / f"episode-{buffer.episodes}.png")
    print(f"Figure \"episode-{buffer.episodes}.png\" saved.")
    for axis in axis:
        axis.cla()
