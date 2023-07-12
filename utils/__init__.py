from .path import RESULTS_PATH, working_path
from .rewards import transform_reward, REWARDS
from .actions import ACTIONS, REVERSED
from .save_functions import save_model, save_plot

from collections import namedtuple
try:
    from rich import print 
except ImportError:
    import warnings
    warnings.warn("If you want colors, you must install rich (pip install rich)", UserWarning, 2)

Paths = namedtuple("Paths", ("path_models", "path_plots", "path_data"))

SUBFOLDERS = ["models", "plots", "recorded-data"]

def start(args):
    RESULTS_PATH.mkdir(exist_ok=True)
    working_dir = working_path(args.stream)
    working_dir.mkdir(exist_ok=True)
    for subfolder in SUBFOLDERS:
        (working_dir / subfolder).mkdir(exist_ok=True)

    make_path = lambda subfolder: working_dir / subfolder
    path_models, path_plots, path_data = map(make_path, SUBFOLDERS)

    if args.stream:
        print(
            "Streaming display (no image saved " +
            " or data saved during execution)"
        )
    elif args.image:
        message = (
            "Saves during execution :\n" +
            f"\t       Models : {path_models}" + "\n" +
            f"\tRecorded data : {path_data}" + "\n" +
            f"\t        Plots : {path_plots}"
        )
        print(message)
    else:
        message = (
            "Saves during execution :\n" +
            f"\t       Models : {path_models}" + "\n" + 
            f"\tRecorded data : {path_data}"
        )
        print(message)
    return Paths(path_models, path_plots, path_data)
