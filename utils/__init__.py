from .path import RESULTS_PATH, working_path
from .utils import transform_reward, REWARDS, ACTIONS, REVERSED

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
    return path_models, path_plots, path_data
