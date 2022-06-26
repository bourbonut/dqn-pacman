from .path import *
from .utils import transform_reward, REWARDS, ACTIONS, REVERSED
from .display import Display

SUBFOLDERS = ["models", "plots", "recorded-data"]

def start(args):
    create(RESULTS_PATH)
    working_dir = working_path(args.stream)
    create(working_dir)
    for subfolder in SUBFOLDERS:
        create(working_dir / subfolder)

    make_path = lambda subfolder: working_dir / subfolder
    PATH_MODELS, PATH_PLOTS, PATH_DATA = map(make_path, SUBFOLDERS)

    if args.stream:
        print("Streaming display (no image or data saved during execution)")
    elif args.image:
        message = "Saves during execution :\n"
        message += "\t       Models : {}".format(PATH_MODELS) + "\n"
        message += "\tRecorded data : {}".format(PATH_DATA) + "\n"
        message += "\t        Plots : {}".format(PATH_PLOTS)
        print(message)
    else:
        message = "Saves during execution :\n"
        message += "\t       Models : {}".format(PATH_MODELS) + "\n"
        message += "\tRecorded data : {}".format(PATH_DATA)
        print(message)
    return PATH_MODELS
