from pathlib import Path
from .parser import args

ABS_PATH = Path().absolute()
RESULTS_PATH = ABS_PATH / "results"
SUBFOLDERS = ["models", "plots", "recorded-data"]  # + ["best-run"]


def create(path):
    if not (path.exists()):
        path.mkdir()
        print('Folder "{}" created.'.format(path))


create(RESULTS_PATH)

# if args.dynamic:
if True:
    WORKING_DIRECTORY = RESULTS_PATH / f"training-only-dynamic"
    create(WORKING_DIRECTORY)
    for subfolder in SUBFOLDERS:
        create(WORKING_DIRECTORY / subfolder)
else:
    folders = list(RESULTS_PATH.iterdir())
    index = 1
    while any(RESULTS_PATH.glob(f"training-{index}*")):
        index += 1

    WORKING_DIRECTORY = RESULTS_PATH / f"training-{index}"
    WORKING_DIRECTORY.mkdir()
    print('Folder "{}" created.'.format(WORKING_DIRECTORY))
    for subfolder in SUBFOLDERS:
        create(WORKING_DIRECTORY / subfolder)


make_path = lambda subfolder: WORKING_DIRECTORY / subfolder
PATH_MODELS, PATH_PLOTS, PATH_DATA = map(make_path, SUBFOLDERS)
