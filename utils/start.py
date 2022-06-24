from pathlib import Path

ABS_PATH = Path().absolute()
RESULTS_PATH = ABS_PATH / "results"
SUBFOLDERS = ["models", "plots", "recorded-data"]  # + ["best-run"]


def create(path):
    if not (path.exists()):
        path.mkdir()
        print('Folder "{}" created.'.format(path))
