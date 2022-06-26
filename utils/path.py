from pathlib import Path

ABS_PATH = Path().absolute()
RESULTS_PATH = ABS_PATH / "results"


def create(path):
    if not (path.exists()):
        path.mkdir()
        print('Folder "{}" created.'.format(path))


def working_path(isstreaming, offset=0):
    if isstreaming:
        working_dir = RESULTS_PATH / f"training-only-stream"
    else:
        folders = list(RESULTS_PATH.iterdir())
        index = 1
        while any(RESULTS_PATH.glob(f"training-{index}*")):
            index += 1

        index -= offset
        working_dir = RESULTS_PATH / f"training-{index}"
    return working_dir
