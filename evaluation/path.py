from .parser import args
from pathlib import Path


def dir_path(path):
    if path == "" or Path(path).exists():
        return path
    else:
        raise NotADirectoryError(path)


path = dir_path(args.path)
ABS_PATH = Path().absolute()
if path == "":
    RESULTS_PATH = ABS_PATH / "results"
    recent = lambda folder: folder.stat().st_mtime
    WORKING_DIRECTORY = max(RESULTS_PATH.iterdir(), key=recent)
else:
    WORKING_DIRECTORY = Path(path)
