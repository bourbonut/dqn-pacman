from pathlib import Path


def dir_path(path):
    if path == "" or Path(path).exists():
        return path
    else:
        raise NotADirectoryError(path)
