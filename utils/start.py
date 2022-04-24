from pathlib import Path

ABS_PATH = Path().absolute()
RESULTS_PATH = ABS_PATH / "results"
SUBFOLDERS = ["models", "plots", "recorded-data"]  # + ["best-run"]


def create(path):
    if not (path.exists()):
        path.mkdir()


create(RESULTS_PATH)

# folders = list(RESULTS_PATH.iterdir())
# index = 1
# while any(RESULTS_PATH.glob(f"training-{index}*")):
#     index += 1
#
# WORKING_DIRECTORY = RESULTS_PATH / f"training-{index}"
# WORKING_DIRECTORY.mkdir()
# for subfolder in SUBFOLDERS:
#     (WORKING_DIRECTORY / subfolder).mkdir()

WORKING_DIRECTORY = RESULTS_PATH / f"training"
create(WORKING_DIRECTORY)
for subfolder in SUBFOLDERS:
    create(WORKING_DIRECTORY / subfolder)


make_path = lambda subfolder: WORKING_DIRECTORY / subfolder
PATH_MODELS, PATH_PLOTS, PATH_DATA = map(make_path, SUBFOLDERS)
