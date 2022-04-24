from pathlib import Path

ABS_PATH = Path().absolute()
RESULTS_PATH = ABS_PATH / "results"
SUBFOLDERS = ["models", "plots", "recorded-data"]

if not (RESULTS_PATH.exists()):
    RESULTS_PATH.mkdir()

folders = list(RESULTS_PATH.iterdir())
index = 1
while any(RESULTS_PATH.glob(f"training-{index}*")):
    index += 1

WORKING_DIRECTORY = RESULTS_PATH / f"training-{index}"
WORKING_DIRECTORY.mkdir()
for subfolder in SUBFOLDERS:
    (WORKING_DIRECTORY / subfolder).mkdir()

make_path = lambda subfolder: WORKING_DIRECTORY / subfolder
PATH_MODELS, PATH_PLOTS, PATH_DATA = map(make_path, SUBFOLDERS)
