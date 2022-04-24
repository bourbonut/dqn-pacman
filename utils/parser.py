import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dynamic",
    action="store_true",
    dest="dynamic",
    help="Display a dynamic graph (no save during execution)",
)
parser.add_argument(
    "--image",
    action="store_true",
    dest="image",
    help="Save data and images",
)
args = parser.parse_args()
