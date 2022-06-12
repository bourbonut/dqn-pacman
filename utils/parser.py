import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stream",
    action="store_true",
    dest="stream",
    help="Open a page where you can see the agent learning (no image or data saved during execution)",
)
parser.add_argument(
    "--image",
    action="store_true",
    dest="image",
    help="Save data and images",
)
args = parser.parse_args()
