import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    default="last",
    dest="episode",
    help="Select the episode. It should be an integer (by default it opens the last episode)",
)
parser.add_argument(
    "--path",
    dest="path",
    default="",
    help="Specify the path for the evaluation else it takes the most recent one.",
)
parser.add_argument(
    "--record",
    action="store_true",
    dest="record",
    help="Record a game of the agent given the `epsisode` value.",
)
parser.add_argument(
    "--reward",
    action="store_true",
    dest="reward",
    help="Load rewards and save graph in `final` folder.",
)
parser.add_argument(
    "--qvalue",
    action="store_true",
    dest="qvalue",
    help="Load Q values and save graph in `final` folder.",
)
parser.add_argument(
    "-a",
    "--all",
    action="store_true",
    dest="all",
    help="Do everything (save results and record the agent movements)",
)

args = parser.parse_args()
