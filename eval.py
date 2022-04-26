from evaluation import *
import re


NUM = re.compile(r"episode-(\d+).pkl")

if args.episode == "last":
    get_num = lambda s: int(NUM.search(str(s))[1])
    selected_episode = max(map(get_num, (WORKING_DIRECTORY / "recorded-data").iterdir()))
else:
    selected_episode = args.episode

print('Opening data of episode {} in "{}"'.format(selected_episode, WORKING_DIRECTORY))

if args.all:
    record(selected_episode)
    only_rewards(selected_episode)
    only_q_values(selected_episode)
    load_save_result(selected_episode)
else:
    if args.record:
        record(selected_episode)
    elif args.reward:
        only_rewards(selected_episode)
    elif args.qvalue:
        only_q_values(selected_episode)
    elif not (args.record):
        load_save_result(selected_episode)
