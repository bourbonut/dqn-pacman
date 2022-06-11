from .parser import args
from .start import PATH_PLOTS, PATH_MODELS, PATH_DATA
from .utils import transform_reward, REWARDS, ACTIONS, REVERSED
from .display import Display

if args.dynamic:
    print("Dynamic display (no save during execution)")
elif args.image:
    message = "Saves during execution :\n"
    message += "\t       Models : {}".format(PATH_MODELS) + "\n"
    message += "\tRecorded data : {}".format(PATH_DATA) + "\n"
    message += "\t        Plots : {}".format(PATH_PLOTS)
    print(message)
else:
    message = "Saves during execution :\n"
    message += "\t       Models : {}".format(PATH_MODELS) + "\n"
    message += "\tRecorded data : {}".format(PATH_DATA)
    print(message)

# display = Display(args.dynamic, args.image)
display = Display(False, False)
