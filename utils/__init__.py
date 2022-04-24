from .parser import args
from .start import PATH_PLOTS, PATH_MODELS, PATH_DATA
from .utils import transform_reward, REWARDS
from .display import Display

if args.dynamic:
    print("Dynamic display (no save during execution)")
else:
    if args.image:
        print(
            "Saves during execution in `results/models` folder,"
            + " `results/recorded-data` folder"
            + " and `results/plots`."
        )
    else:
        print(
            "Saves during execution in `results/models`"
            + " folder and `results/recorded-data` folder"
        )

display = Display(args.dynamic, args.image)
