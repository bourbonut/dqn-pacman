from .parser import args
from .start import PATH_PLOTS, PATH_MODELS, PATH_DATA
from .utils import transform_reward

if args.dynamic:
    print("Dynamic display (no save during execution)")
else:
    print("Saves during execution in `results` folder and `evolution` folder")

display = Display(args.dynamic, args.image)
