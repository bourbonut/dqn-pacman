# Deep Q-Network on the Atari Game Ms-Pacman

## For installation
```sh
pip install ale-py==0.7
wget http://www.atarimania.com/roms/Roms.rar
unrar e Roms.rar
unzip -qq ROMS.zip
ale-import-roms /content/ROMS/ | grep pacman

pip install -U gym
pip install -U gym[atari]
```

## For usage

In `agent` folder, you have `parameters.py` where parameters are set.
To run the code :
```sh
python main.py
```

### Dynamic display (slow)

Line 49 in `main.py` :
```python
display = Display(DYNAMIC, IMAGE)
# show = (lambda: display.show()) if display.dynamic else (lambda: None)
show = lambda: None
save = (lambda: None) if display.dynamic else (lambda: display.save())
```
Change it to :
```python
display = Display(DYNAMIC, IMAGE)
show = (lambda: display.show()) if display.dynamic else (lambda: None)
# show = lambda: None
save = (lambda: None) if display.dynamic else (lambda: display.save())
```
and run :
```sh
python main.py --dynamic
```

### Evolution step by step (a lot of memory)
To save the evolution step by step :
Run :
```sh
python main.py --image
```

## Structure of the code
```
.
├── agent
│   ├── decision.py # for decision making
│   ├── display.py # to store and to display results (rewards, losses, Q-values)
│   ├── __init__.py
│   ├── memory.py # replay memory
│   ├── model.py # dueling DQN and optimization (see the report and the class for more details)
│   ├── parameters.py # all parameters except how rewards are managed
│   ├── preprocessing.py # for preprocessing observations
│   └── __pycache__
├── clear.sh # to clear results and run a new simulation
├── evaluation.py # to evaluate the agent
├── evolution # folder for `python main.py --image` command
├── export.sh # for the server
├── final # folder where final results are stored
│   ├── best_video.avi
│   ├── output_video.avi
│   ├── q_values.png
│   ├── result.png
│   └── rewards.png
├── images # for experiment
├── lab.py # for experiment
├── main.py # main file
├── models # where models are stored (8Go normally)
├── others # folder
├── pacman-states # folder
├── quick_check.py
├── README.md
└── results # folder with pickle files
```

You have the `videos` folder that contains some videos.
