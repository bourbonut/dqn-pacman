import streamlit as st
from deep_Q_network import *
from utils import *
from utils.parser import args

torch.autograd.set_detect_anomaly(True)
optimization = lambda it, r: it % K_FRAME == 0 and r  # or r in (-10, 50, 200)

episodes = 0
learn_counter = 0
best_score = 0

if args.stream:
    print("Streaming display (no image or data saved during execution)")
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

create(RESULTS_PATH)

if args.stream:
    WORKING_DIRECTORY = RESULTS_PATH / f"training-only-stream"
    create(WORKING_DIRECTORY)
    for subfolder in SUBFOLDERS:
        create(WORKING_DIRECTORY / subfolder)
else:
    folders = list(RESULTS_PATH.iterdir())
    index = 1
    while any(RESULTS_PATH.glob(f"training-{index}*")):
        index += 1

    WORKING_DIRECTORY = RESULTS_PATH / f"training-{index}"
    WORKING_DIRECTORY.mkdir()
    print('Folder "{}" created.'.format(WORKING_DIRECTORY))
    for subfolder in SUBFOLDERS:
        create(WORKING_DIRECTORY / subfolder)

# Set environment
ale = ALEInterface()
ale.loadROM(Pacman)

env = gym.make("MsPacman-v0")

# Set neural networks
policy_DQN = DQN(N_ACTIONS).to(device)
target_DQN = DQN(N_ACTIONS).to(device)
target_DQN.load_state_dict(policy_DQN.state_dict())

# Set optimizer
# optimizer = optim.Adam(policy_DQN.parameters(), lr=LEARNING_RATE)
optimizer = optim.SGD(
    policy_DQN.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True
)
# optimizer = optim.RMSprop(policy_DQN.parameters(), lr=LEARNING_RATE)

# Set memory
memory = ReplayMemory(REPLAY_MEMORY_SIZE, BATCH_SIZE)

# Set decision maker
dmaker = DecisionMaker(0, policy_DQN)
make_path = lambda subfolder: WORKING_DIRECTORY / subfolder
PATH_MODELS, PATH_PLOTS, PATH_DATA = map(make_path, SUBFOLDERS)
display = Display(args.stream, args.image)

# one_game = [] # useful to save a video

# Main loop
while True:
    if dmaker.steps_done > MAX_FRAMES:
        break
    episodes += 1

    obs = env.reset()
    lives = 3
    jump_dead_step = False
    old_action = 0

    # Avoid beginning steps of the game
    for i_step in range(AVOIDED_STEPS):
        obs, reward, done, info = env.step(3)

    observations = init_obs(env)
    obs, reward, done, info = env.step(3)
    state = preprocess_observation(observations, obs)

    got_reward = False

    old_action = 3

    no_move_count = 0
    while True:
        if dmaker.steps_done > MAX_FRAMES:
            break
        # epsilon greedy decision maker
        action = dmaker.select_action(state, policy_DQN, display, learn_counter)
        action_ = ACTIONS[old_action][action.item()]

        obs, reward_, done, info = env.step(action_)
        display.obs = obs.copy()
        reward = transform_reward(reward_)

        update_all = False
        if info["lives"] < lives:
            lives -= 1
            jump_dead_step = True
            got_reward = False
            reward += REWARDS["lose"]
            dmaker.old_action = 3
            update_all = True

        if done and lives > 0:
            reward += REWARDS["win"]

        got_reward = got_reward or reward != 0
        display.data.rewards.append(reward)
        reward = torch.tensor([reward], device=device)

        old_action = action_
        if reward != 0:
            dmaker.old_action = action.item()

        next_state = preprocess_observation(observations, obs)

        if got_reward:
            memory.push(state, action, reward, next_state, done)

        state = next_state
        if optimization(dmaker.steps_done, got_reward):
            learn_counter = optimize_model(
                policy_DQN, target_DQN, memory, optimizer, display, learn_counter, device
            )

        if dmaker.steps_done % TARGET_UPDATE == 0:
            target_DQN.load_state_dict(policy_DQN.state_dict())
            print("Tranfered")

        display.stream(update_all)
        if done:
            display.data.successes += info["lives"] > 0
            torch.cuda.empty_cache()
            break
        if jump_dead_step:
            for i_dead in range(DEAD_STEPS):
                obs, reward, done, info = env.step(0)
            jump_dead_step = False
        torch.cuda.empty_cache()

    if episodes % SAVE_MODEL == 0:
        torch.save(policy_DQN.state_dict(), PATH_MODELS / f"policy-model-{episodes}.pt")
        torch.save(target_DQN.state_dict(), PATH_MODELS / f"target-model-{episodes}.pt")
        display.save()

    display.data.round()

torch.save(policy_DQN.state_dict(), PATH_MODELS / f"policy-model-final.pt")
torch.save(target_DQN.state_dict(), PATH_MODELS / f"target-model-final.pt")
print("Complete")
