import numpy as np
import torch
from torch import optim
import gym

from deep_Q_network import parameters as params
from deep_Q_network.parameters import EPS_MAX, EPS_MIN, EPS_DECAY
from deep_Q_network import device, init_obs, preprocess_observation
from deep_Q_network import DQN, ReplayMemory, Buffer, ALEInterface, Pacman
from utils import start, REWARDS, ACTIONS, REVERSED, transform_reward, save_model, save_plot
from utils.parser import args

import random

class DataHandler:
    """
    Main class which trains the Deep Q Network
    """

    def __init__(self, env, policy, target, memory, buffer, paths, save=False):
        # Arguments
        self.env = env
        self.policy = policy
        self.target = target
        self.memory = memory
        self.buffer = buffer
        self.buffer.episodes = 1
        self.paths = paths
        self.save = save
 
        # Common variables
        self.episodes = 0
        self.learn_counter = 0
        self.best_score = 0
        self.lives = 3
        self.jump_dead_step = False
        self.old_action = 3
        self.steps_done = 0

        # Set optimizer
        self.optimizer = optim.SGD(
            self.policy.parameters(),
            lr=params.LEARNING_RATE,
            momentum=params.MOMENTUM,
            nesterov=True,
        )
    
    def optimization(self, reward):
        return self.steps_done % params.K_FRAME == 0 and reward # or reward in (-10, 50, 200)

    def avoid_beginning_steps(self):
        for i_step in range(params.AVOIDED_STEPS):
            obs, reward, done, info = self.env.step(3)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = max(
            EPS_MIN,
            EPS_MAX - (EPS_MAX - EPS_MIN) * self.learn_counter / EPS_DECAY
        )
        self.steps_done += 1
        with torch.no_grad():
            q_values = self.policy(state)
        self.buffer.qvalues.append(q_values.max(1)[0].item())
        if sample > eps_threshold:
            # Optimal action
            return q_values.max(1)[1].view(1, 1)
        else:
            # Random action
            action = random.randrange(params.N_ACTIONS)
            while action == REVERSED[self.old_action]:
                action = random.randrange(params.N_ACTIONS)
            return torch.tensor([[action]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < params.BATCH_SIZE:
            return
        self.learn_counter += 1
        states, actions, rewards, next_states, dones = self.memory.sample()
        # self.memory.sample()

        predicted_targets = self.policy(states).gather(1, actions)

        target_values = self.target(next_states).detach().max(1)[0]
        labels = rewards + params.DISCOUNT_RATE * (1 - dones.squeeze(1)) * target_values

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(predicted_targets, labels.detach().unsqueeze(1)).to(device)
        self.buffer.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
           param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
     
    def run(self):
        while True:
            if self.steps_done > params.MAX_FRAMES:
                save_model(self.policy.state_dict(), "policy", self.episodes)
                save_model(self.target.state_dict(), "target", self.episodes)
                break
            for _ in self.run_one_episode():
                yield

    def run_one_episode(self):
        self.episodes += 1
        obs = self.env.reset()
        lives = 3
        jump_dead_step = False

        # Avoid beginning steps of the game
        self.avoid_beginning_steps()
        # Initialization first observations
        observations = init_obs(self.env)
        obs, reward, done, info = self.env.step(3)
        state = preprocess_observation(observations, obs)

        got_reward = False
        old_action = 3
        no_move_count = 0
        while True:
            if self.steps_done > params.MAX_FRAMES:
                break
            # epsilon greedy decision maker
            action = self.select_action(state)
            action_ = ACTIONS[old_action][action.item()]

            obs, reward_, done, info = self.env.step(action_)
            self.buffer.image = obs.copy()
            reward = transform_reward(reward_)

            if info["lives"] < lives:
                lives -= 1
                jump_dead_step = True
                got_reward = False
                reward += REWARDS["lose"]
                self.old_action = 3

            if done and lives > 0:
                reward += REWARDS["win"]

            got_reward = got_reward or reward != 0
            self.buffer.rewards.append(reward)
            reward = torch.tensor([reward], device=device)

            old_action = action_
            if reward != 0:
                self.old_action = action.item()

            next_state = preprocess_observation(observations, obs)

            if got_reward:
                self.memory.push(
                    state.to("cpu"),
                    action.to("cpu"),
                    reward.to("cpu"),
                    next_state.to("cpu"),
                    done,
                )

            state = next_state
            if self.optimization(got_reward):
                self.optimize_model()

            if self.steps_done % params.TARGET_UPDATE == 0:
                self.target.load_state_dict(self.policy.state_dict())

            if done:
                self.buffer.successes += info["lives"] > 0
                break
            if jump_dead_step:
                for i_dead in range(params.DEAD_STEPS):
                    obs, reward, done, info = self.env.step(0)
                jump_dead_step = False
            yield


        if self.episodes % params.SAVE_MODEL == 0 and self.save:
            save_model(self.paths.path_models, self.policy.state_dict(), "policy", self.episodes)
            save_model(self.paths.path_models, self.target.state_dict(), "target", self.episodes)
            save_plot(self.paths.path_plots, self.buffer)
            buffer.save(self.paths.path_data)

        self.buffer.update()
        yield


if __name__ == "__main__":
    # Get paths
    paths = start(args)
    # Set environment
    ale = ALEInterface()
    ale.loadROM(Pacman)
    env = gym.make("MsPacman-v0")

    # Set Deep Q Networks and memory
    policy = DQN(params.N_ACTIONS).to(device)
    target = DQN(params.N_ACTIONS).to(device)
    memory = ReplayMemory(params.REPLAY_MEMORY_SIZE, params.BATCH_SIZE)

    # Set buffer where data for post processing is stored
    buffer = Buffer()
    datahandler = DataHandler(env, policy, target, memory, buffer, paths, save=not args.stream)
    if args.stream:
        from quart import Quart, render_template, websocket
        app = Quart(__name__)

        @app.route("/")
        async def hello():
            return await render_template("index.html")

        @app.websocket("/ws")
        async def ws():
            for _ in datahandler.run():
                await websocket.send_json(buffer.json())
        app.run(port=5000)
    else:
        generator = datahandler.run()
        for _ in generator:
            continue
