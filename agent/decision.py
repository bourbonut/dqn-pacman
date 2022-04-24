import random, math, torch
from .parameters import EPS_MAX, EPS_MIN, EPS_DECAY, N_ACTIONS, device


class DecisionMaker:
    def __init__(self, steps_done, policy_DQN):
        self.steps_done = steps_done

    def select_action(self, state, policy_DQN, display, learn_counter):
        sample = random.random()
        eps_threshold = max(EPS_MIN, EPS_MAX - (EPS_MAX - EPS_MIN) * learn_counter / EPS_DECAY)
        self.steps_done += 1
        with torch.no_grad():
            q_values = policy_DQN(state)
        display.add_q_value(q_values.max(1)[0].item())
        if sample > eps_threshold:
            # Optimal action
            return q_values.max(1)[1].view(1, 1)
        else:
            # Random action
            return torch.tensor(
                [[random.randrange(N_ACTIONS)]], device=device, dtype=torch.long
            )
