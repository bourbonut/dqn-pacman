import random, math, torch
from .parameters import EPS_MAX, EPS_MIN, EPS_DECAY, N_ACTIONS, device
from utils.utils import ACTIONS, REVERSED


class DecisionMaker:
    def __init__(self, steps_done):
        self.steps_done = steps_done
        self.old_action = 3

    def select_action(self, state, policy_DQN, buffer, learn_counter):
        sample = random.random()
        eps_threshold = max(
            EPS_MIN,
            EPS_MAX - (EPS_MAX - EPS_MIN) * learn_counter / EPS_DECAY
        )
        self.steps_done += 1
        with torch.no_grad():
            q_values = policy_DQN(state)
        buffer.qvalues.append(q_values.max(1)[0].item())
        if sample > eps_threshold:
            # Optimal action
            return q_values.max(1)[1].view(1, 1)
        else:
            # Random action
            action = random.randrange(N_ACTIONS)
            while action == REVERSED[self.old_action]:
                action = random.randrange(N_ACTIONS)
            return torch.tensor([[action]], device=device, dtype=torch.long)
