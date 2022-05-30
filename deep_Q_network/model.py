from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
import torch

from .parameters import BATCH_SIZE, DISCOUNT_RATE, TAU, device


class DQN(nn.Module):

    CONV_N_MAPS = [12, 64, 64]
    CONV_KERNEL_SIZES = [(4, 4), (2, 2)]
    CONV_STRIDES = [2, 2]
    CONV_PADDINGS = [2, 0]
    N_HIDDEN_IN = 64 * 11 * 10
    N_HIDDEN = 512

    def __init__(self, outputs):
        super(DQN, self).__init__()
        conv2d = lambda i: nn.Conv2d(
            self.CONV_N_MAPS[i],
            self.CONV_N_MAPS[i + 1],
            kernel_size=self.CONV_KERNEL_SIZES[i],
            stride=self.CONV_STRIDES[i],
            padding=self.CONV_PADDINGS[i],
        )
        self.conv1 = conv2d(0)
        self.conv2 = conv2d(1)

        self.hidden = nn.Linear(self.N_HIDDEN_IN, self.N_HIDDEN)
        self.output = nn.Linear(self.N_HIDDEN, outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.hidden(x))
        return self.output(x)


def optimize_model(policy_DQN, target_DQN, memory, optimizer, display, learn_counter, device):
    if len(memory) < BATCH_SIZE:
        return learn_counter
    learn_counter += 1
    states, actions, rewards, next_states, dones = memory.sample()

    predicted_targets = policy_DQN(states).gather(1, actions)

    target_values = target_DQN(next_states).max(1)[0].detach()
    labels = rewards + DISCOUNT_RATE * (1 - dones.squeeze(1)) * target_values

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(predicted_targets, labels.unsqueeze(1)).to(device)
    display.data.losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    for param in policy_DQN.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # # Softmax update
    # for target_param, local_param in zip(target_DQN.parameters(), policy_DQN.parameters()):
    #     target_param.data.copy_(TAU * local_param.data + (1 - TAU) * target_param.data)

    return learn_counter
