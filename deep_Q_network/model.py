from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
import torch

from .parameters import BATCH_SIZE, DISCOUNT_RATE, device


def my_rely(x):
    return torch.maximum(x, torch.zeros_like(x))


class DQN(nn.Module):

    CONV_N_MAPS = [4, 32, 32]
    CONV_KERNEL_SIZES = [(4, 4), (2, 2)]
    CONV_STRIDES = [2, 2]
    CONV_PADDINGS = [2, 0]
    N_HIDDEN_IN = 32 * 11 * 10
    N_HIDDEN = [512, 128]

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
        self.bn1 = nn.BatchNorm2d(self.CONV_N_MAPS[1])
        self.conv2 = conv2d(1)
        self.bn2 = nn.BatchNorm2d(self.CONV_N_MAPS[2])

        self.hidden1 = nn.Linear(self.N_HIDDEN_IN, self.N_HIDDEN[0])
        self.hidden2 = nn.Linear(self.N_HIDDEN[0], self.N_HIDDEN[1])
        self.output = nn.Linear(self.N_HIDDEN[1], outputs)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.output(x)
