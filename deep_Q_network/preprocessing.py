from .parameters import PACMAN_COLOR
import numpy as np
import torch
from torchvision import transforms as T

# from PIL import Image

resize = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(160, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ]
)


# def preprocessing_observation(obs):
#     truncated_obs = obs[1:176:2, ::2]  # crop and truncate the observation
#     gray_truncated_obs = truncated_obs.sum(axis=2)  # to grayscale
#     indices = gray_truncated_obs == PACMAN_COLOR
#     gray_truncated_obs[gray_truncated_obs == PACMAN_COLOR] = 0
#     normalized_img = (gray_truncated_obs // 3 - 128).astype(np.int8)
#     reshaped_array = normalized_img.reshape(88, 80, 1).transpose((2, 0, 1)).astype(np.int8)
#     screen = torch.from_numpy((reshaped_array / 128).astype(np.float32))
#     return screen.unsqueeze(0)


def unit_prepr_obs(obs):
    cropped_obs = obs[1:176:2, ::2]
    gray_truncated_obs = cropped_obs.sum(axis=2)  # to grayscale
    indices = gray_truncated_obs == PACMAN_COLOR
    gray_truncated_obs[gray_truncated_obs == PACMAN_COLOR] = 0
    normalized_img = (gray_truncated_obs // 3 - 128).astype(np.int8)
    return (normalized_img / 128).astype(np.float32)


# def unit_prepr_obs(obs):
#     gray_truncated_obs = obs.sum(axis=2)  # to grayscale
#     indices = gray_truncated_obs == PACMAN_COLOR
#     gray_truncated_obs[gray_truncated_obs == PACMAN_COLOR] = 0
#     grayscale_img = gray_truncated_obs // 3
#     screen = np.ascontiguousarray(grayscale_img, dtype=np.float32) / 255
#     return resize(screen).view(210, 160)


def preprocessing_observation(observations, new_obs):
    for i in range(3):
        observations[3 - i, :, :] = observations[2 - i, :, :]
    observations[0, :, :] = unit_prepr_obs(new_obs)
    screen = torch.from_numpy(observations)
    return screen.unsqueeze(0)


def init_obs(env):
    return np.stack([unit_prepr_obs(env.step(0)[0]) for i_step in range(4)][::-1])
