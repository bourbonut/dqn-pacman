# Only on local device
from agent import *
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt

try:
    from rich import print
except:
    print("To get colors, you can install rich with the command `pip install rich-cli`")


# PATH = Path().absolute() / "images"
#
# img = env.render(mode="rgb_array")
# if cv2.imwrite(str(PATH / "color-rgb.png"), img):
#     print("Image `color-rgb.png` written")
#
#
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# if cv2.imwrite(str(PATH / "color-bgr.png"), img):
#     print("Image `color-bgr.png` written")
#
#
# # Preprocessing part
# # General idea
# truncated = img[1:176:2, ::2]
# print(f"img.shape = {img.shape}")
# print(f"truncated.shape = {truncated.shape}")
#
# gray_truncated = truncated.mean(axis=2)
# if cv2.imwrite(str(PATH / "grayscale.png"), gray_truncated):
#     print("Image `grayscale.png` written")
#
# gray_truncated[gray_truncated == PACMAN_COLOR] = 0
#
# normalized_img = (gray_truncated // 3 - 128).astype(np.int8)

# Test part
# prepr_img = preprocessing_observation(img)
# print(prepr_img.dtype)
# a = np.array(prepr_img[0]).transpose((1, 2, 0))
# print(a.reshape(88, 80))
# print((a > 0).any())
# print((a < 0).any())
# print(a.max())
# print(a.min())
# plt.imshow(a, cmap="gray")
# plt.show()
# print(prepr_img.shape)
# print((-1 < prepr_img).all() and (prepr_img < 1).all())
for action, meaning in zip(
    env.unwrapped.get_keys_to_action(), env.unwrapped.get_action_meanings()
):
    print(f"{action}, {meaning}")

# print(help(env))
