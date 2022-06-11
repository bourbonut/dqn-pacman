from math import log

try:
    import cv2
    from .start import PATH_VIDEO
except Exception as e:
    # print(e)
    pass

REWARDS = {
    "default": -0.2,
    200: 20,
    50: 15,
    10: 10,
    0: 0,
    "lose": -25,
    "win": 10,
    "reverse": -2,
}
REVERSED = {0: 1, 1: 0, 2: 3, 3: 2}
isreversed = (
    lambda last_action, action: "default" if REVERSED[action] - last_action else "reverse"
)

ACTIONS = {
    1: [1, 4, 6, 5],
    2: [5, 7, 3, 2],
    3: [6, 8, 3, 2],
    4: [1, 4, 8, 7],
    5: [1, 4, 3, 2],
    6: [1, 4, 3, 2],
    7: [1, 4, 3, 2],
    8: [1, 4, 3, 2],
}

# if best_score < sum(display.rewards):
# best_score = sum(display.rewards)
def save_run(one_game):
    frameSize = (160, 210)
    bin_loader = cv2.VideoWriter_fourcc(*"DIVX")  # Binary extension loader
    out = cv2.VideoWriter(str(PATH_VIDEO / "video.avi"), bin_loader, 15, frameSize)
    for img in one_game:
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out.release()


def transform_reward(reward):
    return log(reward, 10) if reward > 0 else reward

    # r = REWARDS["default"]  # isreversed(last_action, action_)
    # r += REWARDS[reward] if reward in REWARDS else reward / 10
