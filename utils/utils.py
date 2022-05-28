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
    "lose": -1,
    "win": 5,
    "reverse": -6,
}
REVERSED = {0: 9, 1: 4, 2: 3, 3: 2, 4: 1, 5: 9, 6: 7, 7: 6, 8: 5}
isreversed = (
    lambda last_action, action: "default" if REVERSED[action] - last_action else "reverse"
)

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
    return reward
    # r = REWARDS["default"]  # isreversed(last_action, action_)
    # r += REWARDS[reward] if reward in REWARDS else reward / 10
