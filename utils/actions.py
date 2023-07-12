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
