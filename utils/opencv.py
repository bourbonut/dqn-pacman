try:
    import cv2
    from .start import PATH_VIDEO
except Exception as e:
    pass

def save_run(one_game):
    frameSize = (160, 210)
    bin_loader = cv2.VideoWriter_fourcc(*"DIVX")  # Binary extension loader
    out = cv2.VideoWriter(str(PATH_VIDEO / "video.avi"), bin_loader, 15, frameSize)
    for img in one_game:
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out.release()
