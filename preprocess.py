import os
import cv2
import numpy as np

def cutter(src):

    cap = cv2.VideoCapture(src)

    frames = []
    if not cap.isOpened():
        cap.open(src)
    ret = True
    while(True and ret):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()

    croped_frames = []
    for cf in frames:
        rsz_f = cf[300:950, 0:1920]  # cut unnecessary sky and car view
        croped_frames.append(rsz_f)

    return np.asarray(croped_frames)
