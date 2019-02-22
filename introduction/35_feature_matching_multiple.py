import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

this_dir = os.path.dirname(os.path.abspath(__file__))
needed_dir = os.path.abspath(os.path.join(this_dir, '../.'))
sys.path.insert(0, needed_dir)

from helpers.detect_feature import Detect_feature

if __name__ == "__main__":

    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    cap = cv2.VideoCapture(video_src)

    app = Detect_feature()

    app.add_feature_from_path("elephant", "images/elephant.png")
    # app.add_feature_from_path("lipton", "images/lipton.jpg")

    # targetImage = cv2.imread('data/s3/20181210-100711-4.jpg')
    # targetCopy = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)

    # targetKPs, targetDescs = app.detector.detectAndCompute(targetCopy, None)

    # if targetDescs is None:
    #     targetDescs = []

    while cap.isOpened():
        _, frame = cap.read()
        target, detected = app.track(frame, draw_points=True)
        if (detected):
            app.warpPerspective(target)

        img_out = app.draw_on_target_image(target, draw_points=True)
        print(img_out)
        app.show_image(target['image'], img_out)
        wait = 0 if detected else 1
        key = cv2.waitKey(wait)

        if key & 0xff == ord("q"):
            break  # esc to quit

    cap.release()
    cv2.destroyAllWindows()
