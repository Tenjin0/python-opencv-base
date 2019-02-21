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
        _, frame = cameraCapture.read()
        app.track('data/s3/20181210-100711-4.jpg')

        app.show_image()

        if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
            break  # esc to quit

    cap.release()
    cv2.destroyAllWindows()
