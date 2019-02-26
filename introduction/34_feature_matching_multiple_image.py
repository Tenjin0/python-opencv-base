import sys
import cv2
# import matplotlib.pyplot as plt
import os


this_dir = os.path.dirname(os.path.abspath(__file__))
needed_dir = os.path.abspath(os.path.join(this_dir, "../."))
sys.path.insert(0, needed_dir)

from helpers.detect_feature import Detect_feature

if __name__ == "__main__":

    app = Detect_feature()

    app.add_feature_from_path("elephant", "images/elephant.png")
    app.add_feature_from_path("lipton", "images/lipton.jpg")

    targetImage = cv2.imread("data/s3/20181210-100711-4.jpg")

    target, detected = app.track(targetImage, draw_points=True)

    app.draw_on_target_image(target, draw_points=True)
    app.show_image(target)
    cv2.waitKey()
    cv2.destroyAllWindows()
