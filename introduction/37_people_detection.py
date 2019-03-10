import cv2
import numpy as np
import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
needed_dir = os.path.abspath(os.path.join(this_dir, '../.'))
sys.path.insert(0, needed_dir)

from helpers.detect_people import Detect_people

# https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
# https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/
# https://github.com/alexander-hamme/Pedestrian_Detector


detector = Detect_people()
currentDirectory = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(currentDirectory, "..", "images", "1_original.jpg"))

detector.DRAW_RAW_RECT = True
detector.DRAW_RECT = True
detector.SHOW_IMAGES = True
detector.set_calibration(2)
image, _, _ = detector.find_people(img)
detector.draw_image(image)

# detector.try_all_calibration_modes(img)
