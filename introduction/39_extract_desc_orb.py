import cv2
import os
from os.path import join, normpath

currentDirectory = os.path.dirname(os.path.abspath(__file__))

datapath = normpath(join(currentDirectory, "..", "training", "cars_light"))

img1 = cv2.imread(join(datapath, "neg-0.pgm"), cv2.IMREAD_GRAYSCALE)

kaze = cv2.KAZE_create()
kpt1, desc1 = kaze.detectAndCompute(img1, None)

print(desc1)
