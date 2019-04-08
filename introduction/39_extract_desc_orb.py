import cv2
import os
from os.path import join, normpath

currentDirectory = os.path.dirname(os.path.abspath(__file__))

datapath = normpath(join(currentDirectory, "..", "training", "cars_light"))

img1 = cv2.imread(join(datapath, "neg-0.pgm"), cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
# kaze = cv2.KAZE_create()
kpt = orb.detect(img1)
# kpt1, desc1 = orb.detectAndCompute(img1, None)
print(kpt)
# print(desc1)
