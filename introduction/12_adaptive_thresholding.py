import cv2
import numpy as np
from scipy import ndimage
import os

# https://docs.opencv.org/3.4.1/d7/d4d/tutorial_py_thresholding.html
th = 0
max_val = 255
# for color do not forget to convert BGR to RBG
original = cv2.imread(os.path.join(
    os.getcwd(), "images", "4.1.05.tiff"), 0)


block_size = 513
constant = 2

th1 = cv2.adaptiveThreshold(
    original, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)
th2 = cv2.adaptiveThreshold(
    original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)

cv2.imshow("original", original)
cv2.imshow("th_mean", th1)
cv2.imshow("th_gauss", th2)
cv2.waitKey()
cv2.destroyAllWindows()