import cv2
import numpy as np
from scipy import ndimage
import os

# https://docs.opencv.org/3.4.1/d7/d4d/tutorial_py_thresholding.html
th = 0
max_val = 255
# for color do not forget to convert BGR to RBG
original = cv2.imread(os.path.join(
    os.getcwd(), "images", "7.1.01.tiff"), 0)
ret, o1 = cv2.threshold(original, th, max_val,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 0 or max_value
ret, o2 = cv2.threshold(original, th, max_val,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# keep as it is none concern pixel
ret, o3 = cv2.threshold(original, th, max_val,
                        cv2.THRESH_TOZERO | cv2.THRESH_OTSU)
ret, o4 = cv2.threshold(original, th, max_val,
                        cv2.THRESH_TOZERO_INV | cv2.THRESH_OTSU)
# all pixel > threshhold => threshold
ret, o5 = cv2.threshold(original, th, max_val,
                        cv2.THRESH_TRUNC | cv2.THRESH_OTSU)

cv2.imshow("original", original)
cv2.imshow("binary", o1)
cv2.imshow("binary_inv", o2)
cv2.imshow("tozero", o3)
cv2.imshow("tozero_inv", o4)
cv2.imshow("trunc", o5)

cv2.waitKey()
cv2.destroyAllWindows()
