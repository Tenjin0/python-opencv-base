import cv2
import numpy as np
from scipy import ndimage
import os

# https://docs.opencv.org/3.4.1/d7/d4d/tutorial_py_thresholding.html
th = 127
max_val = 255
# for color do not forget to convert BGR to RBG


import cv2

cameraCapture = cv2.VideoCapture(0)


fps = 30
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))


while cameraCapture.isOpened():
    _, frame = cameraCapture.read()
    ret, o1 = cv2.threshold(frame, th, max_val, cv2.THRESH_BINARY)  # 0 or max_value
    ret, o2 = cv2.threshold(frame, th, max_val, cv2.THRESH_BINARY_INV)
    ret, o3 = cv2.threshold(frame, th, max_val, cv2.THRESH_TOZERO)   # keep as it is none concern pixel
    ret, o4 = cv2.threshold(frame, th, max_val, cv2.THRESH_TOZERO_INV)
    ret, o5 = cv2.threshold(frame, th, max_val, cv2.THRESH_TRUNC)  # all pixel > threshhold => threshold

    cv2.imshow('MyWindow',	frame)
    cv2.imshow("binary", o1)
    cv2.imshow("binary_inv", o2)
    cv2.imshow("tozero", o3)
    cv2.imshow("tozero_inv", o4)
    cv2.imshow("trunc", o5)

    if cv2.waitKey(1) == 27:

        break  # esc to quit

cameraCapture.release()
cv2.destroyAllWindows()