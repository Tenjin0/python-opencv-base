import cv2
import os
import numpy as np

# imgpath = os.path.join(os.getcwd(),
#                        "images", "4.2.06.tiff")

# img = cv2.imread(imgpath, 0)

original = np.zeros((200, 200), dtype=np.uint8)
original[50:150, 50:150] = 255
original[90:110, 90:110] = 128

ret, thresh = cv2.threshold(original, 127, 255, 0)

image, contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

imgContour = cv2.drawContours(gray, contours, -1, (0, 255, 0), 1)

cv2.imshow('original', original)
cv2.imshow('thresh', thresh)
cv2.imshow('gray', gray)
cv2.imshow('imgContour', imgContour)
cv2.waitKey()
cv2.destroyAllWindows()
