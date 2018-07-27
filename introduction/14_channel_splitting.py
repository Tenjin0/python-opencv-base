import cv2
import os
import numpy as np

img = cv2.imread(os.path.join(os.getcwd(),
                              "images", "peppers_color.tif"))

r = img[:, :, 2]
g = img[:, :, 1]
b = img[:, :, 0]
r = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
merge = cv2.merge((b, g, r))

cv2.imshow('original', img)
# cv2.imshow("red", r)
# cv2.imshow("blue", b)
# cv2.imshow("green", g)
cv2.imshow("merge", merge)

cv2.waitKey()
cv2.destroyAllWindows()
