import cv2
import os
import numpy as np
imgpath = os.path.join(os.getcwd(),
                       "images", "4.2.06.tiff")

img = cv2.imread(imgpath, 1)
rows, columns, channels = img.shape

T = np.float32([[1, 0, 100], [0, 1, 50]])
R = cv2.getRotationMatrix2D((columns/2, rows/2), 90, 1)
img2 = cv2.warpAffine(img, T, (columns, rows))
img3 = cv2.warpAffine(img, R, (columns, rows))
cv2.imshow('original', img)
cv2.imshow("img2", img2)
cv2.imshow("img3", img3)

cv2.waitKey()
cv2.destroyAllWindows()
