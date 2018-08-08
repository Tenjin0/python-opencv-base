import cv2
import os
import numpy as np
imgpath = os.path.join(os.getcwd(),
                       "images", "4.2.06.tiff")

img = cv2.imread(imgpath, 1)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img3 = abs(255 - img)
img4 = abs(255 - img2)
cv2.imshow('original', img)
cv2.imshow("Gray", img2)
cv2.imshow("IOriginal", img3)
cv2.imshow("IGray", img4)

cv2.waitKey()
cv2.destroyAllWindows()
