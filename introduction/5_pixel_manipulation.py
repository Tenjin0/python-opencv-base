import cv2
import numpy as np 

img = cv2.imread("images/peppers_color.tif")
img2 = cv2.imread("images/peppers_color.tif")
img3 = cv2.imread("images/peppers_color.tif")
img[:, :, 1] = 0
img2[:, :, 0] = 0
img3[:, :, 2] = 0

cv2.imshow("displayWithOutGreen", img)
cv2.imshow("displayWithOutBlue", img2)
cv2.imshow("displayWithOutRed", img3)

cv2.waitKey()
cv2.destroyAllWindows()