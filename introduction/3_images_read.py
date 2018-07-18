import cv2
import numpy as numpy
import os

original_image = cv2.imread(os.path.join(os.getcwd(),
                            "images", "peppers_color.tif"))
img2 = cv2.imread("images/peppers_color.tif", cv2.IMREAD_GRAYSCALE)
# img3 = cv2.imread("images/peppers_color.tif", cv2.IMREAD_ANYCOLOR)
# img4 = cv2.imread("images/peppers_color.tif", cv2.IMREAD_ANYDEPTH)
# img5 = cv2.imread("images/peppers_color.tif", cv2.IMREAD_COLOR)
# img6 = cv2.imread("images/peppers_color.tif", cv2.IMREAD_LOAD_GDAL)
# img7 = cv2.imread("images/peppers_color.tif", cv2.IMREAD_UNCHANGED)
cv2.imshow("display", original_image)
cv2.imshow("display2", img2)
# cv2.imshow("display3", img3)
# cv2.imshow("display4", img4)
# cv2.imshow("display5", img5)
# cv2.imshow("display6", img6)
# cv2.imshow("display7", img7)

while True:
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
