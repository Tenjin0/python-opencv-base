import cv2
import os

imgpath = os.path.join(os.getcwd(),
                       "images",  "4.2.06.tiff")

img = cv2.imread(imgpath, 0)

cv2.imwrite("canny.jpg", cv2.Canny(img, 200, 300))
cv2.imshow('original', img)
cv2.imshow("canny", cv2.imread("canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()
