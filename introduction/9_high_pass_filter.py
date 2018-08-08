import cv2
import numpy as np
from scipy import ndimage
import os

# http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php
kernel_3x3 = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1,	-1,	-1]])

original = cv2.imread(os.path.join(
    os.getcwd(), "images", "lena_color_512.tif"))

# converting to gray scale
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# remove noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

laplacian = cv2.Laplacian(blurred, -1, ksize=3, scale=1, delta=0,
                          borderType=cv2.BORDER_DEFAULT)

sobelx = cv2.Sobel(blurred, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3, scale=1,
                   delta=0, borderType=cv2.BORDER_DEFAULT)
sobely = cv2.Sobel(blurred, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3, scale=1,
                   delta=0, borderType=cv2.BORDER_DEFAULT)

scharrx = cv2.Scharr(blurred, ddepth=cv2.CV_64F, dx=0, dy=1, scale=1,
                     delta=0, borderType=cv2.BORDER_DEFAULT)
scharry = cv2.Scharr(blurred, ddepth=cv2.CV_64F, dx=1, dy=0, scale=1,
                     delta=0, borderType=cv2.BORDER_DEFAULT)

sobel = sobelx + sobely
scharr = scharrx + scharry

k3 = ndimage.convolve(blurred, kernel_3x3)
blurred = cv2.GaussianBlur(blurred, (11, 11), 0)
g_hpf = gray - blurred
cv2.imshow("original", original)
cv2.imshow("blurred", blurred)
cv2.imshow("3x3", k3)
cv2.imshow("g_hpf", g_hpf)
cv2.imshow("laplacian", laplacian)
cv2.imshow("sobel", sobel)
cv2.imshow("scharr", scharr)
cv2.waitKey()
cv2.destroyAllWindows()
