import cv2
import numpy as np
from scipy import ndimage
import os

kernel_3x3 = np.array([[-1, -1, -1],
                       [-1, 8,	-1],
                       [-1,	-1,	-1]])
kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, 2, 4, 2, -1],
                       [-1, 1, 2, 1, -1],
                       [-1,	-1,	-1,	-1,	-1]])

a = np.array([[1, 2, 0, 0],
              [5, 3, 0, 4],
              [0, 0, 0, 7],
              [9, 3, 0, 0]])

k = np.array([[1, 1, 1],
              [1, 1, 0],
              [1, 0, 0]])


ka = ndimage.convolve(a, k)

img = cv2.imread(os.path.join(os.getcwd(), "images", "peppers_color.tif"),
                 cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(os.getcwd(), "images", "peppers_color.tif"))

if img is not None:
    k3 = ndimage.convolve(img, kernel_3x3)
    k5 = ndimage.convolve(img, kernel_5x5)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    g_hpf = img - blurred
    cv2.imshow("original", img2)
    cv2.imshow("3x3", k3)
    cv2.imshow("5x5", k5)
    cv2.imshow("blurred", blurred)
    cv2.imshow("g_hpf", g_hpf)
    cv2.waitKey()
    cv2.destroyAllWindows()
