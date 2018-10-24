
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

img1 = cv2.imread('images/manowar_logo.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('images/manowar_single.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

if __name__ == "__main__":

    if img1 is None:
        print("img1 not found")
        sys.exit()

    if img2 is None:
        print("img2 not found")
        sys.exit()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = bf.knnMatch(des1, des2, k=2)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img2, flags=2)

    plt.imshow(img3), plt.show()
