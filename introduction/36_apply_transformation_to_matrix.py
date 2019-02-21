import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math


if __name__ == "__main__":

    img = np.zeros((512, 512, 3), np.uint8)

    h = 10

    w = 50

    quad = np.float32(
        [[[0, 0], [0, h], [w, h], [w, 0]]])

    rows, cols, grade = quad.shape
    # print(rows, cols)
    # M = np.float32([[1, 0, 100], [0, 1, 50]])
    # quad = cv2.warpAffine(quad, M, (cols, rows))
    # print(quad)
    quad = [1, 10] * quad
    # quad = [100, 50] + quad
    cv2.polylines(img, [np.int32(quad)],
                  True, (255, 255, 255), 2)

    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
