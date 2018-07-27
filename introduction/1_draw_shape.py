import cv2
import numpy as np


def main():
    img1 = np.zeros((512, 512, 3), np.uint8)
    cv2.line(img1, (0, 99), (99, 0), (255, 0, 0), 2)
    cv2.rectangle(img1, (100, 60), (80, 70), (0, 255, 0), 2)
    cv2.circle(img1, (60, 60), 50, (0, 255, 0), 2)
    cv2.imshow("Lena", img1)
    cv2.waitKey(0)
    cv2.destroyWindow("Lena")


if __name__ == "__main__":
    main()
