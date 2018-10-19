
import cv2
import sys
import numpy as np


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("USAGE: [file].py [file_image].png")
        exit()
    imgpath = sys.argv[1]

    img = cv2.imread("images/" + sys.argv[1])

    if img is None:
        print("no image found images/{}".format(sys.argv[1]))
        exit()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()

    img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color=(51, 163, 236))

    cv2.imshow('sift_keypoints', img)

    while(True):
        if cv2.waitKey(1000/12) & 0xff == ord("q"):
            break

    cv2.destroyAllWindows()
