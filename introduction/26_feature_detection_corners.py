
import sys
import os
import cv2
import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
needed_dir = os.path.abspath(os.path.join(this_dir, '../.'))
sys.path.insert(0, needed_dir)

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print "USAGE: [file].py [file_image].png"
        exit()

    img = cv2.imread("images/" + sys.argv[1])
    if img is None:
        print "no image found images/{}".format(sys.argv[1])
        exit()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 23, 0.04)
    print dst
    img[dst > 0.02 * dst.max()] = [0, 0, 255]

    while(True):
        cv2.imshow("corners", img)
        if cv2.waitKey(1000/12) & 0xff == ord("q"):
            break

    cv2.destroyAllWindows()
