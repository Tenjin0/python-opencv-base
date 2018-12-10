
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2
FLANN_INDEX_KDTREE = 0
index_params2 = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

# queryImage = cv2.imread('images/manowar_single.jpg', cv2.IMREAD_GRAYSCALE)
# trainingImage = cv2.imread('images/manowar_logo.png', cv2.IMREAD_GRAYSCALE)
# queryImage = cv2.imread('images/bathory_album.jpg', 0)
# trainingImage = cv2.imread('images/bathory_vinyls.jpg', 0)
queryImage = cv2.imread('images/elephant.jpg', 0)
trainingImage = cv2.imread('data/s3/20181210-100736-2.jpg', 0)

orb = cv2.ORB_create()

if __name__ == "__main__":

    if queryImage is None:
        print("queryImage not found")
        sys.exit()

    if trainingImage is None:
        print("trainingImage not found")
        sys.exit()

    kp1, des1 = orb.detectAndCompute(queryImage, None)
    kp2, des2 = orb.detectAndCompute(trainingImage, None)

    # FLANN matcher parameters
    flann = cv2.FlannBasedMatcher(index_params, {})
    matches = flann.knnMatch(des1, des2, k=2)
    # prepare an empty mask to draw good matches
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, match in enumerate(matches):
        if len(match) > 1 and match[0].distance < 0.7 * match[1].distance:
            matchesMask[i] = [1, 0]

    drawParams = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matchesMask,
        flags=0
    )

    resultImage = cv2.drawMatchesKnn(
        queryImage, kp1, trainingImage, kp2, matches, None, **drawParams)

    plt.imshow(resultImage), plt.show()
