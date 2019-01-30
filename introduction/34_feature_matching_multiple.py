import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH = 6
flann_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2

MIN_MATCH_COUNT = 10


def tuple_float_to_int(a_tuple):
    return tuple(int(i) for i in a_tuple)


def addImageToMatcher(trainingImage, matcher, KPs):

    trainingImage = cv2.imread(trainingImage)
    trainingCopy = cv2.cvtColor(trainingImage, cv2.COLOR_BGR2GRAY)
    trainingKPs, trainingDescs = detector.detectAndCompute(trainingCopy, None)

    KPs.append(trainingKPs)

    if trainingDescs is None:
        trainingDescs = []

    matcher.add([trainingDescs])


if __name__ == "__main__":

    # try:
    #     video_src = sys.argv[1]
    # except:
    #     video_src = 0
    # cap = cv2.VideoCapture(video_src)

    detector = cv2.ORB_create(nfeatures=1000)

    matcher = cv2.FlannBasedMatcher(flann_params, {})
    KPs = []
    addImageToMatcher('images/elephant_old.png', matcher, KPs)
    addImageToMatcher('images/lipton.jpg', matcher, KPs)

    targetImage = cv2.imread('data/s3/20181210-100718-3.jpg')
    targetCopy = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)

    targetKPs, targetDescs = detector.detectAndCompute(targetCopy, None)

    if targetDescs is None:
        targetDescs = []

    matches = matcher.knnMatch(targetDescs, k=2)

    matches = [m[0] for m in matches if len(
        m) == 2 and m[0].distance < m[1].distance * 0.75]

    if len(matches) < MIN_MATCH_COUNT:
        matches = []

    p0 = []
    p1 = []

    for m in matches:
        print(m.imgIdx, m.trainIdx, m.queryIdx)
        if (m.imgIdx == 0):
            p0.append(KPs[m.imgIdx][m.trainIdx].pt)
            p1.append(targetKPs[m.queryIdx].pt)

    p0, p1 = np.float32((p0, p1))

    # for (x, y) in np.int32(p0):
    #     cv2.circle(trainingImage, (x, y), 10, (0, 0, 0))

    for (x, y) in np.int32(p1):
        cv2.circle(targetImage, (x, y), 10, (0, 255, 255))

    H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    status = status.ravel() != 0

    p0, p1 = p0[status], p1[status]

    # for (x, y) in np.int32(p0):
    #     cv2.circle(trainingImage, (x, y), 8, (255, 255, 0))
    for (x, y) in np.int32(p1):
        cv2.circle(targetImage, (x, y), 8, (255, 255, 0))

    # width, heigth = trainingCopy.shape

    # x0 = 0
    # y0 = 0
    # x1 = width
    # y1 = heigth

    # quad = np.float32([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
    # quad = quad.reshape(-1, 1, 2)
    # quad = cv2.perspectiveTransform(quad, H)
    # cv2.polylines(targetImage, [np.int32(quad)],
    #               True, (255, 255, 255), 2)
    for (x, y) in np.int32(p1):
        cv2.circle(targetImage, (x, y), 2, (255, 255, 255), 2)

    # cv2.imshow('train', trainingImage)
    cv2.imshow('queryImage', targetImage)
    cv2.waitKey()
    cv2.destroyAllWindows()
