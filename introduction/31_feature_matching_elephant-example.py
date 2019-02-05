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


if __name__ == "__main__":

    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    cap = cv2.VideoCapture(video_src)

    detector = cv2.ORB_create(nfeatures=1000)

    matcher = cv2.FlannBasedMatcher(flann_params, {})

    trainingImage = cv2.imread('images/elephant_old.png')
    trainingImage = cv2.imread('images/elephant.png')
    trainingCopy = cv2.cvtColor(trainingImage, cv2.COLOR_BGR2GRAY)

    trainingKPs, trainingDescs = detector.detectAndCompute(trainingCopy, None)

    if trainingDescs is None:
        trainingDescs = []

    matcher.add([trainingDescs])

    # targetImage = cv2.imread('data/s3/20181210-100718-3.jpg')
    targetImage = cv2.imread('data/s3/20181210-100711-4.jpg')
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
        p0.append(trainingKPs[m.trainIdx].pt)
        p1.append(targetKPs[m.queryIdx].pt)

    p0, p1 = np.float32((p0, p1))

    for (x, y) in np.int32(p0):
        cv2.circle(trainingImage, (x, y), 10, (0, 0, 0))

    for (x, y) in np.int32(p1):
        cv2.circle(targetImage, (x, y), 10, (0, 255, 255))

    H, status = cv2.findHomography(p0, p1, cv2.LMEDS, 5.0)
    status = status.ravel() != 0

    p0, p1 = p0[status], p1[status]

    for (x, y) in np.int32(p0):
        cv2.circle(trainingImage, (x, y), 8, (255, 255, 0))
    for (x, y) in np.int32(p1):
        cv2.circle(targetImage, (x, y), 8, (255, 255, 0))

    h = trainingCopy.shape[0]
    w = trainingCopy.shape[1]
    trainBorder = np.float32(
        [[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
    queryBorder = cv2.perspectiveTransform(trainBorder, H)
    cv2.polylines(targetImage, [np.int32(queryBorder)], True, (255, 255, 255), 2)
    # quad = quad.reshape(-1, 1, 2)
    # quad = cv2.perspectiveTransform(quad, H)
    # cv2.polylines(targetImage, [np.int32(quad)],
    #               True, (255, 255, 255), 2)
    for (x, y) in np.int32(p1):
        cv2.circle(targetImage, (x, y), 2, (255, 255, 255), 2)

    cv2.imshow('train', trainingImage)
    cv2.imshow('queryImage', targetImage)
    cv2.waitKey()
    cv2.destroyAllWindows()
