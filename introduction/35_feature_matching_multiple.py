import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

this_dir = os.path.dirname(os.path.abspath(__file__))
needed_dir = os.path.abspath(os.path.join(this_dir, '../.'))
sys.path.insert(0, needed_dir)

from helpers.detect_feature import Detect_feature

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

    app = Detect_feature()

    app.add_feature_from_path("elephant", "images/elephant.png")
    # app.add_feature_from_path("lipton", "images/lipton.jpg")

    targetImage = cv2.imread('data/s3/20181210-100711-4.jpg')
    targetCopy = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)

    targetKPs, targetDescs = app.detector.detectAndCompute(targetCopy, None)

    if targetDescs is None:
        targetDescs = []

    matches = app.matcher.knnMatch(targetDescs, k=2)

    matches = [m[0] for m in matches if len(
        m) == 2 and m[0].distance < m[1].distance * 0.75]

    if len(matches) < MIN_MATCH_COUNT:
        matches = []

    ptsTraining = [[] for _ in xrange(len(app.targets))]
    ptsTarget = [[] for _ in xrange(len(app.targets))]

    for m in matches:
        ptsTraining[m.imgIdx].append(
            app.targets[m.imgIdx].keypoints[m.trainIdx].pt)
        ptsTarget[m.imgIdx].append(targetKPs[m.queryIdx].pt)

    p0, p1 = np.float32((ptsTraining[0], ptsTarget[0]))
    print(len(p1))

    # for (x, y) in np.int32(ptsTarget[0]):
    #     cv2.circle(targetImage, (x, y), 10, (0, 255, 255))

    # for (x, y) in np.int32(p1):
    M, status = cv2.findHomography(p0, p1, cv2.LMEDS, 5.0)
    status = status.ravel() != 0

    p0, p1 = p0[status], p1[status]
    print(len(p1))
    # for (x, y) in np.int32(p1):
    #     cv2.circle(targetImage, (x, y), 2, (255, 255, 255), 2)
    # for (x, y) in np.int32(ptsTraining[0]):
    #     cv2.circle(app.targets[0].image, (x, y), 8, (255, 255, 0))
    # for (x, y) in np.int32(p1):
    #     cv2.circle(targetImage, (x, y), 8, (255, 255, 0))

    h = app.targets[0].image.shape[0]
    w = app.targets[0].image.shape[1]

    quad = np.float32(
        [[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])

    ss = M[0, 1]
    sc = M[0, 0]
    scaleRecovered = math.sqrt(ss * ss + sc * sc)
    thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
    print("MAP: Calculated scale difference: %.2f, "
          "Calculated rotation difference: %.2f" %
          (scaleRecovered, thetaRecovered))

    # deskew image
    im_out = cv2.warpPerspective(targetImage,
                                 np.linalg.inv(M),
                                 (app.targets[0].image.shape[1],
                                  app.targets[0].image.shape[0]))

    quad = cv2.perspectiveTransform(quad, M)

    cv2.polylines(targetImage, [np.int32(quad)],
                  True, (255, 255, 255), 2)

    # for (x, y) in np.int32(p1):
    #     cv2.circle(targetImage, (x, y), 2, (255, 255, 255), 2)

    # cv2.imshow('trainingImage', app.targets[0].image)
    cv2.imshow('im_out', im_out)
    cv2.imshow('queryImage', targetImage)
    cv2.waitKey()
    cv2.destroyAllWindows()
