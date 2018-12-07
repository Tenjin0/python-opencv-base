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

    trainingImage = cv2.imread('images/elephant.png')
    trainingCopy = cv2.cvtColor(trainingImage, cv2.COLOR_BGR2GRAY)

    trainingKPs, trainingDescs = detector.detectAndCompute(trainingImage, None)

    if trainingDescs is None:
        trainingDescs = []

    matcher.add([trainingDescs])

    targetImage = cv2.imread('data/s2/20181205-095524-1.jpg')
    targetCopy = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)

    targetKPs, targetDescs = detector.detectAndCompute(trainingImage, None)

    if targetDescs is None:
        targetDescs = []

    matches = matcher.knnMatch(targetDescs, k=2)
    matches = [m[0] for m in matches if len(
        m) == 2 and m[0].distance < m[1].distance * 0.75]

    print(matches)
    
    # if len(matches) < MIN_MATCH_COUNT:
    #     return []
