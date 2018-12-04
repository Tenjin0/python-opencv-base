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


def convertTuple(a_tuple):
    return tuple(int(i) for i in a_tuple)


if __name__ == "__main__":

    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    cap = cv2.VideoCapture(video_src)

    targetImage = cv2.imread('images/elephant.png', 0)
    imageToTest = cv2.imread('data/s2/20181112-102950-3.jpg', 0)
    # bug : need to pass empty dict (#1329)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    # matcher = cv2.BFMatcher()
    targets = []
    frame_points = []

    detector = cv2.ORB_create(nfeatures=1000)

    test_mask = np.ones(targetImage.shape, np.uint8)

    kpsTarget, descTarget = detector.detectAndCompute(targetImage, None)
    if descTarget is None:  # detectAndCompute returns descs=None if not keypoints found
        descTarget = []
    else:
        descTarget = np.uint8(descTarget)
    matcher.add([descTarget])

    kpsToTest, descToTest = detector.detectAndCompute(imageToTest, None)

    matches = matcher.knnMatch(descToTest,  k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    good.sort(key=lambda x: x[0].distance)
    # print(matches)
    # print(good)
    img3 = cv2.drawMatchesKnn(targetImage, kpsTarget,
                              imageToTest, kpsToTest, good, None, flags=2)
    plt.imshow(img3), plt.show()
    # while True:
    #     ret, frame = self.cap.read()
    #     if not ret:
    #         break
    #     frame = frame.copy()

    #     keypoints, descrs = self.detector.detectAndCompute(frame, None)
    #     if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
    #         descrs = []

    #     if len(keypoints) < MIN_MATCH_COUNT:
    #         tracked = []
