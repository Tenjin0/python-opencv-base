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

if __name__ == "__main__":

    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    cap = cv2.VideoCapture(video_src)

    targetImage = cv2.imread('images/elephant.jpg', 0)
    imageToTest = cv2.imread('data/s2/20181112-102950-3.jpg', 0)
    # bug : need to pass empty dict (#1329)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    targets = []
    frame_points = []

    detector = cv2.ORB_create(nfeatures=1000)

    kpsTarget, descTarget = detector.detectAndCompute(targetImage, None)
    if descTarget is None:  # detectAndCompute returns descs=None if not keypoints found
        descTarget = []
    else:
        descTarget = np.uint8(descTarget)
    matcher.add([descTarget])

    kpsToTest, descToTest = detector.detectAndCompute(imageToTest, None)

    matches = matcher.knnMatch(descToTest, k=2)

    matches = [m[0] for m in matches if len(
        m) >= 2 and m[0].distance < m[1].distance * 0.75]

    # for m in matches:
    #     print(m.imgIdx, m.trainIdx, m.queryIdx, m.distance)
    #     print(kpsTarget[m.trainIdx].pt)

    # print(matches)
    filteredMatch = matches[20]
    # targetKPID20 = filteredMatch[0].trainIdx
    # testKPID20 = filteredMatch[0].queryIdx

    # targetKPID21 = filteredMatch[0].trainIdx
    # testKPID21 = filteredMatch[0].queryIdx
    # print(filteredMatch, filteredMatch.trainIdx, filteredMatch.queryIdx)
    # print(kpsTarget[filteredMatch.imgIdx].pt,
    #       kpsToTest[filteredMatch.imgIdx].pt)
    # print(filteredMatch)
    # print(matches[20].imgIdx,
    #       matches[20].trainIdx,  matches[20].queryIdx, matches[20].distance,
    #       kpsTarget[targetKPID20].pt, kpsToTest[testKPID20].pt
    #       )
    # print(matches[21].imgIdx,
    #       matches[21].trainIdx,  matches[21].queryIdx, matches[21].distance,
    #       kpsTarget[targetKPID21].pt, kpsToTest[testKPID21].pt
    #       )
    # drawParams = dict(
    #     matchColor=(0, 255, 0),
    #     singlePointColor=(255, 0, 0),
    #     matchesMask=matchesMask,
    #     flags=0
    # )
    # img3 = cv2.drawMatches(
    #     targetImage, kpsTarget,
    #     imageToTest, kpsToTest,
    #     filteredMatch,
    #     None, flags=2)
    print(targetImage.shape, kpsTarget[filteredMatch.imgIdx].pt)
    print(imageToTest.shape, kpsToTest[filteredMatch.queryIdx].pt)
    img3 = cv2.drawMatches(
        targetImage, kpsTarget, imageToTest, kpsToTest, [filteredMatch], None)
    # img3 = cv2.drawMatchesKnn(
    #     targetImage, kpsTarget, imageToTest, kpsToTest, matches, None)
    cv2.circle(img3, kpsTarget[filteredMatch.imgIdx].pt, 63, (0, 0, 255), -1)

    plt.imshow(img3)
    plt.show()
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
