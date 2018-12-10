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


# def convertTuple(a_tuple):
#     return tuple(int(i) for i in a_tuple)


if __name__ == "__main__":

    # try:
    #     video_src = sys.argv[1]
    # except:
    #     video_src = 0
    # cap = cv2.VideoCapture(video_src)

    trainingImage = cv2.imread('images/elephant.png', 0)
    queryImage = cv2.imread('data/s3/20181210-100711-4.jpg', 0)
    # bug : need to pass empty dict (#1329)

    if queryImage is None:
        raise ValueError('query image not found')

    matcher = cv2.FlannBasedMatcher(flann_params, {})
    querys = []
    frame_points = []

    vis = queryImage.copy()

    detector = cv2.ORB_create(nfeatures=1000)

    # test_mask[0:100] = 0
    # test_mask[200:300] = 0
    # print(test_mask)
    kpsTraining, descTraining = detector.detectAndCompute(
        trainingImage, None)
    if descTraining is None:  # detectAndCompute returns descs=None if not keypoints found
        descQdescTraininguery = []
    else:
        descTraining = np.uint8(descTraining)
    matcher.add([descTraining])
    # test_mask = np.ones(trainingImage.shape, np.uint8)
    # cv2.rectangle(test_mask, (40, 40), (150, 150), 255, cv2.FILLED)
    # cv2.rectangle(trainingImage, (40, 40), (45, 45), 0, 2)
    # img = cv2.drawKeypoints(trainingImage, kpsQuery, img)

    # test_mask[10: 400] = 250
    kpsQuery, descQuery = detector.detectAndCompute(queryImage, None)

    # img = cv2.drawKeypoints(queryImage, kpsTraining, img)

    matches = matcher.knnMatch(descTraining, k=2)

    matches = [m[0] for m in matches if len(
        m) == 2 and m[0].distance < m[1].distance * 0.75]
    # if len(matches) < MIN_MATCH_COUNT:
    #     return []

    p0 = [kpsTraining[m.trainIdx].pt for m in matches]
    p1 = [kpsQuery[m.queryIdx].pt for m in matches]
    p0, p1 = np.float32((p0, p1))

    H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    status = status.ravel() != 0
    if status.sum() >= MIN_MATCH_COUNT:
        p0, p1 = p0[status], p1[status]

    x0, y0, x1, y1 = (0, 0, 210, 210)
    quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    quad = cv2.perspectiveTransform(
        quad.reshape(1, -1, 2), H).reshape(-1, 2)
    quad = np.int32([quad])

    # cv2.polylines(vis, quad,
    #               True, (255, 255, 255), 2)

    for (x, y) in np.int32(p1):
            cv2.circle(vis, (x, y), 2, (255, 255, 255))

    cv2.imshow('plane', vis)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
    # tracked.append(track)
    # # while True:
    #     ret, frame = self.cap.read()
    #     if not ret:
    #         break
    #     frame = frame.copy()

    #     keypoints, descrs = self.detector.detectAndCompute(frame, None)
    #     if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
    #         descrs = []

    #     if len(keypoints) < MIN_MATCH_COUNT:
    #         tracked = []
