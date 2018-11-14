import sys
import cv2
import numpy as np

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10

if __name__ == "__main__":

    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    cap = cv2.VideoCapture(video_src)

    targetImage = cv2.imread('images/elephant.jpg', 0)
    matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    imageToTest = cv2.imread('data/s2/20181112-102950-3.jpg', 0)
    targets = []
    frame_points = []

    detector = cv2.ORB_create( nfeatures = 1000 )

    kpTarget, descTarget = detector.detectAndCompute(targetImage, None)
    if descTarget is None:  # detectAndCompute returns descs=None if not keypoints found
        descTarget = []
    else:
        descTarget = np.uint8(descTarget)
    matcher.add([descTarget])


    kpToTest, descToTest = detector.detectAndCompute(imageToTest, None)

    matches = matcher.knnMatch(descToTest, k = 2)
    matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
    
    for m in matches:
        print(m.imgIdx, m.trainIdx, m.queryIdx, len(m), m.distance)
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