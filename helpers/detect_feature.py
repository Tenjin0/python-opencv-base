import sys
import cv2
import numpy as np

# built-in modules
from collections import namedtuple

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


Target = namedtuple('Target', 'name, image, keypoints, descrs')


class Detect_feature():

    def __init__(self):

        self.FLANN_INDEX_KDTREE = 1
        self.FLANN_INDEX_LSH = 6
        self.flann_params = dict(algorithm=self.FLANN_INDEX_LSH,
                                 table_number=6,  # 12
                                 key_size=12,     # 20
                                 multi_probe_level=1)  # 2

        self.MIN_MATCH_COUNT = 10

        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.FlannBasedMatcher(self.flann_params, {})
        self.targets = []
        self.targetImage = None

    def add_feature_from_path(self, name, filepath):

        trainingImage = cv2.imread(filepath)
        self.add_feature(name, trainingImage)

    def add_feature(self, name, feature_image):

        feature_copy = cv2.cvtColor(feature_image, cv2.COLOR_BGR2GRAY)
        keypoints, descrs = self.detect_features(feature_copy)

        feature = Feature(name=name, image=feature_image,
                          keypoints=keypoints, descrs=descrs)
        self.targets.append(feature)
        self.matcher.add([descrs])

    def detect_features(self, image_to_process):

        keypoints, descrs = self.detector.detectAndCompute(
            image_to_process, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs

    def track(self, target_filepath):

        self.targetImage = cv2.imread(target_filepath)
        targetCopy = cv2.cvtColor(self.targetImage, cv2.COLOR_BGR2GRAY)
        keypoints, descrs = self.detect_features(targetCopy)

        matches = app.matcher.knnMatch(targetDescs, k=2)

        matches = [m[0] for m in matches if len(
            m) == 2 and m[0].distance < m[1].distance * 0.75]

        if len(matches) < MIN_MATCH_COUNT:
            matches = []

        p0 = []
        p1 = [] 
