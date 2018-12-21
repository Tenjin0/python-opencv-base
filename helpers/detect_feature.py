import sys
import cv2
import numpy as np


PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


Target = namedtuple('Target', 'name, image, keypoints, descrs')


class detect_feature():

    def __init__():

        self.FLANN_INDEX_KDTREE = 1
        self.FLANN_INDEX_LSH = 6
        self.flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                 table_number=6,  # 12
                                 key_size=12,     # 20
                                 multi_probe_level=1)  # 2

        self.MIN_MATCH_COUNT = 10

        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.targets = []

    def add_feature_from_path(self, name, filepath):

        trainingImage = cv2.imread('images/lipton5.png')
        self.add_feature(name, trainingImage)

    def add_feature(self, name, feature_image):
        feature_copy = cv2.cvtColor(trainingImage, cv2.COLOR_BGR2GRAY)
        keypoints, descrs = self.detect_feature(feature_copy)
        target = Target(name=name, image=feature_image,
                        keypoints=keypoints, descrs=descrs)
        self.targets.append(target)
        self.matcher.add([descrs])

    def detect_features(self, image_to_process):

        keypoints, descrs = self.detector.detectAndCompute(
            image_to_process, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs
