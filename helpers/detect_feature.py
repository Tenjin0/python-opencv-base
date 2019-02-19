import sys
import cv2
import math
# import numpy as np

# built-in modules
from collections import namedtuple

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


Feature = namedtuple('Feature', 'name, image, keypoints, descrs')


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

        matches = self.matcher.knnMatch(descrs, k=2)

        matches = [m[0] for m in matches if len(
            m) == 2 and m[0].distance < m[1].distance * 0.75]

        if len(matches) < self.MIN_MATCH_COUNT:
            matches = []

        pts = []

        for m in matches:
            print(m.imgIdx, m.trainIdx, m.queryIdx)
            if (m.imgIdx == 0):
                pts.append(keypoints[m.queryIdx].pt)

    '''((translationx, translationy), rotation, (scalex, scaley), shear)'''
    def getComponents(self, normalised_homography):
        a = normalised_homography[0,0]
        b = normalised_homography[0,1]
        c = normalised_homography[0,2]
        d = normalised_homography[1,0]
        e = normalised_homography[1,1]
        f = normalised_homography[1,2]

        p = math.sqrt(a*a + b*b)
        r = (a*e - b*d)/(p)
        q = (a*d+b*e)/(a*e - b*d)

        translation = (c,f)
        scale = (p,r)
        shear = q
        theta = math.atan2(b,a)

        # ss = M[0, 1]
        # sc = M[0, 0]
        # scaleRecovered = math.sqrt(ss * ss + sc * sc)
        # thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
        # print("MAP: Calculated scale difference: %.2f, "
        #     "Calculated rotation difference: %.2f" %
        #     (scaleRecovered, thetaRecovered))
        return (translation, theta, scale, shear)

