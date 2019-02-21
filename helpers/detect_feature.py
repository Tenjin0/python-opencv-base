import sys
import cv2
import math
import numpy as np

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
        self.features = []
        self.targetImage = None

    def add_feature_from_path(self, name, filepath):

        trainingImage = cv2.imread(filepath)
        self.add_feature(name, trainingImage)

    def add_feature(self, name, feature_image):

        keypoints, descrs = self.detect_feature(feature_image)

        feature = Feature(name=name, image=feature_image,
                          keypoints=keypoints, descrs=descrs)
        self.features.append(feature)
        self.matcher.add([descrs])

    def detect_feature(self, image_to_process):

        image_to_process_copy = cv2.cvtColor(
            image_to_process, cv2.COLOR_BGR2GRAY) if image_to_process.shape[2] > 1 else image_to_process

        keypoints, descrs = self.detector.detectAndCompute(
            image_to_process_copy, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs

    def track(self, target_filepath):

        self.targetImage = cv2.imread(target_filepath)
        keypoints, descrs = self.detect_feature(self.targetImage)

        matches = self.matcher.knnMatch(descrs, k=2)

        matches = [m[0] for m in matches if len(
            m) == 2 and m[0].distance < m[1].distance * 0.75]

        if len(matches) < self.MIN_MATCH_COUNT:
            matches = []

        pts = self.extract_matches_points(matches, keypoints)

        pts_with_m = self.check_homography(pts)

        self.draw_on_target_image([M for (p0, p1, M) in pts_with_m])

    def extract_matches_points(self, matches, targetKPs):

        ptsTraining = [[] for _ in xrange(len(self.features))]
        ptsTarget = [[] for _ in xrange(len(self.features))]

        for m in matches:
            ptsTraining[m.imgIdx].append(
                self.features[m.imgIdx].keypoints[m.trainIdx].pt)
            ptsTarget[m.imgIdx].append(targetKPs[m.queryIdx].pt)

        pts = []

        for i in xrange(len(self.features)):
            pts.append(np.float32((ptsTraining[i], ptsTarget[i])))

        return pts

    def draw_on_target_image(self, homograph_matrixes):

        for i in xrange(len(self.features)):

            h = self.features[i].image.shape[0]
            w = self.features[i].image.shape[1]

            quad = np.float32(
                [[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])

            (translationx, translationy), rotation, (scalex, scaley), shear = self.getComponents(homograph_matrixes[i])

            quad2 = [translationx, translationy] + [scalex, scaley] * quad
            cv2.polylines(self.targetImage, [np.int32(quad2)],
                          True, (255, 255, 255), 2)

            quad = cv2.perspectiveTransform(quad, homograph_matrixes[i])

            cv2.polylines(self.targetImage, [np.int32(quad)],
                          True, (255, 255, 255), 1)

    def show_image(self):

        cv2.imshow('targetImage', self.targetImage)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    def warpPerspective(self):

        return cv2.warpPerspective(self.targetImage,
                                   np.linalg.inv(M),
                                   (self.features[0].image.shape[1],
                                    self.targefeaturests[0].image.shape[0]))

    def check_homography(self, pts):

        new_pts = []
        for (p0, p1) in pts:
            M, status = cv2.findHomography(p0, p1, cv2.LMEDS, 5.0)
            status = status.ravel() != 0

            new_pts.append((p0[status], p1[status], M))
        return new_pts

    '''((translationx, translationy), rotation, (scalex, scaley), shear)'''

    def getComponents(self, normalised_homography):

        a = normalised_homography[0, 0]
        b = normalised_homography[0, 1]
        c = normalised_homography[0, 2]
        d = normalised_homography[1, 0]
        e = normalised_homography[1, 1]
        f = normalised_homography[1, 2]

        p = math.sqrt(a*a + b*b)
        r = (a*e - b*d)/(p)
        q = (a*d+b*e)/(a*e - b*d)

        translation = (c, f)
        scale = (p, r)
        shear = q
        theta = math.atan2(b, a)

        # ss = M[0, 1]
        # sc = M[0, 0]
        # scaleRecovered = math.sqrt(ss * ss + sc * sc)
        # thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
        # print("MAP: Calculated scale difference: %.2f, "
        #     "Calculated rotation difference: %.2f" %
        #     (scaleRecovered, thetaRecovered))
        return (translation, theta, scale, shear)
