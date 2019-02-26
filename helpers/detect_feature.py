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


Feature = namedtuple("Feature", "name, image, keypoints, descrs")


class Detect_feature:
    def __init__(self):

        self.FLANN_INDEX_KDTREE = 1
        self.FLANN_INDEX_LSH = 6
        self.flann_params = dict(
            algorithm=self.FLANN_INDEX_LSH,
            table_number=6,  # 12
            key_size=12,  # 20
            multi_probe_level=1,
        )

        self.MIN_MATCH_COUNT = 10

        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.FlannBasedMatcher(self.flann_params, {})
        self.features = []

    def add_feature_from_path(self, name, filepath):

        trainingImage = cv2.imread(filepath)
        self.add_feature(name, trainingImage)

    def add_feature(self, name, feature_image):

        keypoints, descrs = self.detect_feature(feature_image)

        feature = Feature(
            name=name, image=feature_image, keypoints=keypoints, descrs=descrs
        )
        self.features.append(feature)
        self.matcher.add([descrs])

    def detect_feature(self, image_to_process):

        image_to_process_copy = (
            cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
            if image_to_process.shape[2] > 1
            else image_to_process
        )

        keypoints, descrs = self.detector.detectAndCompute(
            image_to_process_copy, None)

        # detectAndCompute returns descs=None if not keypoints found
        if descrs is None:
            descrs = []
        return keypoints, descrs

    def track(self, target_filepath, draw_points=False):

        targetImage = None
        if type(target_filepath) == "string":
            targetImage = cv2.imread(target_filepath)
        else:
            targetImage = target_filepath

        target = {"image": targetImage, "features": []}

        keypoints, descrs = self.detect_feature(target["image"])

        matches = self.matcher.knnMatch(descrs, k=2)

        matches = [
            m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.7
        ]

        if len(matches) < self.MIN_MATCH_COUNT:
            matches = []
        else:
            pts = self.extract_matches_points(matches, keypoints)

            self.check_homography(pts, target)
            print(len(matches) > 0, len(target["features"]) > 0, len(
                matches) > 0 and len(target["features"]) > 0)
        return target, len(matches) > 0 and len(target["features"]) > 0

    def extract_matches_points(self, matches, targetKPs):

        ptsTraining = [[] for _ in xrange(len(self.features))]
        ptsTarget = [[] for _ in xrange(len(self.features))]

        for m in matches:
            ptsTraining[m.imgIdx].append(
                self.features[m.imgIdx].keypoints[m.trainIdx].pt
            )
            ptsTarget[m.imgIdx].append(targetKPs[m.queryIdx].pt)

        pts = []

        for i in xrange(len(self.features)):
            pts.append(np.float32((ptsTraining[i], ptsTarget[i])))

        return pts

    def draw_on_target_image(self, target, draw_points=False):

        for feature_i in target["features"]:

            self.warpPerspective(target)

            h = self.features[feature_i["id"]].image.shape[0]
            w = self.features[feature_i["id"]].image.shape[1]

            quad = np.float32(
                [[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])

            if "homographic_matrice" in feature_i:

                (translationx, translationy), _, (
                    scalex,
                    scaley,
                ), _ = self.getComponents(feature_i["homographic_matrice"])

                quad2 = [translationx, translationy] + [scalex, scaley] * quad

                cv2.polylines(
                    target["image"], [
                        np.int32(quad2)], True, (255, 255, 255), 2
                )

                quad = cv2.perspectiveTransform(
                    quad, feature_i["homographic_matrice"])

                cv2.polylines(
                    target["image"], [np.int32(quad)], True, (255, 255, 255), 1
                )
            if draw_points and "target_keypoints" in feature_i:
                for (x, y) in feature_i["target_keypoints"]:
                    cv2.circle(target["image"], (x, y), 2, (255, 255, 0), 2)

    def foundFeatureById(self, id):

        return self.features[id]

    def show_image(self, target):

        cv2.imshow("targetImage", target["image"])

        for feature_i in target["features"]:
            cv2.imshow(self.foundFeatureById(
                feature_i["id"]).name, feature_i["warp_perspective"])

    def warpPerspective(self, target):

        for feature_i in target["features"]:
            feature_i['warp_perspective'] = cv2.warpPerspective(
                target["image"],
                np.linalg.inv(feature_i["homographic_matrice"]),
                (self.features[feature_i["id"]].image.shape[1],
                 self.features[feature_i["id"]].image.shape[0]))

    def nice_homography(self, M):

        if M is None:
            return False
        det = M[0][0] * M[1][1] - M[1][0] * M[0][1]

        if (det < 0):
            return False
        N1 = math.sqrt(M[0][0] * M[0][0] + M[1][0] * M[1][0])

        if (N1 > 4 or N1 < 0.1):
            return False
        N2 = math.sqrt(M[0][1] * M[0][1] + M[1][1] * M[1][1])

        if (N2 > 4 or N1 < 0.1):
            return False
        N3 = math.sqrt(M[2][0] * M[2][0] + M[2][1] * M[2][1])

        if (N3 > 0.002):
            return False

        return True

    def check_homography(self, pts, target):

        for i in xrange(len(pts)):
            (p0, p1) = pts[i]
            M = None
            status = None
            try:
                M, status = cv2.findHomography(p0, p1, cv2.LMEDS, 5.0)
            except Exception:
                print("p0", p0)
                print("p1", p1)
                raise
            status = status.ravel() != 0
            is_homography_correct = self.nice_homography(M)
            if (is_homography_correct):
                feature = {
                    'id': i,
                    'feature_keypoints': p0[status],
                    'target_keypoints': p1[status],
                    'homographic_matrice': M
                }
                target["features"].append(feature)

    """((translationx, translationy), rotation, (scalex, scaley), shear)"""

    def getComponents(self, normalised_homography):

        a = normalised_homography[0, 0]
        b = normalised_homography[0, 1]
        c = normalised_homography[0, 2]
        d = normalised_homography[1, 0]
        e = normalised_homography[1, 1]
        f = normalised_homography[1, 2]

        p = math.sqrt(a * a + b * b)
        r = (a * e - b * d) / (p)
        q = (a * d + b * e) / (a * e - b * d)

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
