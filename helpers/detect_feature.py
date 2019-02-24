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
        )  # 2

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

        keypoints, descrs = self.detector.detectAndCompute(image_to_process_copy, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs

    def track(self, target_filepath, draw_points=False):

        targetImage = None
        if type(target_filepath) is "string":
            targetImage = cv2.imread(target_filepath)
        else:
            targetImage = target_filepath

        target = {"image": targetImage}

        keypoints, descrs = self.detect_feature(target["image"])

        matches = self.matcher.knnMatch(descrs, k=2)

        matches = [
            m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.7
        ]

        if len(matches) < self.MIN_MATCH_COUNT:
            matches = []
        else:
            print("matches", len(matches))
            pts = self.extract_matches_points(matches, keypoints)

            p0, p1, M = self.check_homography(pts)

            target["feature_keypoints"] = p0
            target["target_keypoints"] = p1
            target["homographic_matrixes"] = M
        return target, len(matches) > 0

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
        
        img_out = self.warpPerspective(target)
        for i in xrange(len(self.features)):

            h = self.features[i].image.shape[0]
            w = self.features[i].image.shape[1]

            quad = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])

            if "homographic_matrixes" in target:

                (translationx, translationy), _, (
                    scalex,
                    scaley,
                ), _ = self.getComponents(target["homographic_matrixes"][i])

                quad2 = [translationx, translationy] + [scalex, scaley] * quad

                cv2.polylines(
                    target["image"], [np.int32(quad2)], True, (255, 255, 255), 2
                )

                quad = cv2.perspectiveTransform(quad, target["homographic_matrixes"][i])

                cv2.polylines(
                    target["image"], [np.int32(quad)], True, (255, 255, 255), 1
                )
            if draw_points and "target_keypoints" in target:
                for (x, y) in target["target_keypoints"][i]:
                    cv2.circle(target["image"], (x, y), 2, (255, 255, 0), 2)

            return img_out

    def show_image(self, targetImage, img_out=None):

        cv2.imshow("targetImage", targetImage)
        if img_out is not None:
            cv2.imshow("img_out", img_out)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    def warpPerspective(self, target):
        if ("homographic_matrixes" in target):
            return cv2.warpPerspective(
                target["image"],
                np.linalg.inv(target["homographic_matrixes"][0]),
                (self.features[0].image.shape[1], self.features[0].image.shape[0]),
            )
        else:
            return None

    def check_homography(self, pts):

        feature_keypoints = []
        target_keypoints = []
        homographic_matrixes = []
        for (p0, p1) in pts:
            M, status = cv2.findHomography(p0, p1, cv2.LMEDS, 5.0)
            status = status.ravel() != 0
            print("p0", len(p0[status]))
            print("p1", len(p1[status]))
            feature_keypoints.append(p0[status])
            target_keypoints.append(p1[status])
            homographic_matrixes.append(M)
            print("M", M)
        return feature_keypoints, target_keypoints, homographic_matrixes

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
