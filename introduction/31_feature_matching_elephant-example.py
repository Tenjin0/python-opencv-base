import sys
import cv2

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

    queryImage = cv2.imread('images/elephant.jpg')
    matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    imageToTest = cv2.imread('data/s2/20181112-102950-3.jpg')
    while True:
        ret, frame = self.cap.read()
        if not ret:
            break
        frame = frame.copy()

        keypoints, descrs = self.detector.detectAndCompute(frame, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []

        if len(keypoints) < MIN_MATCH_COUNT:
            tracked = []