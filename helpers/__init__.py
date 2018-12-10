import os
import sys
import numpy as np
import cv2
from datetime import datetime

DIRECTORY = "data"
PREFIXFOLDER = "s"


def read_images(path, sz=None):

    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):

        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if (filename == ".direction"):
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                    if (sz is not None):
                        im = cv2.resize(im, (sz, sz))

                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(0)

                except IOError as err:
                    errno, strerror = e.args
                    print("I/O error({0}): {1}".format(errno, strerror))
                except:
                    print("Unexpected error:", sys.exc_info()[0])
            c = c + 1

    return [X, y]


def createDir(id, directory=None):

    a_directory = getPathFolder(id, directory)
    if not os.path.exists(a_directory):
        os.makedirs(a_directory)
    return a_directory


def searchNextId(directory=None):

    path_to_search_in = directory if directory is not None else DIRECTORY
    next_id = 0
    for dirname, dirnames, filenames in os.walk(path_to_search_in):
        for subdirname in dirnames:
            if subdirname[0:1] is PREFIXFOLDER:
                folder_id = getIdFromFolderName(subdirname)
                next_id = folder_id if folder_id > next_id else next_id
    return next_id + 1


def getIdFromFolderName(folder):

    return int(folder[len(PREFIXFOLDER):])


def getPathFolder(id, directory=None):

    global DIRECTORY
    global PREFIXFOLDER
    return (directory if directory is not None else DIRECTORY) + "/" + PREFIXFOLDER + str(id)


def generatePathFolder(id=None, directory=DIRECTORY):

    next_id = None
    if id is None:
        next_id = searchNextId(directory)
    else:
        next_id = id
    return createDir(next_id, directory)


def generate(id=None, count=10, fileFormat="pgm"):
    print(id, count, fileFormat)
    storeFolder = generatePathFolder(id=id)
    face_detected = False
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        './cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        './cascades/haarcascade_eye.xml')

    while(True):
        if count == 0:
            break

        wait = int(1000 / 100)
        ret, frame = camera.read()

        copy = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 3)
        face_detected = False
        fs = []

        for (x, y, w, h) in faces:

            face_gray = gray[y:y+h, x:x+w]

            if len(face_gray) > 0:
                eyes = eye_cascade.detectMultiScale(face_gray, 1.3, 8)
                if len(eyes) >= 2:
                    face_detected = True
                    img = cv2.rectangle(
                        copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    fs.append(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))

                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(img, (x + ex, y + ey),
                                      (x + ex+ew, y + ey+eh), (0, 255, 0), 2)
        cv2.imshow("camera", copy)

        if (face_detected):
            wait = 0
        key = cv2.waitKey(wait) & 0xff

        if key == ord("q") or count == 0:
            break
            camera.release()
            cv2.destroyAllWindows()
        elif key == ord("s"):
            for f in fs:
                filename = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(count) + "." + fileFormat
                f = cv2.imwrite(storeFolder + "/%s" % filename, f)
                count -= 1
        elif key == ord("f"):
            filename = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(count) + "." + fileFormat
            f = cv2.imwrite(storeFolder + "/%s" % filename, frame)
            count -= 1

    def detect_corners(image):
        pass
