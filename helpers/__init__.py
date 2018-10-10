import os
import sys
import numpy as np
import cv2

DIRECTORY = "data"
PREFIXFOLDER = "s"


def read_images(path, sz=None):

    c = 0
    X, y = [], []

    for dirname, dirnames, filenames in os.walk(path):

        for subdirname in dirnames:
            print subdirname
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

                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
            c = c + 1

    return [X, y]


def createDir(directory, id):

    directory = directory + "/" + PREFIXFOLDER + id
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


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


def getPathFolder(id=None):

    if id is not None:
        return createDir(DIRECTORY, id)
    else:
        next_id = searchNextId(DIRECTORY)
        return DIRECTORY + "/" + PREFIXFOLDER + next_id


def generate(storeFolder):

    face_detected = False
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        './cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        './cascades/haarcascade_eye.xml')

    count = 10

    while(True):
        wait = 1000 / 100
        ret, frame = camera.read()
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
                        frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    fs.append(cv2.resize(gray[y:y+h, x:x+w], (200, 200)))

                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(img, (x + ex, y + ey),
                                      (x + ex+ew, y + ey+eh), (0, 255, 0), 2)
        cv2.imshow("camera", frame)

        if (face_detected):
            wait = 0
        key = cv2.waitKey(wait) & 0xff

        if key == ord("q") or count == 0:
            break
            camera.release()
            cv2.destroyAllWindows()
        elif key == ord("s"):
            for f in fs:
                f = cv2.imwrite(storeFolder + "/%s.pgm" % str(count), f)
                count -= 1
