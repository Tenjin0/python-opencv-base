
import cv2
import os
import sys
import numpy as np
from datetime import datetime


def createDir():

    directory = "data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


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


def read_images(path, sz=None):

    c = 0
    X, y = [], []

    for dirname, dirnames, filenames in os.walk(path):

        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            print subject_path
            for filename in os.listdir(subject_path):
                try:
                    if (filename == ".direction"):
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                    if (sz is not None):
                        im = cv2.resize(im, (sz, sz))

                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)

                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
            c = c + 1

    return [X, y]


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "USAGE: facerec_demo.py </path/to/images> "
        sys.exit()

    [X, y] = read_images(sys.argv[1])
    y = np.asarray(y, dtype=np.int32)

    if len(sys.argv) == 3:
        out_dir = sys.argv[2]

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))
    camera = cv2.VideoCapture(0)
    camera.release()

    face_cascade = cv2.CascadeClassifier(
        "./cascades/haarcascade_frontalface_default.xml")

    while(True):
        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x+w, y:y+h]
            # try:
            #     roi = cv2.resize(roi, (200, 200),
            #                      interpolation=cv2.INTER_LINEAR)
            #     params = model.predict(roi)
            #     print "Label: %s, Confidence: %.2f" % (params[0], params[1])
            #     cv2.putText(img, "toto", (x, y - 20),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            # except:
            #     continue
            cv2.imshow("camera", img)

            if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
                break
    cv2.destroyAllWindows()
    camera.close()
