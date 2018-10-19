
import cv2
import sys
import os
import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
needed_dir = os.path.abspath(os.path.join(this_dir, '../.'))
sys.path.insert(0, needed_dir)

# https://www.superdatascience.com/opencv-face-recognition/
from helpers import read_images

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "USAGE: facerec_demo.py </path/to/images> "
        sys.exit()

    [X, y] = read_images(sys.argv[1])
    y = np.asarray(y, dtype=np.int32)

    if len(sys.argv) == 3:
        out_dir = sys.argv[2]

    model1 = cv2.face.EigenFaceRecognizer_create()
    # model2 = cv2.face.FisherFaceRecognizer_create()
    model3 = cv2.face.LBPHFaceRecognizer_create()

    model1.train(np.asarray(X), np.asarray(y))
    # model2.train(np.asarray(X), np.asarray(y))
    model3.train(np.asarray(X), np.asarray(y))

    camera = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        "./cascades/haarcascade_frontalface_default.xml")

    while(True):
        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 3)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x+w, y:y+h]
            try:
                roi = cv2.resize(roi, (200, 200),
                                 interpolation=cv2.INTER_LINEAR)
                params1 = model1.predict(roi)
                # params2 = model2.predict(roi)
                params3 = model3.predict(roi)
                # print params1
                # print params3
                cv2.putText(img, "%s" % params1[0], (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
                cv2.putText(img, "{0:.2f}%".format(params1[1] / 100), (x + w - 50, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

                # cv2.putText(img, "%s" % params2[0], (x, y - 15),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
                # cv2.putText(img, "{0:.2f}%".format(params2[1] / 100), (x + w - 50, y - 15),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)

                cv2.putText(img, "%s" % params3[0], (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)
                cv2.putText(img, "{0:.2f}%".format(params3[1] / 100), (x + w - 50, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)


            except:
                cv2.putText(img, "unknown", (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, 255, 2)
        cv2.imshow("camera", img)

        if cv2.waitKey(1000 / 50) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()
    camera.release()
