import cv2
import numpy as np
import os

filename = os.path.join(os.getcwd(),
                        "images", "people.jpg")

gray_faces = []


def detect(filename):
    global gray_faces

    face_cascade = cv2.CascadeClassifier(
        './cascades/haarcascade_frontalface_alt2.xml')

    eye_cascade = cv2.CascadeClassifier(
        './cascades/haarcascade_eye.xml')

    smile_cascade = cv2.CascadeClassifier(
        './cascades/haarcascade_smile.xml')

    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print gray

    faces = face_cascade.detectMultiScale(gray, 1.3, 6)
    for (x, y, w, h) in faces:
        # print x, y, w, h
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_gray = gray[y:y+h, x:x+w]
        if len(face_gray) > 0:
            gray_faces.append(face_gray)
            eyes = eye_cascade.detectMultiScale(face_gray,  1.1, 1)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (x + ex, y + ey),
                              (x + ex+ew, y + ey+eh), (0, 255, 0), 2)
            smiles = smile_cascade.detectMultiScale(face_gray, 2.0, 5)
            for (ex, ey, ew, eh) in smiles:
                cv2.rectangle(img, (x + ex, y + ey),
                              (x + ex+ew, y + ey+eh), (0, 0, 255), 2)

    cv2.namedWindow("Face detected")
    cv2.imshow("Face detected", img)
    cv2.waitKey(0)


detect(filename)
