import cv2


def detect():

    face_cascade_default = cv2.CascadeClassifier(
        './cascades/haarcascade_frontalface_default.xml')
    # face_cascade_profile = cv2.CascadeClassifier(
    #     './cascades/haarcascade_profileface.xml')
    face_cascade_alt1 = cv2.CascadeClassifier(
        './cascades/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier(
        './cascades/haarcascade_eye.xml')

    smile_cascade = cv2.CascadeClassifier(
        './cascades/haarcascade_smile.xml')
    camera = cv2.VideoCapture(0)

    while(True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade_alt1.detectMultiScale(gray, 1.3, 3)
        if len(faces) > 0:
            print "alt1"
        if len(faces) == 0:
            faces = face_cascade_default.detectMultiScale(gray, 1.3, 4)
            if len(faces) > 0:
                print "default"
        # elif len(faces) == 0:
        #     faces = face_cascade_profile.detectMultiScale(gray, 1.3, 1)
        #     if len(faces) > 0:
        #         print "profile"

        for (x, y, w, h) in faces:
            # print x, y, w, h
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_gray = gray[y:y+h, x:x+w]
            if len(face_gray) > 0:

                eyes = eye_cascade.detectMultiScale(face_gray, 1.3, 8)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(img, (x + ex, y + ey),
                                  (x + ex+ew, y + ey+eh), (0, 255, 0), 2)
                smiles = smile_cascade.detectMultiScale(face_gray, 1.3, 8)
                for (ex, ey, ew, eh) in smiles:
                    cv2.rectangle(img, (x + ex, y + ey),
                                  (x + ex+ew, y + ey+eh), (0, 0, 255), 2)

        cv2.imshow("camera", frame)
        if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
            break
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    detect()
