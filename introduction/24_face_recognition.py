
import cv2


def generate():
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
                f = cv2.imwrite("./data/%s.pgm" % str(count), f)
                count -= 1


if __name__ == "__main__":
    generate()
