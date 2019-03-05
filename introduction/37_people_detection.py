import cv2
import numpy as np
import os


def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)


if __name__ == "__main__":

    img_path = os.path.join(os.getcwd(), "images/people.jpg")
    img = cv2.imread(img_path)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    found, w = hog.detectMultiScale(img)
    print("people found:", len(found), len(w))
    # found_filtered = []

    for ri, r in enumerate(found):
        draw_person(img, r)

    cv2.imgshow()


# https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
# https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/