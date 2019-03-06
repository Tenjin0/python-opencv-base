import cv2
import numpy as np
import os


def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)


if __name__ == "__main__":

    img_path = os.path.join(os.getcwd(), "images/people.jpg")
    img = cv2.imread(img_path)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    found, w = hog.detectMultiScale(
        img, winStride=(4, 4), padding=(16, 16), scale=1.06)

    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
        else:
            found_filtered.append(r)

    for person in found_filtered:
        draw_person(img, person)

    cv2.imshow("people detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
# https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/
# https://github.com/alexander-hamme/Pedestrian_Detector
