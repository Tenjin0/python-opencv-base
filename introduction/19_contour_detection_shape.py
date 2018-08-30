import cv2
import numpy as np
import os

imgpath = os.path.join(os.getcwd(),
                       "images", "hammer.jpg")
original = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)

pyrDown = cv2.pyrDown(original)


ret, thresh = cv2.threshold(cv2.cvtColor(
    pyrDown.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

image, contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for c in contours:
    # find bounding box coordinates
    x, y, w, h = cv2.boundingRect(c)
    copy = cv2.rectangle(pyrDown.copy(), (x, y), (x+w, y+h), (0, 255, 0), 2)

    # find minimum area
    rect = cv2.minAreaRect(c)

    # calculate coordinates of the minimum area reactangle
    box = cv2.boxPoints(rect)

    # normalize coordinates to integers
    box = np.int32(box)

    # draw contours
    contours = cv2.drawContours(pyrDown.copy(), [box], 0, (0, 0, 255), 3)

    # calculate center and radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)

    # cast to integers
    center = (int(x), int(y))
    radius = int(radius)
    # draw the circle
    cv2.circle(copy, center, radius, (0, 255, 0), 2)

# cv2.imshow('original', original)
cv2.imshow('pyrDown', copy)
cv2.imshow('contours', contours)
cv2.waitKey()
cv2.destroyAllWindows()
