import cv2
import numpy as np
import os

imgpath = os.path.join(os.getcwd(),
                       "images", "hammer.jpg")

original = cv2.pyrDown(cv2.imread(imgpath, cv2.IMREAD_UNCHANGED))

# original = np.zeros((200, 200), dtype=np.uint8)
# original[50:150, 50:150] = 255
# original[90:110, 90:110] = 128

ret, thresh = cv2.threshold(cv2.cvtColor(
    original.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

image, contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for c in contours:
    # find bounding box coordinates
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # find minimum area
    rect = cv2.minAreaRect(c)
    # calculate coordinates of the minimum area reactangle
    box = cv2.boxPoints(rect)
    # normalize coordinates to integers
    box = np.int16(box)
    # draw contours
    cv2.drawContours(original, [box], 0, (0, 0, 255), 3)

    # calculate center and radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    # cast to integers
    center = (int(x), int(y))
    radius = int(radius)
    # draw the circle
    img = cv2.circle(original, center, radius, (0, 255, 0), 2)

cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

cv2.imshow('original', original)
cv2.imshow('contours', img)
cv2.waitKey()
cv2.destroyAllWindows()