import cv2
import numpy as numpy


img = numpy.zeros((3, 3), dtype=numpy.uint8)
shape = img.shape
print shape, len(shape)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
shape = img.shape
print shape, len(shape)

cv2.imshow("display", img)

while True:
    k = cv2.waitKey()
    if cv2.waitKey(113):
        break

cv2.destroyAllWindows()
