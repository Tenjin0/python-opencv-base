import cv2
import numpy as numpy


img = numpy.zeros((3, 3), dtype=numpy.uint8)
shape = img.shape
print img
img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
shape = img.shape
print img2

cv2.imshow("display", img)
cv2.imshow("display2", img2)

while True:
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
