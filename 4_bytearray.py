import cv2
import numpy
import os

randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)  # numpy.random.randint(0,	256, 120000)
grayImage = flatNumpyArray.reshape(300, 400)
bgrImage = flatNumpyArray.reshape(100, 400, 3)

cv2.imshow("gray", grayImage)
cv2.imshow("color", bgrImage)
cv2.waitKey()
cv2.destroyAllWindows()
