import cv2
import numpy as np
import os

imgpath = os.path.join(os.getcwd(),
                       "images", "lines.jpg")
original = cv2.imread(imgpath)

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 120, apertureSize=3)

minLineLength = 20
maxLineGap = 5

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLineLength, maxLineGap)

for line in lines:
    cv2.line(original, (line[0][0], line[0][1]),
             (line[0][2], line[0][3]), (0, 255, 0), 2)

cv2.imshow("edges", edges)
cv2.imshow("lines", original)
cv2.waitKey()
cv2.destroyAllWindows()
