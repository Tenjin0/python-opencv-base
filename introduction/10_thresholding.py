import cv2
import os

th = 127
max_val = 255
# for color do not forget to convert BGR to RBG
original = cv2.imread(os.path.join(
    os.getcwd(), "images", "gray21.512.tiff"))

ret, o1 = cv2.threshold(original, th, max_val, cv2.THRESH_BINARY)  # 0 or max_value
ret, o2 = cv2.threshold(original, th, max_val, cv2.THRESH_BINARY_INV)
ret, o3 = cv2.threshold(original, th, max_val, cv2.THRESH_TOZERO)   # keep as it is none concern pixel
ret, o4 = cv2.threshold(original, th, max_val, cv2.THRESH_TOZERO_INV)
ret, o5 = cv2.threshold(original, th, max_val, cv2.THRESH_TRUNC)  # all pixel > threshhold => threshold

cv2.imshow("original", original)
cv2.imshow("original1", o1)
# cv2.imshow("original2", o2)
# cv2.imshow("original3", o3)
# cv2.imshow("original4", o4)
# cv2.imshow("original5", o5)
# cv2.imshow("original6", o6)
cv2.waitKey()
cv2.destroyAllWindows()
