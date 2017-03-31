import cv2
import numpy as np


img = cv2.imread('star1.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv2.moments(cnt)
print M

print " "

print cnt

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()