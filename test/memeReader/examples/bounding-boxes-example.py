import cv2

#im = cv2.imread('ikea_689-442.jpeg', 0)
#im = cv2.imread('matrix-300x269.jpg', 0)
#im = cv2.imread('pet-rock-meme_430-512.jpeg', 0)
#im = cv2.imread('crazy_girl.jpeg', 0)
im = cv2.imread('test_impossible_418-418.jpeg', 0)

#gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#im2, contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

ret,thresh = cv2.threshold(im,127,255,0)
im,contours,hierarchy = cv2.findContours(thresh, 1, 2)

im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

idx =0 
for cnt in contours:
    idx += 1
    x,y,w,h = cv2.boundingRect(cnt)
    roi=im[y:y+h,x:x+w]
    cv2.imwrite('./dummy/' + str(idx) + '.jpg', roi)
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,200,0),2)
cv2.imshow('im',im)
cv2.waitKey(0)