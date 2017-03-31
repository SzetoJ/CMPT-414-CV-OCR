import cv2

#read an image
image = cv2.imread('this-is-a-meme.jpg')

#convert to gray-scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#write/save to harddisk
cv2.imwrite('gray_image.jpg', gray_image)

cv2.imshow('color_image', image)
cv2.imshow('gray_image', gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


