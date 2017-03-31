import cv2

#URL: http://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv

image = cv2.imread('this-is-a-meme.jpg')


hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



show_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#show_image[:,:,2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255) for pixel in row] for row in show_image[:,:,2]]
show_image[:,:,2] = [[(0) if pixel < 190 else (255) for pixel in row] for row in show_image[:,:,2]]


cv2.imshow('original image', image)
#cv2.imshow('hsv image', image)

contrast = cv2.cvtColor(show_image, cv2.COLOR_HSV2BGR)
cv2.imshow('contrast', contrast)


gray_scale = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

gray1[:] = [[(0) if pixel < 240 else (255) for pixel in row] for row in gray_scale[:]]
gray2[:] = [[(0) if pixel < 255 else (255) for pixel in row] for row in gray_scale[:]]
cv2.imshow('240 cutoff from grayscale', gray1)
cv2.imshow('255 cutoff from grayscale', gray2)


cv2.waitKey(0)
cv2.destroyAllWindows()

