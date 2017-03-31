from os import listdir
from os.path import isfile, join
import numpy
import cv2


print "Starting box extraction..."
#imageDir and outputDir
imageDir = "./output/"
print "Input directory " + imageDir + "..."
imageList = [ f for f in listdir(imageDir) if isfile(join(imageDir,f)) ]
outputDir ='./boxes/'
print "Output directory " + outputDir + "..."
outputList = [ join(outputDir,f) for f in listdir(imageDir) if isfile(join(imageDir,f)) ]
imageNames = [ f for f in listdir(imageDir) ]
images = numpy.empty(len(imageList), dtype=object)

print "Reading all images from " + imageDir + "for extraction";
for n in range(0, len(imageList)):
	images[n] = cv2.imread( join(imageDir, imageList[n]), 0)
	ret,thresh = cv2.threshold(images[n],127,255,0)
	#im is an image buffer to help determine coordinates/bounding boxes of "boxes"
	im,contours,hierarchy = cv2.findContours(thresh, 1, 2)
	images[n] = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
	print ".";
        idx =0 
        for cnt in contours:
            idx += 1
            x,y,w,h = cv2.boundingRect(cnt)
            #
            rec=im[y:y+h,x:x+w]
            cv2.imwrite(outputDir + imageNames[n].split(".")[0] + str(idx) + '.jpg', rec)
            #Debugger: drawing green boxes to input images
            cv2.rectangle(images[n],(x,y),(x+w,y+h),(0,200,0),2)
        #saving image with box locations indicated by green boxes
        cv2.imwrite(outputList[n], images[n])

print "Done"


'''
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
'''
