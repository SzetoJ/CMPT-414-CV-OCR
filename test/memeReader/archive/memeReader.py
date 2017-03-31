from os import listdir
from os.path import isfile, join
import numpy
import cv2


def contrastHSV(images):
    print "Converting images to HSV..."
    images = [(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)) for image in images]
    print "Contrasting data..."
    for image in images:
        image[:,:,2] = [[(0) if pixel < 200 else (255) for pixel in row] for row in image[:,:,2]]
    print "Converting back to BGR..."
    images = [(cv2.cvtColor(image, cv2.COLOR_HSV2BGR)) for image in images]
    return images

def contrastGRAY(images):
    print "Converting images to GRAY..."
    images = [(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) for image in images]
    print "Contrasting data..."
    for image in images:
        image[:] = [[(0) if pixel < 250 else (255) for pixel in row] for row in image[:]]
    #print "Converting back to BGR..."
    #images = [(cv2.cvtColor(image, cv2.COLOR_HSV2BGR)) for image in images]
    return images

def gaussianBlur(images):
    print "Smoothing images to reduce noise..."
    images = [(cv2.GaussianBlur(image,(5,5),0)) for image in images]
    return images


#main
print "Determining image locations..."
imageDir = './images/'
imageList = [ f for f in listdir(imageDir) if isfile(join(imageDir,f)) ]

outputDir ='./output/'
outputList = [ join(outputDir,f) for f in listdir(imageDir) if isfile(join(imageDir,f)) ]

imageNames = [ f for f in listdir(imageDir) ]
images = numpy.empty(len(imageList), dtype=object)

print "Reading all images from " + imageDir + "..."
for n in range(0, len(imageList)):
	images[n] = cv2.imread( join(imageDir, imageList[n]) )

#print imageNames[0]
#cv2.imshow(imageNames[0], images[0])
images = contrastHSV(images)
images = contrastGRAY(images)
images = gaussianBlur(images)

#x,y,w,h = cv2.boundingRect(cnt)
#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


print "Saving changes into " + outputDir + "..."
for i in range(0, len(images)):
	cv2.imwrite(outputList[i], images[i])


print "Done"
#cv2.waitKey(0)
#cv2.destroyAllWindows()










'''
0	'this-is-a-meme.jpg'
1	'meme-baby.png',
2	'meme-willy-wonka.jpg',
3	'you-mean-to-tell-me-269x300.jpg',
4	'pet-rock-meme_430-512.jpeg',
5	'ikea_689-442.jpeg',
6	'crazy_girl.jpeg',
7	'test_impossible_418-418.jpeg',
8	'misunderstood-kim-jong-un-meme-300x239.jpg',
9	'matrix-300x269.jpg',
10	'Drunk-Baby2.jpg',
11	'austin.jpg'

imageNames = [
	'this-is-a-meme.jpg'
	'meme-baby.png',
	'meme-willy-wonka.jpg',
	'you-mean-to-tell-me-269x300.jpg',
	'pet-rock-meme_430-512.jpeg',
	'ikea_689-442.jpeg',
	'crazy_girl.jpeg',
	'test_impossible_418-418.jpeg',
	'misunderstood-kim-jong-un-meme-300x239.jpg',
	'matrix-300x269.jpg',
	'Drunk-Baby2.jpg',
	'austin.jpg'
]
'''
