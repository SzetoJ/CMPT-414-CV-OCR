import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot


def extract_boxes(input_image):
    image_negative = (255 - input_image)
    ret, thresh = cv2.threshold(image_negative, 127, 255, cv2.THRESH_BINARY)
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rec = im[y:y+h, x:x+w]
        output_boxes.append((x, rec))

    output_boxes.sort(key=lambda x: x[0])
    output_boxes = [x[1] for x in output_boxes]

    return output_boxes


def scale_image(input_image):
    scaled_image = cv2.resize(input_image, (96, 96), cv2.INTER_AREA)
    padded_image = cv2.copyMakeBorder(scaled_image, top=16, bottom=16, left=16, right=16, borderType= cv2.BORDER_CONSTANT, value=0 )
    output_image = cv2.resize(padded_image, (28, 28), cv2.INTER_AREA)
    ret, thresh = cv2.threshold(output_image, 40, 255, cv2.THRESH_BINARY)
    return thresh


if __name__ == "__main__":
    input_image = cv2.imread("../data/input/ABC.jpg", 0)
    boxes = extract_boxes(input_image)
    for i in boxes:
        pyplot.imshow(i)
        pyplot.show()
