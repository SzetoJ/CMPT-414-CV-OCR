import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot


def check_line_filled(line, threshold):
    for cell in line:
        if cell < threshold:
            return True
    return False


def segment_lines_debug(input_image, threshold, line_thickness):
    segmented_indexes = []

    index = 0

    while index < (len(input_image)):
        if check_line_filled(input_image[index], threshold):
            current_line = [index]
            index += 1

            while index < len(input_image) and check_line_filled(input_image[index], threshold):
                index += 1

            current_line.append(index)
            segmented_indexes.append(current_line)

        index += 1

    drawn_image = input_image.copy()

    for j in segmented_indexes:
        cv2.rectangle(drawn_image, (0, j[0]), (len(drawn_image[0]) - 1, j[1]), 0, line_thickness)

    return drawn_image


def segment_characters_debug(input_image, threshold, line_thickness):
    segmented_indexes = []

    index = 0

    while index < (len(input_image[0])):

        if check_line_filled([x[index] for x in input_image], threshold):
            current_line = [index]
            index += 1

            while index < len(input_image[0]) and check_line_filled([x[index] for x in input_image], threshold):
                index += 1

            current_line.append(index)
            segmented_indexes.append(current_line)

        index += 1

    drawn_image = input_image.copy()

    for j in segmented_indexes:
        cv2.rectangle(drawn_image, (j[0], 0), (j[1] - 1, len(drawn_image)), 0, line_thickness)

    return drawn_image


def segment_lines(input_image, threshold):
    segmented_indexes = []

    index = 0

    while index < (len(input_image)):
        if check_line_filled(input_image[index], threshold):
            current_line = [index]
            index += 1

            while index < len(input_image) and check_line_filled(input_image[index], threshold):
                index += 1

            current_line.append(index)
            segmented_indexes.append(current_line)

        index += 1

    segmented_lines = []

    for line in segmented_indexes:
        segmented_lines.append(input_image[line[0]: line[1], 0: len(input_image[0])])

    return segmented_lines


def segment_characters(input_image, threshold):
    segmented_indexes = []

    index = 0

    while index < (len(input_image[0])):

        if check_line_filled([x[index] for x in input_image], threshold):
            current_line = [index]
            index += 1

            while index < len(input_image[0]) and check_line_filled([x[index] for x in input_image], threshold):
                index += 1

            current_line.append(index)
            segmented_indexes.append(current_line)

        index += 1

    segmented_lines = []

    for line in segmented_indexes:
        segmented_lines.append(input_image[0: len(input_image[0]), line[0]: line[1]])

    return segmented_lines


def extract_outer_box(input_image):
    image_negative = (255 - input_image)
    ret, thresh = cv2.threshold(image_negative, 127, 255, cv2.THRESH_BINARY)
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_start = None
    y_start = None
    x_end = None
    y_end = None

    for cnt in contours:
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)

        if x_start is None or x_cnt < x_start:
            x_start = x_cnt

        if y_start is None or y_cnt < y_start:
            y_start = y_cnt

        if x_end is None or x_cnt + w_cnt > x_end:
            x_end = x_cnt + w_cnt

        if y_end is None or y_cnt + h_cnt > y_end:
            y_end = y_cnt + h_cnt

    return input_image[y_start: y_end, x_start:x_end]


def scale_image(input_image):
    length = len(input_image)
    width = len(input_image[0])
    diff_length_width = int(abs(length - width) / 2)

    if length > width:
        scale_correct = cv2.copyMakeBorder(input_image, top=0, bottom=0, left=diff_length_width, right=diff_length_width, borderType= cv2.BORDER_CONSTANT, value=255)
    else:
        scale_correct = cv2.copyMakeBorder(input_image, top=diff_length_width, bottom=diff_length_width, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=255)

    scaled_image = cv2.resize(scale_correct, (96, 96), cv2.INTER_AREA)
    padded_image = cv2.copyMakeBorder(scaled_image, top=16, bottom=16, left=16, right=16, borderType=cv2.BORDER_CONSTANT, value=255)
    output_image = cv2.resize(padded_image, (56, 56), cv2.INTER_AREA)
    ret, thresh = cv2.threshold(output_image, 60, 255, cv2.THRESH_BINARY)
    return thresh


def preprocess(input_image):
    filled_lines = segment_lines(input_image, 100)
    output = []

    for line in filled_lines:
        characters = segment_characters(line, 100)
        for character in characters:
            output.append(scale_image(character))

    return output

if __name__ == "__main__":
    input_image = cv2.imread("/Users/adrianlim/IdeaProjects/CMPT-414-CV-OCR/data/images/font/Sample001/img001-00003.png", 0)

    i = extract_outer_box(input_image)
    i = scale_image(i)

    pyplot.imshow(i)
    pyplot.show()
