from tensorflow.contrib.learn.python import SKCompat
from tensorflow.contrib import learn
from model import cnn_model_fn
import cv2
import numpy as np
import preprocess
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
import mapping


def predict(input_image):
    # Create the Estimator
    mnist_classifier = SKCompat(learn.Estimator(model_fn=cnn_model_fn, model_dir="../models/char74_convnet_model"))

    # Do Prediction and print results
    predictions = mnist_classifier.predict(np.array([input_image], dtype=np.float32))
    print(mapping.char74_mapping[predictions['classes'][0]])


if __name__ == "__main__":
    SHOW_PROCESSED_IMAGES = False

    image = cv2.imread("/Users/adrianlim/IdeaProjects/CMPT-414-CV-OCR/data/input/ABC.jpg", 0)
    segmented_image = preprocess.extract_boxes(image)
    for segment in segmented_image:
        scaled_segment = preprocess.scale_image(segment)
        if SHOW_PROCESSED_IMAGES:
            pyplot.imshow(scaled_segment)
            pyplot.show()
        predict(scaled_segment)
