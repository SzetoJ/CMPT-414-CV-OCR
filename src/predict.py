from tensorflow.contrib.learn.python import SKCompat
from tensorflow.contrib import learn
from model import cnn_model_fn
import cv2
import numpy as np


def predict(input_image):
    # Create the Estimator
    mnist_classifier = SKCompat(learn.Estimator(model_fn=cnn_model_fn, model_dir="../models/mnist_convnet_model"))

    # Do Prediction and print results
    predictions = mnist_classifier.predict(np.array([input_image], dtype=np.float32))
    print(predictions)


if __name__ == "__main__":
    image = cv2.imread("../images/TestImage7.jpg", 0)
    predict(image)
