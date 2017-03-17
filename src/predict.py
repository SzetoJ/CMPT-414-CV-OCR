from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2
import numpy

FLAGS = None


def predict(image):
    # Read Image
    image = cv2.imread(image, 0)
    flattened_image = numpy.reshape(image, 784).astype('float32')

    # Create Model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    saver.restore(sess, "../models/model.ckpt")

    # Do Prediction
    z = tf.argmax(y, 1)
    print(sess.run(z, feed_dict={x: [flattened_image]}))


if __name__ == "__main__":
    predict("../images/TestImage7.jpg")
