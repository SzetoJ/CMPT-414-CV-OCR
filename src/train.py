from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat
from model import cnn_model_fn
import numpy as np
import tensorflow as tf


def train(unused_argv):
    mnist = learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = SKCompat(learn.Estimator(model_fn=cnn_model_fn, model_dir="../models/mnist_convnet_model"))

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    mnist_classifier.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
        steps=20000,
        monitors=[logging_hook]
    )

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # Evaluate the model and print results
    eval_results = mnist_classifier.score(x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=train)
