from tensorflow.contrib import learn
from tensorflow.contrib.learn.python import SKCompat
from model import cnn_model_fn
import numpy as np
import tensorflow as tf
import cv2


def split_image_data(file_list_path, training_set_percent=0.01):
    with open(file_list_path, 'r') as input_file:
        file_list = input_file.read().splitlines()

    number_files = len(file_list)
    print("Total number of files: {}".format(number_files))

    training_set = np.random.choice(file_list, int(round(number_files * training_set_percent)), replace=False)
    print("Size training set: {}".format(str(len(training_set))))

    evaluation_set = list(set(file_list) ^ set(training_set))
    print("Size test set: {}".format(str(len(evaluation_set))))

    return {"training_set": training_set,
            "evaluation_set": evaluation_set}


def image_generator_builder(dir_path, image_list):
    def image_generator():
        for i in image_list:
            image = cv2.imread(dir_path + i + ".png", 0)
            shrunk_image = cv2.resize(image, (28, 28), cv2.INTER_AREA)
            yield np.array([shrunk_image], dtype=np.float32)
    return image_generator()


def label_generator_builder(image_list):
    def label_generator():
        for i in image_list:
            sub_path = i.split("/")[0]
            stripped_path = sub_path.strip("Sample")
            index = int(stripped_path) - 1
            yield np.asarray(index, dtype=np.float32)
    return label_generator()


def train(unused_argv):
    # Prepare training and evaluation data
    file_dict = split_image_data("../data/images/list/all.txt")
    training_set = file_dict["training_set"]
    evaluation_set = file_dict["evaluation_set"]

    # Create the Estimator
    mnist_classifier = SKCompat(learn.Estimator(model_fn=cnn_model_fn, model_dir="../models/char74_convnet_model"))

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    mnist_classifier.fit(
        x=image_generator_builder("../data/images/font/", training_set),
        y=label_generator_builder(training_set),
        batch_size=4,
        steps=2,
        monitors=[logging_hook]
    )

    print("Model has been trained, beginning evaluation")

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # Evaluate the model and print results
    eval_results = mnist_classifier.score(x=image_generator_builder("../data/images/font/", evaluation_set),
                                          y=label_generator_builder(evaluation_set),
                                          metrics=metrics)
    print(eval_results)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=train)
