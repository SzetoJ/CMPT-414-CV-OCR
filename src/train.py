from tensorflow.contrib import learn
from model import cnn_model_fn
import numpy as np
import tensorflow as tf
import cv2
import preprocess


def split_image_data(file_list_path, training_set_percent=0.7):
    with open(file_list_path, 'r') as input_file:
        file_list = input_file.read().splitlines()

    number_files = len(file_list)
    print("Total number of files: {}".format(number_files))

    training_set = np.random.choice(file_list, int(round(number_files * training_set_percent)), replace=False)
    print("Size training set: {}".format(str(len(training_set))))

    evaluation_set = list(set(file_list) ^ set(training_set))
    print("Size test set: {}".format(str(len(evaluation_set))))

    return {"training_set": training_set,
            "evaluation_set": evaluation_set[:len(evaluation_set)],
            "test_set": evaluation_set[len(evaluation_set):]}


def image_generator_builder(dir_path, image_list):
    def image_generator():
        for i in image_list:
            image = cv2.imread(dir_path + i + ".png", 0)
            outer_image = preprocess.extract_outer_box(image)
            scaled_image = preprocess.scale_image(outer_image)
            yield np.array([scaled_image], dtype=np.float32)
    return image_generator()


def label_generator_builder(image_list):
    def label_generator():
        for i in image_list:
            sub_path = i.split("/")[0]
            stripped_path = sub_path.strip("Sample")
            index = int(stripped_path) - 1
            yield np.asarray(index, dtype=np.float32)
    return label_generator()


def repeat_list(list, repeats):
    output_list = []
    for i in range(repeats):
        output_list.extend(list)
    return output_list


def get_subset(list, size):
    return np.random.choice(list, size, replace=False)


def train(unused_argv):
    # Prepare training and evaluation data
    file_dict = split_image_data("../data/images/list/all.txt", training_set_percent=0.9)
    training_set = file_dict["training_set"]
    number_epochs = 3
    new_training_set = repeat_list(training_set, number_epochs)
    evaluation_set = file_dict["evaluation_set"]
    test_set = file_dict["test_set"]

    # Create the Estimator
    classifier = learn.Estimator(model_fn=cnn_model_fn,
                                 model_dir="../models/56x56_3_layers_scaled_char74_convnet_model",
                                 config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60))

    # Set up logging for predictions
    evaluation_metrics = {"validation_accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes")}

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        np.array(list(image_generator_builder("../data/images/font/", evaluation_set))),
        np.array(list(label_generator_builder(evaluation_set))),
        every_n_steps=50,
        metrics=evaluation_metrics,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=200
    )

    training_test_set = get_subset(training_set, len(evaluation_set))
    training_metrics = {"training_accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes")}
    training_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        np.array(list(image_generator_builder("../data/images/font/", training_test_set))),
        np.array(list(label_generator_builder(training_test_set))),
        every_n_steps=50,
        metrics=training_metrics,
        early_stopping_metric="loss",
        early_stopping_metric_minimize=True,
        early_stopping_rounds=200
    )

    # Train the model
    classifier.fit(
        x=image_generator_builder("../data/images/font/", new_training_set),
        y=label_generator_builder(new_training_set),
        batch_size=100,
        steps=1280,
        monitors=[validation_monitor, training_monitor]
    )

    print("Model has been trained, beginning evaluation")

    # Configure the accuracy metric for evaluation
    prediction_metrics = {
        "test_accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    # Evaluate the model and print results
    eval_results = classifier.evaluate(x=image_generator_builder("../data/images/font/", test_set),
                                       y=label_generator_builder(test_set),
                                       metrics=prediction_metrics)
    print(eval_results)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=train)
