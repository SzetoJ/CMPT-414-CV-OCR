from tensorflow.contrib import learn
from model import cnn_model_fn
import numpy as np
import tensorflow as tf
import cv2
import preprocess
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
import mapping


def split_image_data(file_list_path, training_set_percent=1):
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


if __name__ == "__main__":
    classifier = learn.Estimator(model_fn=cnn_model_fn,
                                 model_dir="/Users/adrianlim/IdeaProjects/CMPT-414-CV-OCR/models/56x56_3_layers_scaled_char74_convnet_model",
                                 config=tf.contrib.learn.RunConfig(save_checkpoints_secs=60))

    metrics = {
        "prediction_accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    for _ in range(1):
        file_dict = split_image_data("../data/images/list/all.txt", training_set_percent=1)
        training_set = file_dict["training_set"]
        evaluation_set = file_dict["evaluation_set"]
        test_set = image_generator_builder("../data/images/font/", training_set[:12600])
        test_labels = list(label_generator_builder(training_set[:12600]))

        fail_prediction = {}

        for i in range(62):
            fail_prediction[mapping.char74_mapping[i]] = 0

        prediction = list(classifier.predict(test_set))

        for i in range(12600):
            if int(prediction[i]['classes']) != int(test_labels[i]):
                fail_prediction[mapping.char74_mapping[int(test_labels[i])]] += 1

        print("Number Errors: {}".format(sum(fail_prediction.values())))
        print("Accuracy: {}".format((12600 - sum(fail_prediction.values()))/12600))

        dictionary = pyplot.figure()
        pyplot.bar(range(len(fail_prediction)), fail_prediction.values(), align='center')
        pyplot.xticks(range(len(fail_prediction)), fail_prediction.keys())
        pyplot.show()
