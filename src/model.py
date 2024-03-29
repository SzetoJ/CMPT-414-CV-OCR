import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


def cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features, [-1, 56, 56, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(input_layer, filters=32, kernel_size=[5, 5], padding="SAME", activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[5, 5], padding="SAME", activation=tf.nn.relu)

    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=[5, 5], padding="SAME", activation=tf.nn.relu)

    # Pooling Layer #3
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # Fully Connected Layer
    pool2_flat = tf.reshape(pool3, [-1, 7 * 7 * 64])
    fc1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=fc1, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=62)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=62)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss=loss,
                                                   global_step=tf.contrib.framework.get_global_step(),
                                                   learning_rate=0.002,
                                                   optimizer="SGD")

    # Generate Predictions
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
