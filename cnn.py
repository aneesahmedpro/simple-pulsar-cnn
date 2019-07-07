import tensorflow as tf

from config import (PHASE_TIME_PLOT_WIDTH, PHASE_TIME_PLOT_HEIGHT,
                    PHASE_BAND_PLOT_WIDTH, PHASE_BAND_PLOT_HEIGHT)


def cnn_model_fn(features, labels, mode, params):

    A_input_shape = (-1, PHASE_TIME_PLOT_HEIGHT, PHASE_TIME_PLOT_WIDTH, 1)
    B_input_shape = (-1, PHASE_BAND_PLOT_HEIGHT, PHASE_BAND_PLOT_WIDTH, 1)

    A_input = tf.reshape(features['phase_time_plots'], A_input_shape)

    # A_shape: (?, 64, 128, 1)

    A_conv1 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=16,
        strides=1,
        padding='same',
        activation=tf.keras.activations.relu)(A_input)

    # A_shape: (?, 64, 128, 32)

    A_pool1 = tf.keras.layers.MaxPooling2D(
        pool_size=8,
        strides=8,
        padding='same')(A_conv1)

    # A_shape: (?, 8, 16, 32)

    A_conv2 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=8,
        strides=1,
        padding='same',
        activation=tf.keras.activations.relu)(A_pool1)

    # A_shape: (?, 8, 16, 32)

    A_pool2 = tf.keras.layers.MaxPooling2D(
        pool_size=4,
        strides=4,
        padding='same')(A_conv2)

    # A_shape: (?, 2, 4, 32)

    A_flat = tf.keras.layers.Flatten()(A_pool2)

    # A_shape: (?, 2*4*32)

    B_input = tf.reshape(features['phase_band_plots'], B_input_shape)

    # B_shape: (?, 128, 128, 1)

    B_conv1 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=16,
        strides=1,
        padding='same',
        activation=tf.keras.activations.relu)(B_input)

    # B_shape: (?, 128, 128, 32)

    B_pool1 = tf.keras.layers.MaxPooling2D(
        pool_size=8,
        strides=8,
        padding='same')(B_conv1)

    # B_shape: (?, 16, 16, 32)

    B_conv2 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=8,
        strides=1,
        padding='same',
        activation=tf.keras.activations.relu)(B_pool1)

    # B_shape: (?, 16, 16, 32)

    B_pool2 = tf.keras.layers.MaxPooling2D(
        pool_size=4,
        strides=4,
        padding='same')(B_conv2)

    # B_shape: (?, 4, 4, 32)

    B_flat = tf.keras.layers.Flatten()(B_pool2)

    # B_shape: (?, 4*4*32)

    flat = tf.keras.layers.Concatenate(axis=1)([A_flat, B_flat])

    dense = tf.keras.layers.Dense(
        units=1024,
        activation=tf.keras.activations.relu)(flat)

    # shape: (?, 1024)

    dropout = tf.keras.layers.Dropout(rate=0.4)(dense)

    # shape: (?, 1024)

    logits = tf.keras.layers.Dense(units=2)(dropout)

    # shape: (?, 2)

    predicted_class = tf.argmax(input=logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_probabilities = tf.nn.softmax(logits)
        prediction_data = {
            'class': predicted_class,
            'probabilities': predicted_probabilities,
        }
        if params['return_all_layers']:
            prediction_data.update({
                'A_conv1': A_conv1,
                'A_pool1': A_pool1,
                'A_conv2': A_conv2,
                'A_pool2': A_pool2,
                'A_flat': A_flat,
                'B_conv1': B_conv1,
                'B_pool1': B_pool1,
                'B_conv2': B_conv2,
                'B_pool2': B_pool2,
                'B_flat': B_flat,
                'flat': flat,
                'dense': dense,
                'dropout': dropout,
                'logits': logits,
            })
        return tf.estimator.EstimatorSpec(mode, prediction_data)

    class_weights = tf.constant(params['class_weights'])

    weights = tf.gather(class_weights, labels)

    loss_op = tf.losses.sparse_softmax_cross_entropy(labels, logits, weights)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss_op,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, train_op=train_op)

    accuracy = tf.metrics.accuracy(labels, predicted_class)
    rms_error = tf.metrics.root_mean_squared_error(labels, predicted_class)
    true_neg = tf.metrics.true_negatives(labels, predicted_class)
    false_pos = tf.metrics.false_positives(labels, predicted_class)
    false_neg = tf.metrics.false_negatives(labels, predicted_class)
    true_pos = tf.metrics.true_positives(labels, predicted_class)

    mask_for_pos = tf.equal(labels, 1)
    precision_for_pos = tf.metrics.precision(
        labels, predicted_class, weights=mask_for_pos)
    recall_for_pos = tf.metrics.recall(
        labels, predicted_class, weights=mask_for_pos)

    eval_metric_ops = {
        'accuracy': accuracy,
        'root_mean_squared_error': rms_error,
        'true_negatives': true_neg,
        'false_positives': false_pos,
        'false_negatives': false_neg,
        'true_positives': true_pos,
        'precision_for_positives': precision_for_pos,
        'recall_for_positives': recall_for_pos,
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, eval_metric_ops=eval_metric_ops)
