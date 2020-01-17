import tensorflow as tf

from config import (PHASE_TIME_PLOT_WIDTH, PHASE_TIME_PLOT_HEIGHT,
                    PHASE_BAND_PLOT_WIDTH, PHASE_BAND_PLOT_HEIGHT,
                    TIME_PLOT_LENGTH, CHI_VS_DM_PLOT_LENGTH)


def cnn_model_fn(features, labels, mode, params):

    A_input_shape = (-1, PHASE_TIME_PLOT_HEIGHT, PHASE_TIME_PLOT_WIDTH, 1)
    B_input_shape = (-1, PHASE_BAND_PLOT_HEIGHT, PHASE_BAND_PLOT_WIDTH, 1)
    C_input_shape = (-1, 2*TIME_PLOT_LENGTH, 1)
    D_input_shape = (-1, CHI_VS_DM_PLOT_LENGTH, 1)

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

    C_input = tf.reshape(features['time_plots'], C_input_shape)

    # C_shape: (?, 256, 1)

    C_conv1 = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=16,
        strides=1,
        padding='same',
        activation=tf.keras.activations.relu)(C_input)

    # C_shape: (?, 256, 32)

    C_pool1 = tf.keras.layers.MaxPooling1D(
        pool_size=8,
        strides=8,
        padding='same')(C_conv1)

    # C_shape: (?, 32, 32)

    C_conv2 = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=8,
        strides=1,
        padding='same',
        activation=tf.keras.activations.relu)(C_pool1)

    # C_shape: (?, 32, 32)

    C_pool2 = tf.keras.layers.MaxPooling1D(
        pool_size=4,
        strides=4,
        padding='same')(C_conv2)

    # C_shape: (?, 8, 32)

    C_flat = tf.keras.layers.Flatten()(C_pool2)

    # C_shape: (?, 8*32)

    D_input = tf.reshape(features['chi_vs_DM_plots'], D_input_shape)

    # D_shape: (?, 1000, 1)

    D_conv1 = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=16,
        strides=1,
        padding='same',
        activation=tf.keras.activations.relu)(D_input)

    # D_shape: (?, 1000, 32)

    D_pool1 = tf.keras.layers.MaxPooling1D(
        pool_size=8,
        strides=8,
        padding='same')(D_conv1)

    # D_shape: (?, 125, 32)

    D_conv2 = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=8,
        strides=1,
        padding='same',
        activation=tf.keras.activations.relu)(D_pool1)

    # D_shape: (?, 125, 32)

    D_pool2 = tf.keras.layers.MaxPooling1D(
        pool_size=4,
        strides=4,
        padding='same')(D_conv2)

    # D_shape: (?, 32, 32)

    D_flat = tf.keras.layers.Flatten()(D_pool2)

    # D_shape: (?, 32*32)

    flat = tf.keras.layers.Concatenate(axis=1)(
        [A_flat, B_flat, C_flat, D_flat])

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
                'C_conv1': C_conv1,
                'C_pool1': C_pool1,
                'C_conv2': C_conv2,
                'C_pool2': C_pool2,
                'C_flat': C_flat,
                'D_conv1': D_conv1,
                'D_pool1': D_pool1,
                'D_conv2': D_conv2,
                'D_pool2': D_pool2,
                'D_flat': D_flat,
                'flat': flat,
                'dense': dense,
                'dropout': dropout,
                'logits': logits,
            })
        return tf.estimator.EstimatorSpec(mode, prediction_data)

    class_weights = tf.constant(params['class_weights'])

    weights = tf.gather(class_weights, labels)

    loss_op = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        labels, logits, weights)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss_op,
            global_step=tf.compat.v1.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, train_op=train_op)

    accuracy = tf.compat.v1.metrics.accuracy(labels, predicted_class)
    true_neg = tf.compat.v1.metrics.true_negatives(labels, predicted_class)
    false_pos = tf.compat.v1.metrics.false_positives(labels, predicted_class)
    false_neg = tf.compat.v1.metrics.false_negatives(labels, predicted_class)
    true_pos = tf.compat.v1.metrics.true_positives(labels, predicted_class)

    mask = tf.equal(predicted_class, 1)
    precision_for_pos = tf.compat.v1.metrics.precision(
        labels, predicted_class, weights=mask)

    mask = tf.equal(labels, 1)
    recall_for_pos = tf.compat.v1.metrics.recall(
        labels, predicted_class, weights=mask)

    eval_metric_ops = {
        'accuracy': accuracy,
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
