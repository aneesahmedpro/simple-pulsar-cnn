import tensorflow as tf

from config import CNN_INPUT_LENGTH


def cnn_model_fn(features, labels, mode, params):

    input_shape = (-1, CNN_INPUT_LENGTH, 1)
    input_layer = tf.reshape(features['input'], input_shape)

    # shape: (?, 256, 1)

    conv1 = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=16,
        strides=1,
        padding='same',
        activation=tf.keras.activations.relu)(input_layer)

    # shape: (?, 256, 32)

    pool1 = tf.keras.layers.MaxPooling1D(
        pool_size=8,
        strides=8,
        padding='same')(conv1)

    # shape: (?, 32, 32)

    conv2 = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=8,
        strides=1,
        padding='same',
        activation=tf.keras.activations.relu)(pool1)

    # shape: (?, 32, 32)

    pool2 = tf.keras.layers.MaxPooling1D(
        pool_size=4,
        strides=4,
        padding='same')(conv2)

    # shape: (?, 8, 32)

    flat = tf.keras.layers.Flatten()(pool2)

    # shape: (?, 8*32)

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
                'conv1': conv1,
                'pool1': pool1,
                'conv2': conv2,
                'pool2': pool2,
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
