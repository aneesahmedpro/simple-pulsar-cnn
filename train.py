#!/usr/bin/env python3

import os

import numpy as np
import tensorflow as tf

from cnn import cnn_model_fn
from config import BATCH_SIZE
from local_settings import MODEL_DIR, TRAINING_DATA_DIR


def main(unused_argv):

    npy_data_dir = TRAINING_DATA_DIR / 'npy'
    cwd_old = os.getcwd()
    os.chdir(npy_data_dir)
    train_data = np.load('x_train.npy').astype(np.float32)
    train_labels = np.load('y_train.npy')
    eval_data = np.load('x_test.npy').astype(np.float32)
    eval_labels = np.load('y_test.npy')
    os.chdir(cwd_old)

    if len(train_data) % BATCH_SIZE != 0:
        print('[WARNING] The mini-batch size ({}) is not a divisor '
              'of the total number of training samples ({}).'.format(
                  BATCH_SIZE, len(train_data)))

    steps_per_epoch = len(train_data) // BATCH_SIZE

    # Labels are integers from set {0, 1}.
    # Samples with label '1' are fewer than those with label '0'.
    counts = np.bincount(train_labels).astype(np.float)
    class_weights = counts[0] / counts

    run_config = tf.estimator.RunConfig(
        model_dir=MODEL_DIR,
        save_summary_steps=1,
        keep_checkpoint_max=100,
        log_step_count_steps=1,
        train_distribute=None)

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        params={'class_weights': class_weights, 'return_all_layers': False},
        config=run_config)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'input': train_data},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'input': eval_data},
        y=eval_labels,
        # batch_size=len(eval_data),  # Might eat up all RAM and freeze system
        num_epochs=1,
        shuffle=False)

    tf.logging.set_verbosity(tf.logging.INFO)

    while True:
        try:
            epochs = input('\nHow many more epochs? ')
        except EOFError:
            print('[EOF]\nExiting...')
            break
        if not epochs:
            epochs = 1
        else:
            try:
                epochs = int(epochs)
            except ValueError:
                print('Invalid input.')
                continue
            if epochs <= 0:
                print('Exiting...')
                break
        for __ in range(epochs):
            classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)
            classifier.evaluate(input_fn=eval_input_fn)
        print()


if __name__ == '__main__':

    tf.app.run()
