import numpy as np
import tensorflow as tf

from cnn import cnn_model_fn
from config import BATCH_SIZE
from local_settings import MODEL_DIR, TRAINING_DATA_DIR


def main(unused_argv):

    data_train = np.load(TRAINING_DATA_DIR/'npy'/'train.npz')
    data_test = np.load(TRAINING_DATA_DIR/'npy'/'test.npz')

    if len(data_train['labels']) % BATCH_SIZE != 0:
        print('[WARNING] The mini-batch size ({}) is not a divisor '
              'of the total number of training samples ({}).'.format(
                  BATCH_SIZE, len(data_train['labels'])))

    steps_per_epoch = len(data_train['labels']) // BATCH_SIZE

    if not steps_per_epoch:
        print('ERROR: Dataset size is itself smaller that mini-batch size.')
        exit(1)

    # Labels are integers from set {0, 1}.
    # Samples with label '1' are fewer than those with label '0'.
    counts = np.bincount(data_train['labels']).astype(np.float)
    class_weights = counts[0] / counts

    run_config = tf.estimator.RunConfig(
        model_dir=str(MODEL_DIR),
        save_summary_steps=1,
        keep_checkpoint_max=100,
        log_step_count_steps=1,
        train_distribute=None)

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        params={'class_weights': class_weights, 'return_all_layers': False},
        config=run_config)

    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={k: v for k, v in data_train.items() if k != 'labels'},
        y=data_train['labels'],
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={k: v for k, v in data_test.items() if k != 'labels'},
        y=data_test['labels'],
        # batch_size=len(eval_data),  # Might eat up all RAM and freeze system
        num_epochs=1,
        shuffle=False)

    print()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    while True:
        try:
            epochs = input('\nHow many more epochs? ')
        except EOFError:
            print('[EOF]\nExiting...')
            break
        except KeyboardInterrupt:
            print('[KeyboardInterrupt]\nExiting...')
            break
        if not epochs:  # if epochs is an empty string
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

    tf.compat.v1.app.run()
