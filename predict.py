import sys
import pathlib
import csv

import numpy as np
import tensorflow as tf

from cnn import cnn_model_fn
from config import CLASS_ID_TO_LABEL
from local_settings import MODEL_DIR


def main(dataset_npz_filepath, result_csv_filepath):

    data = np.load(dataset_npz_filepath)
    pfd_filepaths = data['pfd_filepaths']

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        params={'class_weights': None, 'return_all_layers': False},
        model_dir=MODEL_DIR)

    input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={k: v for k, v in data.items() if k != 'labels'},
        num_epochs=1,
        shuffle=False)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    output_generator = classifier.predict(input_fn=input_fn)

    with open(str(result_csv_filepath), 'w') as result_csv_file:
        csv_writer = csv.writer(result_csv_file)
        for output, pfd_filepath in zip(output_generator, pfd_filepaths):
            predicted_class_id = output['class']
            predicted_class_label = CLASS_ID_TO_LABEL[predicted_class_id]
            prediction_confidence = output['probabilities'][predicted_class_id]
            csv_writer.writerow([
                str(pfd_filepath),
                predicted_class_label,
                prediction_confidence])


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Invalid invocation.\nUsage: {} {} {}'.format(
            sys.argv[0],
            '/path/to/dataset_file.npz',
            '/path/to/prediction_result.csv'))
        exit(1)

    dataset_npz_filepath = pathlib.Path(sys.argv[1]).absolute()
    if not dataset_npz_filepath.exists() or dataset_npz_filepath.is_dir():
        print('Bad path: "{}"'.format(dataset_npz_filepath))
        exit(1)

    result_csv_filepath = pathlib.Path(sys.argv[2]).absolute()
    if result_csv_filepath.exists() and result_csv_filepath.is_dir():
        print('Bad path: "{}"'.format(result_csv_filepath))
        exit(1)

    with open(str(result_csv_filepath), 'w') as f:
        f.write('Just testing writability...')

    main(dataset_npz_filepath, result_csv_filepath)
