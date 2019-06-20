#!/usr/bin/env python3

import os
import sys
import pathlib

import numpy as np

from stratified_shuffle_split import stratified_shuffle_split_for_binary
from local_settings import TRAINING_DATA_DIR


def main(dataset_npz_filepath):

    data = np.load(dataset_npz_filepath)
    x = data['chi_vs_DM_plots']
    y = data['labels']

    retval = stratified_shuffle_split_for_binary(x, y, test_fraction=0.333)
    x_train, y_train, x_test, y_test = retval

    os.chdir(str(TRAINING_DATA_DIR/'npy'))
    np.save('x_train.npy', x_train, allow_pickle=False)
    np.save('y_train.npy', y_train, allow_pickle=False)
    np.save('x_test.npy', x_test, allow_pickle=False)
    np.save('y_test.npy', y_test, allow_pickle=False)

    print('Data successfully compiled into npy containers.')


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Invalid invocation.\nUsage: {} {}'.format(
            sys.argv[0],
            '/path/to/dataset_file.npz'))
        exit(1)

    dataset_npz_filepath = pathlib.Path(sys.argv[1]).absolute()
    if not dataset_npz_filepath.exists() or dataset_npz_filepath.is_dir():
        print('Bad path: "{}"'.format(dataset_npz_filepath))
        exit(1)

    main(dataset_npz_filepath)
