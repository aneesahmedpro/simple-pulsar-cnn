#!/usr/bin/env python3

import sys
import pathlib

import numpy as np

from stratified_shuffle_split import stratified_shuffle_split_for_binary
from local_settings import TRAINING_DATA_DIR


def main(dataset_npz_filepath):

    data = np.load(dataset_npz_filepath)
    x2 = data['phase_band_plots'].astype(np.float32)
    x1 = data['phase_time_plots'].astype(np.float32)
    x3 = data['chi_vs_DM_plots'].astype(np.float32)
    y = data['labels']

    train_idx, test_idx = stratified_shuffle_split_for_binary(
        y, test_fraction=0.333)
    x1_train = x1[train_idx]
    x2_train = x2[train_idx]
    x3_train = x3[train_idx]
    y_train = y[train_idx]
    x1_test = x1[test_idx]
    x2_test = x2[test_idx]
    x3_test = x3[test_idx]
    y_test = y[test_idx]

    data_train = {
        'phase_time_plots': x1_train,
        'phase_band_plots': x2_train,
        'chi_vs_DM_plots': x3_train,
        'labels': y_train,
    }
    data_test = {
        'phase_time_plots': x1_test,
        'phase_band_plots': x2_test,
        'chi_vs_DM_plots': x3_test,
        'labels': y_test,
    }

    np.savez_compressed(TRAINING_DATA_DIR/'npy'/'train.npz', **data_train)
    np.savez_compressed(TRAINING_DATA_DIR/'npy'/'test.npz', **data_test)

    print('Data successfully compiled into npz containers.')


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
