import numpy as np
import pickle
import sys
import pathlib

from config import (PHASE_TIME_PLOT_WIDTH, PHASE_TIME_PLOT_HEIGHT,
                    PHASE_BAND_PLOT_WIDTH, PHASE_BAND_PLOT_HEIGHT,
                    TIME_PLOT_LENGTH, CHI_VS_DM_PLOT_LENGTH)


def main(dataset_pickled_filepath, dataset_npz_filepath):

    print('Filtering... ', end='', flush=True)

    with open(str(dataset_pickled_filepath), 'rb') as f:
        data = pickle.load(f)
    stats = data['stats']
    time_plots = data['time_plots']
    phase_time_plots = data['phase_time_plots']
    phase_band_plots = data['phase_band_plots']
    chi_vs_DM_plots = data['chi_vs_DM_plots']
    labels = data['labels']
    pfd_filepaths = data['pfd_filepaths']

    new_stats = []
    new_time_plots = []
    new_phase_time_plots = []
    new_phase_band_plots = []
    new_chi_vs_DM_plots = []
    new_chi_vs_DM_plots_lengths = []
    new_labels = []
    new_pfd_filepaths = []

    count_accepted = 0

    for i in range(len(labels)):

        if len(time_plots[i]) != TIME_PLOT_LENGTH:
            continue
        if phase_time_plots[i].shape != (PHASE_TIME_PLOT_HEIGHT, PHASE_TIME_PLOT_WIDTH):
            continue
        if phase_band_plots[i].shape != (PHASE_BAND_PLOT_HEIGHT, PHASE_BAND_PLOT_WIDTH):
            continue
        if len(chi_vs_DM_plots[i]) > CHI_VS_DM_PLOT_LENGTH:
            continue

        padded = np.full(CHI_VS_DM_PLOT_LENGTH, fill_value=0)
        padded[:len(chi_vs_DM_plots[i])] = chi_vs_DM_plots[i]

        # Repeat the plot once to concatenate the spikes split by the edge
        time_plots[i] = np.hstack([time_plots[i], time_plots[i]])

        new_stats.append(stats[i])
        new_time_plots.append(time_plots[i])
        new_phase_time_plots.append(phase_time_plots[i])
        new_phase_band_plots.append(phase_band_plots[i])
        new_chi_vs_DM_plots.append(padded)
        new_chi_vs_DM_plots_lengths.append(len(chi_vs_DM_plots[i]))
        new_labels.append(labels[i])
        new_pfd_filepaths.append(pfd_filepaths[i])

        count_accepted += 1

    new_stats = np.array(new_stats).astype(np.float32)
    new_time_plots = np.array(new_time_plots).astype(np.float32)
    new_phase_time_plots = np.array(new_phase_time_plots).astype(np.float32)
    new_phase_band_plots = np.array(new_phase_band_plots).astype(np.float32)
    new_chi_vs_DM_plots = np.array(new_chi_vs_DM_plots).astype(np.float32)
    new_chi_vs_DM_plots_lengths = np.array(
        new_chi_vs_DM_plots_lengths).astype(np.int)
    labels = np.array(labels).astype(np.int)
    pfd_filepaths = np.array(pfd_filepaths)

    print(count_accepted, 'accepted.')

    print('\nWriting to disk... ', end='', flush=True)

    data = {
        'stats': new_stats,
        'time_plots': new_time_plots,
        'phase_time_plots': new_phase_time_plots,
        'phase_band_plots': new_phase_band_plots,
        'chi_vs_DM_plots': new_chi_vs_DM_plots,
        'chi_vs_DM_plots_lengths': new_chi_vs_DM_plots_lengths,
        'labels': new_labels,
        'pfd_filepaths': new_pfd_filepaths,
    }
    np.savez_compressed(str(dataset_npz_filepath), **data)

    print('Done.')


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Invalid invocation.\nUsage: {} {} {}'.format(
            sys.argv[0],
            '/path/to/dataset_file.pickled',
            '/path/to/dataset_file.npz'))
        exit(1)

    dataset_pickled_filepath = pathlib.Path(sys.argv[1]).absolute()
    if dataset_pickled_filepath.exists() and dataset_pickled_filepath.is_dir():
        print('Bad path: "{}"'.format(dataset_pickled_filepath))
        exit(1)

    dataset_npz_filepath = pathlib.Path(sys.argv[2]).absolute()
    if dataset_npz_filepath.exists() and dataset_npz_filepath.is_dir():
        print('Bad path: "{}"'.format(dataset_npz_filepath))
        exit(1)

    with open(str(dataset_npz_filepath), 'wb') as f:
        f.write(b'Just testing writability...')

    main(dataset_pickled_filepath, dataset_npz_filepath)
