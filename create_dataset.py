import sys
import subprocess
import pathlib
import multiprocessing
import pickle

import numpy as np
import scipy.stats
import presto.prepfold as prepfold
import presto.psr_utils as psr_utils


def normalise_2d_rowwise(array2d):
    # This is somewhat similar to: prepfold.pfd.greyscale
    # The difference here is in normalising formula

    min_parts = np.minimum.reduce(array2d, 1)
    array2d -= min_parts[:, np.newaxis]
    array2d /= np.max(array2d)
    return array2d


def normalise_1d(array1d):
    # This is somewhat similar to: prepfold.pfd.plot_sumprofs
    # The difference here is in normalising formula

    array1d -= np.min(array1d)
    array1d /= np.max(array1d)
    return array1d


def calc_features_from_pfd(pfd_filepath):

    pfd_data = prepfold.pfd(str(pfd_filepath))

    if pfd_filepath.parent.name == 'positive':
        label = 1
    elif pfd_filepath.parent.name == 'negative':
        label = 0
    else:
        label = -1
        # raise RuntimeError('unable to decide the label of pfd file: {}'.format(
        #     str(pfd_filepath)))

    pfd_data.dedisperse()

    #### As done in: prepfold.pfd.plot_sumprofs
    profile = pfd_data.sumprof
    profile = normalise_1d(profile)
    ####

    profile_mean = np.mean(profile)
    profile_std_dev = np.std(profile)
    profile_skewness = scipy.stats.skew(profile)
    profile_excess_kurtosis = scipy.stats.kurtosis(profile)

    profiles_sum_axis0 = pfd_data.profs.sum(0)

    #### As done in: prepfold.pfd.plot_chi2_vs_DM
    loDM = 0
    hiDM = pfd_data.numdms
    N = pfd_data.numdms
    profs = profiles_sum_axis0.copy()  # = pfd_data.profs.sum(0)
    DMs = psr_utils.span(loDM, hiDM, N)
    chis = np.zeros(N, dtype='f')
    subdelays_bins = pfd_data.subdelays_bins.copy()
    for ii, DM in enumerate(DMs):
        subdelays = psr_utils.delay_from_DM(DM, pfd_data.barysubfreqs)
        hifreqdelay = subdelays[-1]
        subdelays = subdelays - hifreqdelay
        delaybins = subdelays*pfd_data.binspersec - subdelays_bins
        new_subdelays_bins = np.floor(delaybins+0.5)
        for jj in range(pfd_data.nsub):
            profs[jj] = psr_utils.rotate(profs[jj], int(new_subdelays_bins[jj]))
        subdelays_bins += new_subdelays_bins
        sumprof = profs.sum(0)
        chis[ii] = pfd_data.calc_redchi2(prof=sumprof, avg=pfd_data.avgprof)
    ####

    best_dm = pfd_data.bestdm

    # crop_radius = 100
    # best_dm_index = np.searchsorted(DMs, best_dm)  # Not accurate, but close.
    # bloated_chis = np.insert(chis, N, np.full(crop_radius, chis[-1]))
    # bloated_chis = np.insert(bloated_chis, 0, np.full(crop_radius, chis[0]))
    # cropped_chis = bloated_chis[ best_dm_index : best_dm_index+2*crop_radius ]
    # chis = cropped_chis

    chis_mean = np.mean(chis)
    chis_std_dev = np.std(chis)
    chis_skewness = scipy.stats.skew(chis)
    chis_excess_kurtosis = scipy.stats.kurtosis(chis)

    #### As done in: prepfold.pfd.plot_intervals
    intervals = pfd_data.profs.sum(1)
    intervals = normalise_2d_rowwise(intervals)
    ####

    #### As done in: prepfold.pfd.plot_subbands
    subbands = profiles_sum_axis0.copy()  # = pfd_data.profs.sum(0)
    subbands = normalise_2d_rowwise(subbands)
    ####

    return (label, profile_mean, profile_std_dev, profile_skewness,
        profile_excess_kurtosis, chis_mean, chis_std_dev, chis_skewness,
        chis_excess_kurtosis, best_dm, profile, intervals, subbands, chis)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Invalid invocation.\nUsage: {} {} {}'.format(
            sys.argv[0],
            '/path/to/dir_containing_pfd_files',
            '/path/to/dataset_file.pickled'))
        exit(1)

    root_dirpath = pathlib.Path(sys.argv[1]).absolute()
    if not root_dirpath.exists() or not root_dirpath.is_dir():
        print('Bad path: "{}"'.format(root_dirpath))
        exit(1)

    dataset_filepath = pathlib.Path(sys.argv[2]).absolute()
    if dataset_filepath.exists() and dataset_filepath.is_dir():
        print('Bad path: "{}"'.format(dataset_filepath))
        exit(1)

    with open(str(dataset_filepath), 'wb') as f:
        f.write(b'Just testing writability...')

    print('\nCollecting ".pfd" files... ', end='', flush=True)

    try:
        cmd = ['find', str(root_dirpath), '-name', '*.pfd']
        pfd_filepaths_str = subprocess.check_output(cmd).decode()
    except:
        print('Failed to execute: "{}"'.format(' '.join(cmd)))
        exit(1)

    if pfd_filepaths_str:
        pfd_filepaths = pfd_filepaths_str.strip().split('\n')
        pfd_filepaths = [pathlib.Path(x) for x in pfd_filepaths]
        print('Done. Found {}.'.format(len(pfd_filepaths)))
    else:
        print('Done. Failed to find any.')
        exit(1)

    # import random; pfd_filepaths = random.sample(pfd_filepaths, 8)

    print('\nExtracting features...')

    pool = multiprocessing.Pool()
    results = []
    for pfd_filepath in pfd_filepaths:
        result = pool.apply_async(calc_features_from_pfd, (pfd_filepath,))
        results.append(result)
    pool.close()

    for result in results:
        result.wait()
        print('#', end='', flush=True)

    print('\nDone.')

    stats = []
    time_plots = []
    phase_time_plots = []
    phase_band_plots = []
    chi_vs_DM_plots = []
    labels = []
    for result in results:
        (label, profile_mean, profile_std_dev, profile_skewness,
            profile_excess_kurtosis, chis_mean, chis_std_dev,
            chis_skewness, chis_excess_kurtosis, best_dm, profile,
            intervals, subbands, chis) = result.get()
        stats.append([
            profile_mean, profile_std_dev, profile_skewness,
            profile_excess_kurtosis, chis_mean, chis_std_dev, chis_skewness,
            chis_excess_kurtosis, best_dm])
        time_plots.append(profile)
        phase_time_plots.append(intervals)
        phase_band_plots.append(subbands)
        chi_vs_DM_plots.append(chis)
        labels.append(label)

    pfd_filepaths = [str(p.relative_to(root_dirpath)) for p in pfd_filepaths]

    print('\nWriting to disk... ', end='', flush=True)

    data = {
        'stats': stats,
        'time_plots': time_plots,
        'phase_time_plots': phase_time_plots,
        'phase_band_plots': phase_band_plots,
        'chi_vs_DM_plots': chi_vs_DM_plots,
        'labels': labels,
        'pfd_filepaths': pfd_filepaths,
    }
    with open(str(dataset_filepath), 'wb') as f:
        pickle.dump(data, f, protocol=1)

    print('Done.')
