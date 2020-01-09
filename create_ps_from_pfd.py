import sys
import os
import subprocess
import pathlib
import multiprocessing


def create_ps_from_pfd(pfd_filepath):

    try:
        cmd = ['show_pfd', '-noxwin', str(pfd_filepath)]
        devnull = open(os.devnull, 'wb')
        subprocess.check_call(
            cmd,
            cwd=str(pfd_filepath.parent),
            stdout=devnull,
            stderr=subprocess.STDOUT)
        devnull.close()
        return '[ DONE ] {}'.format(' '.join(cmd))
    except subprocess.CalledProcessError:
        return '[FAILED] {}'.format(' '.join(cmd))


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Invalid invocation.\n'
              'Usage: {} /path/to/dir_containing_pfd_files'.format(
                  sys.argv[0]))
        exit(1)

    root_dirpath = pathlib.Path(sys.argv[1]).absolute()
    if not root_dirpath.exists() or not root_dirpath.is_dir():
        print('Bad path: "{}"'.format(root_dirpath))
        exit(1)

    try:
        cmd = ['find', str(root_dirpath), '-name', '*.pfd']
        pfd_filepaths_str = subprocess.check_output(cmd).decode()
    except subprocess.CalledProcessError:
        print('Failed to execute: "{}"'.format(' '.join(cmd)))
        exit(1)

    if pfd_filepaths_str:
        pfd_filepaths = pfd_filepaths_str.strip().split('\n')
        pfd_filepaths = [pathlib.Path(x) for x in pfd_filepaths]
        print('Found {} PFDs.\n'.format(len(pfd_filepaths)))
    else:
        print('Failed to find any PFDs.\n')
        exit(1)

    pool = multiprocessing.Pool()
    results = []
    for pfd_filepath in pfd_filepaths:
        result = pool.apply_async(create_ps_from_pfd, (pfd_filepath,))
        results.append(result)
    pool.close()

    for result in results:
        result.wait()
        msg = result.get()
        print(msg)

    try:
        cmd = ['find', str(root_dirpath), '-name', '*.bestprof', '-delete']
        pfd_filepaths_str = subprocess.check_output(cmd)
    except subprocess.CalledProcessError:
        print('Failed to execute: "{}"'.format(' '.join(cmd)))
        exit(1)

    print('\nDone.')
