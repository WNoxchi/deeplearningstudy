import shutil
import os
import numpy as np
import time

def count_files(path, verbose=False):
    count = 0
    for folder in path.iterdir():
        count += len(list(folder.glob('*')))
        if verbose: print(f'{folder}: {len(list(folder.glob("*")))}')
    return count

def clear_cpu_dir(cpu_path, verbose=False):
    if verbose: print('deleting: ', cpu_path)
    shutil.rmtree(cpu_path)
    if verbose: print('done')

def get_leaf(path):
    """returns the last element of a path"""
    return str(path).split('/')[-1]

def create_cpu_dataset(path, p=1000, subfolders='train', seed=0):
    """
        Creates a temporary sub-dataset for cpu-machine work.

        path : (pathlib.Path) root directory of dataset
        p    : (float, int) proportion for subset. If p <= 1: treated
                            as percetnage. If p > 1: treated as absolute
                            count and converetd to percentage.
        subfolders : (list(str), str) data subdirectories to copy.

        NOTE: currently the `shutil.copyfile` calls take quite
              a bit of time.
    """
    cpu_path = path/'cpu'
    if subfolders == None: subfolders = ['train','valid','test']
    if type(subfolders) == str: subfolders = [subfolders]
    np.random.seed(seed=seed)

    # delete & recreate cpu_path directory
    clear_cpu_dir(cpu_path, verbose=False)
    os.makedirs(cpu_path)

    for subfolder in subfolders:
        # if p absolute: calculate percentage
        if p > 1:
            count = count_files(path/subfolder)
            t = p
            p = min(1.0, p/count)
            count *= p
        else:
            count = p * count_files(path/subfolder)
        # copy files to cpu directory
        os.makedirs(cpu_path/subfolder)
        for clas in os.listdir(path/subfolder):
            os.makedirs(cpu_path/subfolder/clas)
            flist = list((path/subfolder/clas).iterdir())
            n_copy = int(np.round(len(flist) * p))
            flist = np.random.choice(flist, n_copy, replace=False)
            for f in flist:
                fname = get_leaf(f)
                shutil.copyfile(f, cpu_path/subfolder/clas/fname)
                count -= 1
        # cap off total copied
        while count > 0:
            for clas in os.listdir(path/subfolder):
                if count == 0: break
                flist = list((path/subfolder/clas).iterdir())
                f = np.random.choice(flist)
                while get_leaf(f) in os.listdir(cpu_path/subfolder/clas):
                    f = np.random.choice(flist)
                fname = get_leaf(f)
                shutil.copyfile(f, cpu_path/subfolder/clas/fname)
                count -= 1
