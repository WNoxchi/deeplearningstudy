from pathlib import Path
from shutil import rmtree, copyfile
import numpy as np
import os

def create_cifar_subset(path, fullpath=Path(), copypath='', p=0.1, copydirs=['train','valid','test']):
    """Copies subset `p` percent of a dataset: dataroot/ -> dataroot_tmp/, uniformly sampled."""
    if not copypath:
        copypath = Path(str(path) + '_tmp')
        if os.path.exists(copypath): rmtree(copypath)
    else:
        copypath/=path
    fullpath /= path
    copies = []
    dirs = os.listdir(fullpath)
    for f in dirs:
        if (fullpath/f).is_dir() and (copydirs==[] or f in copydirs):
            os.makedirs(copypath/f)
            create_cifar_subset(f, fullpath, copypath, copydirs=[])
        else:
            copies.append(f)
    if copies:
        copies = np.random.choice(copies, max(1, int(len(copies)*p)), replace=False)
        for copy in copies:
            copyfile(fullpath/copy, copypath/copy)
    
    return copypath
    
def count_files(path, fullpath=Path(), count=0):
    """Counts all files in a directory recursively."""
    fullpath /= path
    # check root exists
    if not os.path.exists(fullpath):
        print('Directory does not exist.')
        return
    dirs = os.listdir(fullpath)
    for direc in dirs:
        if (fullpath/direc).is_dir():
            count += count_files(direc, fullpath)
        else:
            count += 1
    return count