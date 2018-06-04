from pathlib import Path
from shutil import rmtree, copyfile
import matplotlib.pyplot as plt
import pandas as pd
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

def generate_csv(path, labelmap=None, folder='train'):
    """Infers a csv from directory structure. 
       `labelmap` is a dictionary mapping class folders to class names.
       Class names are taken from class folder names if no mapping is provided.
       For Single-Category Classification.
    """
    # find categories
    catfolders = [f for f in os.listdir(path/folder) if (path/folder/f).is_dir()]
    if labelmap is None: labelmap = {cf:cf for cf in catfolders}

    rows = []

    for cat in catfolders:
        catpath = path/folder/cat
        fpaths  = list(map(lambda x: cat+'/'+x, os.listdir(catpath)))
        rows.extend(list(zip(fpaths,[labelmap[cat] for i in range(len(fpaths))])))

    df = pd.DataFrame(rows, columns=['file','class'])
    return df

def pops_from_df(df, catdx=1, colnames=['cat','pop']):
    """Extracts category populations from a DataFrame.
       If `colnames=None`: returns a dictionary, otherwise
       a dataframe with `colnames` columns.
    """
    catcol = df.columns[catdx] # prevsly: y = df.columns[ydx]
    cats = df[catcol].unique()
    pops = [df[df[catcol]==cat].count()[0] for cat in cats] # prevsly: y -> catcol
    cat_pops = {cat:pop for cat,pop in zip(cats,pops)}
    
    if colnames:
        cat_pops = list(zip(cats,pops))
        cat_pops = pd.DataFrame(cat_pops, columns=colnames)
    else:
        cat_pops = {cat:pop for cat,pop in zip(cats,pops)}
    
    return cat_pops

def csv_subset(df, catdx=1, p=0.5):
    """Returns a percetnage of the original dataset, sampled uniformly by category."""
    if type(p)==int: p /= df.count()
    
    catcol = df.columns[catdx]
    cats   = df[catcol].unique()
    df_slice  = df[catcol]
    keep_idxs = np.array([], dtype=np.int64)
    
    for cat in cats:
        cat_idxs = np.where(df_slice==cat)[0]
        n = max(1, int(len(cat_idxs)*p))
        keep_idxs = np.concatenate((keep_idxs, np.random.choice(cat_idxs, size=n, replace=False)))
    
    return df.iloc[keep_idxs]

def smooth_csv_dataset(df, eps=0.1, full_df=None, catdx=1):
    """'Smooths' out a dataset by adding copied samples.
        For use with single-label classification (2-column) CSVs.
    """
    # result DF and sampling DF
    new_df  = df.copy()
    full_df = df if full_df is None else full_df
    # get category column name
    catcol = df.columns[catdx]
    
    # get category populations & calculate desired range
    cat_pops = pops_from_df(df)
    c_totals = cat_pops.as_matrix(columns=['pop'])
    sd       = eps * c_totals.mean()
    # Normalize category sizes
    c_norm     = c_totals/c_totals.max()
    new_mean   = c_totals.max() - sd
    new_totals = (2*sd * c_norm + (new_mean - sd)).astype(int)
    
    # Increase category sizes by differences
    diffs     = new_totals - c_totals
    cats      = cat_pops['cat'].values
    copy_idxs = []
    
    for i,cat in enumerate(cats):
        diff         = diffs[i]
        cat_idxs     = np.where(full_df[catcol]==cat)[0]
        full_cat_pop = len(cat_idxs)
        
        # if the difference is more than Nx greater, copy the whole category N times
        if diff > full_cat_pop:
            n_copy = int(diff) // full_cat_pop
            diff  -= n_copy * full_cat_pop
            for i in range(n_copy): copy_idxs.extend(cat_idxs)
        copy_idxs.extend(np.random.choice(cat_idxs, size=diff, replace=False))
    
    copy_rows = full_df.iloc[copy_idxs]
    new_df    = new_df.append(copy_rows)
    
    return new_df

def plot_pops(df, print_ms=True):
    if print_ms: print(f"{df.mean()[0]:.2f} {df.std()[0]:.2f}")
    df.plot.bar(x=df[df.columns[0]], ylim=(0,1.2*df[df.columns[1]].values.max()), 
            yerr=max(df.mean()[0]*0.005, df.std()[0]), alpha=.8)
    df.mean()[0], df.std()[0]
    
def basic_pop_plot(c_totals=c_totals, sd=sd, pseudomean=None, catlist=None):
    sa,mean = c_totals.std(),c_totals.mean() if pseudomean is None else pseudomean
    plt.bar(x=range(len(c_totals)),height=c_totals, alpha=0.4, color='k');
    plt.axhline(y=mean,c='r'); plt.axhline(y=mean+sd,c='k'); plt.axhline(y=mean-sd,c='k');
    if catlist is not None: plt.xticks(range(len(c_totals)), catlist, rotation=90)
    print(mean, (sd, sa))
    print(c_totals)