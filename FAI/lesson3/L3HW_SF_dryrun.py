# This is something of a dryrun for the code in the JNB. Code here is run in
# Atom using Hydrogen. Kernel: Python2.7

# FAI1 - Practical Deep Learning I: Week 3: Convolutions
# StateFarm Distracted Driver Kaggle Competition
# Wayne Nixalo - 2017-May-20 00:48

# Imports
import os, sys
import numpy as np
from glob import glob

# will need this to access any libraries in superdirectories
sys.path.insert(1, os.path.join(DATA_PATH, '../../utils'))
from vgg16 import Vgg16

# #### 1. Run this the First Time Only
# Download the Data & get it into the right directories
# kaggle-cli needs to be set up beforehand

HOME_DIR = os.getcwd()
DATA_PATH = HOME_DIR + '/data'
TRAIN_PATH = DATA_PATH + '/train'
VAL_PATH = DATA_PATH + '/valid'
TEST_PATH = DATA_PATH + '/test'

# create the validation directories
os.mkdir(VAL_PATH)
for i in xrange(10):
    os.mkdir(VAL_PATH + '/c' + str(i))
# # another way to do this:
# %mkdir $VAL_PATH
# for i in xrange(10):
#     %mkdir $VAL_PATH/c"$i"

# #### 2. Run this if you don't have an Accurate Validation Set
# grab a random permutation from the training data for validation.
# do this until validation accuracy matches test accuracy
# also see: http://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python

%cd $DATA_PATH

VAL_PORTION = 0.2
for i in xrange(10):
    %cd c"$i"
    g = glob('*.jpg')
    shuff = np.random.permutation(g)
    for n in xrange(number * VAL_PORTION):
        os.rename(shuff[n], VAL_PATH + shuff[n])
    %cd ..


# load the VGG model & download its weights
VGGModel = Vgg16()
