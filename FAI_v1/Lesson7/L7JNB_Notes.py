# 2017-Jul-05 02:11
#
# Notes for L7JNB - Linux / full dataset
#
# relevant: pulling data from disk in batches:
# https://github.com/WNoxchi/Kaukasos/blob/master/FAI/conv_test_Asus.ipynb
#
# JNB:
# https://github.com/WNoxchi/Kaukasos/blob/master/FAI/Lesson7/L7JNB_CodeAlong_ULINUX.ipynb

# Imports:
import theano
%matplotlib inline
import sys, os
sys.path.insert(1, os.path.join('../utils'))
import utils; reload(utils)
from utils import *
from __future__ import print_function, division

# Setup
path = "data/fisheries/"
batch_size = 32

# be in FAI root folder
%cd ../..

batches = get_batches(path + 'train', batch_size=batch_size)
val_batches = get_batches(path + 'valid',
