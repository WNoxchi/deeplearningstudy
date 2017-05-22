# This is something of a dryrun for the code in the JNB. Code here is run in
# Atom using Hydrogen. Kernel: Python2.7

# FAI1 - Practical Deep Learning I: Week 3: Convolutions
# StateFarm Distracted Driver Kaggle Competition
# Wayne Nixalo - 2017-May-20 00:48

# NOTE: for reference:
#       On Keras predict_generator: https://keras.io/models/sequential/#predict_generator
#       Example successful JNB & using Pandas to write a submission:
#       https://github.com/philippbayer/cats_dogs_redux/blob/master/Statefarm.ipynb

# Imports
import keras
import os, sys
import numpy as np
import Pandas as pd
from glob import glob
from keras.optimizers import Adam
from keras.layers.core import Dense
from keras.preprocessing import image

# will need this to access any libraries in superdirectories
sys.path.insert(1, os.path.join(os.getcwd(), '../utils'))
import utils
from vgg16 import Vgg16

# #### 1. Run this the First Time Only
# Download the Data & get it into the right directories
# kaggle-cli needs to be set up beforehand

HOME_DIR    = os.getcwd()
DATA_DIR    = HOME_DIR + '/data'
TRAIN_DIR   = DATA_DIR + '/train'
VAL_DIR     = DATA_DIR + '/valid'
TEST_DIR    = DATA_DIR + '/test'

# create the validation directories
# os.mkdir(VAL_DIR)
# for i in xrange(10):
#     os.mkdir(VAL_DIR + '/c' + str(i))
# # another way to do this:
# %mkdir $VAL_DIR
# for i in xrange(10):
#     %mkdir $VAL_DIR/c"$i"

# #### 2. Run this if you don't have an Accurate Validation Set
# grab a random permutation from the training data for validation.
# do this until validation accuracy matches test accuracy
# also see: http://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python

# %cd $TRAIN_DIR
#
# VAL_PORTION = 0.2
# for i in xrange(10):
#     %cd c"$i"
#     g = glob('*.jpg')
#     number = len(g)
#     shuff = np.random.permutation(g)
#     for n in xrange(int(number * VAL_PORTION)):
#         os.rename(shuff[n], VAL_DIR + '/c' + str(i) + '/' + shuff[n])
#     % cd ..

# modified:
def reset_valid():
    """Moves all images in validation set back to
    their respective classes in the training set."""
    for i in xrange(10):
        %mv $VALID_DIR/c"$i"/*.jpg $TRAIN_DIR/c"$i"/*.jpg

# modified from: http://forums.fast.ai/t/statefarm-kaggle-comp/183/20
def set_valid(number=1):
    if number < 0: number = 0
    for n in xrange(number):
        # read CSV file into Pandas DataFrame
        dil = pd.read_csv(path + 'driver_imgs_list.csv')
        # group frame by subject in image
        grouped_subjects = dil.groupby('subject')
        # pick <number> subjects at random
        subject = groups.keys()[np.random.randint(0, high=len(groups))]
        # get the group assoc w/ subject
        group = grouped.get_group(subject)
        # loop over group & move imgs to validation dir
        for (subject, clssnm, img) in group.values:
            source = '{}train/{}/{}'.format(data_path, clssnm, img)
            target = source.replace('train', 'valid')
            print('mv {} {}'.format(source, target))
            os.rename(source, target)

# some more setup
data_path    = DATA_DIR  + '/'
train_path   = TRAIN_DIR + '/'
valid_path   = VAL_DIR   + '/'
test_path    = TEST_DIR  + '/'
results_path = DATA_DIR  + '/results/'

batch_size = 64
no_epochs = 3

# batch generator to feed data into the model
gen = image.ImageDataGenerator()
trn_batches = gen.flow_from_directory(train_path, target_size=(640,480),
                class_mode='categorical', shuffle=True, batch_size=batch_size)
val_batches = gen.flow_from_directory(valid_path, target_size=(640,480),
                class_mode='categorical', shuffle=False, batch_size=batch_size)

# load the VGG model, download its weights, and finetune it to the data
VGGModel = Vgg16()
VGGModel.pop()
for layer in VGGModel.layers: layer.trainable = False
VGGModel.add(Dense(10, activation='softmax'))
VGGModel.compile(Adam(), loss='cateogrical_crossentropy', metrics=['accuracy'])

# run the model until it overfits
VGGModel.optimizer.lr = 0.001
VGGModel.fit_generator(trn_batches, trn_batches.n, nb_epoch=1, verbose=1,
                       validation_data=val_batches, nb_val_samples=val_batches.n)
