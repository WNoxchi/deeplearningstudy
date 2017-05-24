# New 'clean-pass' of L3HW-SF ~ usin' lessons learned
# Wayne Nixalo - 2017-May-23 02:37
#
# useful links:
# DataAugmentation:
# https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson3.ipynb
# I forgot but reference anyway:
# https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson2.ipynb
# Good followthru of lecture & how to save to submission w/ Pandas:
# https://github.com/philippbayer/cats_dogs_redux/blob/master/Statefarm.ipynb
# Me:
# https://github.com/WNoxchi/Kaukasos/blob/master/FAI/lesson3/L3HW_SF.ipynb

import keras
import bcolz
import os, sys
import numpy as np
import pandas as pd
from glob import glob
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense
from keras.models import Sequential

sys.path.insert(1, os.path.join(os.getcwd(), '../utils'))
import utils
from vgg16bn import Vgg16BN

# directory setup

HOME_DIR  = os.getcwd()
DATA_DIR  = HOME_DIR + '/data'
TEST_DIR  = DATA_DIR + '/test'
TRAIN_DIR = DATA_DIR + '/train'
VALID_DIR = DATA_DIR + '/valid'

data_path    = DATA_DIR  + '/'
test_path    = TEST_DIR  + '/'
train_path   = TRAIN_DIR + '/'
valid_path   = VALID_DIR + '/'
results_path = DATA_DIR  + '/results/'

# utility functions

def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

def reset_valid(verbose=1):
    """Moves all images in validation set back to
    their respective classes in the training set."""
    counter = 0
    %cd $valid_path
    for i in xrange(10):
        %cd c"$i"
        g = glob('*.jpg')
        for n in xrange(len(g)):
            os.rename(g[n], TRAIN_DIR + '/c' + str(i) + '/' + g[n])
            counter += 1
        %cd ..
    if verbose: print("Moved {} files".format(counter))

# modified from: http://forums.fast.ai/t/statefarm-kaggle-comp/183/20
def set_valid(number=1, verbose=1):
    """Moves <number> subjects from training to validation
    directories. Verbosity: 0: Silent; 1: print no. files moved;
    2: print each move operation. Default=1"""
    counter = 0
    if number < 0: number = 0
    # repeat for <number> subjects
    for n in xrange(number):
        # read CSV file into Pandas DataFrame
        dil = pd.read_csv(data_path + 'driver_imgs_list.csv')
        # group frame by subject in image
        grouped_subjects = dil.groupby('subject')
        # pick subject at random
        subject = grouped_subjects.groups.keys()[np.random.randint(0, \
                                            high=len(grouped_subjects.groups))]
        # get group assoc w/ subject
        group = grouped_subjects.get_group(subject)
        # loop over group & move imgs to validation dir
        for (subject, clssnm, img) in group.values:
            source = '{}train/{}/{}'.format(data_path, clssnm, img)
            target = source.replace('train', 'valid')
            if verbose > 1: print('mv {} {}'.format(source, target))
            os.rename(source, target)
    if verbose: print("Files moved: {}".format(counter))

# function to build FCNet w/ BatchNormalization & Dropout
def create_FCbn_layers(p=0):
    return [
            MaxPooling2D(input_shape=Conv_model[-1].output_shape[1:]),
            Flatten(),
            BatchNormalization()
            Dense(4096, activation='relu')
            BatchNormalization()
            Dropout(p)
            Dense(10, activation='softmax')
            ]

# # creating validation directories
# os.mkdir(VAL_DIR)
# for i in xrange(10):
#     os.mkdir(VAL_DIR + '/c' + str(i))
#
# # another way to do this:
# %mkdir $VAL_PATH
# for i in xrange(10):
#     %mkdir $VAL_PATH/c"$i"

# setting/resetting validation set
reset_valid()
set_valid(number=3)

# parameters
batch_size = 32
target_size = (224, 224)

# train/valid batch generators

gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.05,
        height_shift_range=0.05, width_zoom_range=0.1, zoom_range=0.1,
        shear_range=0.1, channel_shift_range=10)
# does it matter that I don't set dim_ordering='tf' ?

trn_batches = gen.flow_from_directory(train_path, target_size=target_size,
                batch_size=batch_size, shuffle=True, class_mode='categorical')
val_batches = gen.flow_from_directory(valid_path, target_size=target_size,
                batch_size=batch_size, shuffle=False, class_mode='categorical')

# load VGG16BN model & its weights
VGGbn = Vgg16BN()
VGGbn.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# (maybe) train the model at low η to train the Conv layers a bit
VGGbn.fit_generator(trn_batches, trn_batches.n, nb_epoch=1,
                    validation_data=val_batches, nb_val_samples=val_batches.n)
# find out how many epochs at what η to do this until it's ~optimal

# separate Conv layers & create new ConvNet (w/ vgg weights)
last_conv_idx = [index, for index, layer in enumerate(VGGbn.model.layers) \
                                            if type(layer) is Convolution2D][-1]
Conv_layers = VGGbn.model.layers[:last_conv_idx + 1]

# create new ConvNet from VGG16BN conv layers
Conv_model = Sequential(Conv_layers)

# now set training batches to not be shuffled. This is critical, because
# classes & labels will be supplied to the FCNet via directory; otherwise
# it won't know what's what. This doesn't need to be done if the whole model
# is left as one whole, but when using output features of one model as inputs
# to another, there has to be some way of keeping track of the labels.
# Remember gen is set to dataaugmentation. Reset this when predicting test set.
trn_batches = gen.flow_from_directory(train_path, target_size=target_size,
                batch_size=batch_size, shuffle=False, class_mode='categorical')

# run Conv Model on trn/val batches to create features as inputs to FCNet
conv_features = Conv_model.predict_generator(trn_batches, trn_batches.nb_sample)
conv_val_feat = Conv_model.predict_generator(val_batches, val_batches.nb_sample)
# (?) does it matter than trn_batches is shuffled? nb_sample vs n?
# batches.n in fit() and batches.nb_sample in predict() ?

# save the convolution model's output features
save_array(results_path + 'conv_features.dat', conv_features)
save_array(results_path + 'conv_val_feat.dat', conv_val_feat)

# create FCNet
FC_model = Sequential(create_FCbn_layers(p=0.3))
FC_model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# train FCNet on the ConvNet features
# is there a way to do this as a generator -- or does it not matter?
# maybe save the features, then pull them from disk in batches?
FC_model.fit(conv_features, trn_batches.labels, batch_size=batch_size,
                nb_epoch=1, validation_data=(conv_val_feat, val_batches.labels))



# non-augmented batch generator for test-data
gen = image.ImageDataGenerator()
tst_batches = gen.flow_from_directory(test_path, batch_size=batch_size,
                shuffle=False, class_mode=None)
# vgg16gn.test() <---> model.predict_generator(tst_batches, tst_batches.nb_sample)
# run test batches through ConvNet
conv_tst_feat = Conv_model.predict_generator(tst_batches, tst_batches.nb_sample)

# run ConvNet test features through FCNet
preds = FC_model.predict(conv_tst_feat, batch_size=batch_size*2)


# Ensemble the above, save models in array, average predictions
# NOTE: the Conv layers are probably not going to learn much after being trained
#       on 1.5M imagenet photos... and there isn't yet a clean way to clear gpu
#       memory in JNBs, each time a VGG model is instantiated ~700MB are loaded
#       into GPU memory.. so to work around for ensembling: I'll initialize a
#       VGG model once & train it's Conv layers, then do a 'hybrid-ensemble' w/
#       the FC Nets.
# NOTE: On further thought, it would've been smarter to do the ConvNet features
#       outside the function, and just save the features and pass them in.
def hybrid_ensemble(number=1):
    batch_size=32
    target_size=(224, 224)

    reset_valid()
    set_valid(number=3)

    gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.05,
            height_shift_range=0.05, width_zoom_range=0.1, zoom_range=0.1,
            shear_range=0.1, channel_shift_range=10)
    trn_batches = gen.flow_from_directory(train_path, target_size=target_size,
                batch_size=batch_size, shuffle=True, class_mode='categorical')
    val_batches = gen.flow_from_directory(valid_path, target_size=target_size,
                batch_size=batch_size, shuffle=False, class_mode='categorical')

    VGGbn = Vgg16BN()
    VGGbn.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    VGGbn.fit_generator(trn_batches, trn_batches.n, nb_epoch=1,
                    validation_data=val_batches, nb_val_samples=val_batches.n)

    last_conv_idx = [index, for index, layer in enumerate(VGGbn.model.layers) \
                                            if type(layer) is Convolution2D][-1]
    Conv_layers = VGGbn.model.layers[:last_conv_idx + 1]
    Conv_model = Sequential(Conv_layers)

    trn_batches = gen.flow_from_directory(train_path, target_size=target_size,
                batch_size=batch_size, shuffle=False, class_mode='categorical')

    conv_features = Conv_model.predict_generator(trn_batches,
                                                        trn_batches.nb_sample)
    conv_val_feat = Conv_model.predict_generator(val_batches,
                                                        val_batches.nb_sample)
    predarray = []
    for n in xrange(number):
        reset_valid()
        set_valid(number=3)

        FC_model = Sequential(create_FCbn_layers(p=0.3))
        FC_model.compile(Adam(), loss='categorical_crossentropy',
                                                        metrics=['accuracy'])
        FC_model.fit(conv_features, trn_batches.labels, batch_size=batch_size,
                nb_epoch=1, validation_data=(conv_val_feat, val_batches.labels))

        gen_t = image.ImageDataGenerator()
        tst_batches = gen_t.flow_from_directory(test_path,
                        batch_size=batch_size, shuffle=False, class_mode=None)
        conv_tst_feat = Conv_model.predict_generator(tst_batches,
                                                        tst_batches.nb_sample)
        preds = FC_model.predict(conv_tst_feat, batch_size=batch_size*2)
        predarray.append(preds)



    # NOTE: I could probably save more memory by loading a tabula-rasa FCNet
    #       from disk, instead of just defining a new one each iteration.

# record submission
