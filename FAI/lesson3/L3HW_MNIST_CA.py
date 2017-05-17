# WNX - 2017-May-15 16:11 - FAI1 - Practical Deep Learning I - 2017-May-16 00:57
# Lesson 3 Homework - MNIST CodeAlong / Notes

# NOTE: when working in Keras 2 later on, be sure to refer to the release notes!
#       a number of common parameters have been renamed.
#       https://github.com/fchollet/keras/wiki/Keras-2.0-release-notes

# This is in Keras 1.2.2, Theano 0.9.0 backend, Python 2.7

import keras
import numpy as np

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# will use test as validation set
# Keras will expect a dimension for color channel. Just black/white so just 1
np.expand_dims(x_train, axis=1) # or : np.expand_dims(x_train, 1)
np.expand_dims(x_test, axis=1)

# Keras wants OneHot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Now we are ready to build our models

# need to normalize the input: subtract mean, divide by stdev.
x_mean = x_train.mean().astype(np.float32)
x_stdv = x_train.std().astype(np.float32)
def norm_input(x): return (x - x_mean) / x_stdv

# Keras.layers.core.Lambda(function, ouput_shape=None, mask=None, arguments=None)
# Keras.layers.core.Dense(units, activation=None, use_bias=True,
#       kernel_initializer='glorot_uniform', bias_initializer='zeros',
#       kernel_regularizer=None, bias_regularizer=None,
#       activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# Keras.models.Sequential.compile(optimizer, loss, metrics)

# all at once:
# Keras.models.Sequential.fit(self, x, y, batch_size=32, epochs=10,
#       verbose=1, callbacks=None, validation_split=0.0, validation_data=None,
#       shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

# in batches:
# Keras.models.Sequential.fit_generator(self, generator, steps_per_epoch,
#       epochs=1, verbose=1, callbacks=None, validation_data=None,
#       validation_steps=None, class_weight=None, max_q_size=10, workers=1,
#       pickle_safe=False, intitial_epoch=0)

# Keras.models.Sequential.compile(self, optimizer, loss, metrics=None,
#       sample_weight_mode=None)

# 1. Linear Model   #### #### #### #### #### #### #### #### #### #### #### ####
def get_lin_model():
    model = Sequential()
    model.add(Lambda(norm_input, input_shape = (1, 28, 28))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# version in Lecture 3:
def get_lin_model():
    model = Sequential([
        # normalize the input
        Lambda(norm_input, input_shape=(1,28,28)),
        # flatten it
        Flatten(),
        # create 1 dense layer w/ 10 ouputs
        Dense(10, activation='softmax'),
        ])
    # compile it
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

L_model = get_lin_model()

# with data fully-loaded in memory:
# model.fit(x_train, y_train, batch_size=32, nb_epoch=4, validation_data=(x_test, y_test))

# with data read from disk in batches:
gen = image.ImageDataGenerator()
trn_batches = gen.flow(x_train, y_train, batch_size=64)
tst_batches = gen.flow(x_test, y_test, batch_size=64)

# train the Linear Model:

# 1. generally the  best way to train a model is to start by doing 1 epoch w/ a
# low learning rate (default η = 0.001), to get it started.
# 2. Once it's gotten started, can set η up high (max: 0.1), and do 1 more
# epoch which will move super fast.
# 3. Then reduce η by an order of magnitude (factor of 10) each bunch of epochs.
# 4. And keep going until you start overfitting.

L_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=1,
        validation_data=trn_batches, nb_val_samples=trn_batches.n)

L_model.optimizer.lr=0.1
L_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=4,
        validation_data=trn_batches, nb_val_samples=trn_batches.n)


# NOTE: JH got acc/valacc:0.9267/0.9240. That's about as far as Linear Models
#       can go, so next we do a NN w/ a single hidden layer -- this is what
#       people in the 80s/90s thought of as a Neural Network: a single
#       Fully-Connected hidden layer.    .. lol.


# 2. Single Dense (FC) Layer  #### #### #### #### #### #### #### #### #### ####
def get_fc_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1, 28, 28)),
        Flatten(),
        # 1 hidden layer, FC
        Dense(512, activation='softmax'),
        Dense(10, activation='softmax')
        ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Then same thing as before: 1 epoch low η, then 1 epoch max η, then starting
# at high η, a few epochs at a time, lowering η until we start to overfit.
FC_model = get_fc_model()
FC_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=1,
        validation_data=tst_batches, nb_val_samples=tst_batches.n)

# now jack up the learning rate for an epoch (or 4... as long as we can)
FC_model.optimizer.lr=0.1
FC_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=4,
        validation_data=tst_batches, nb_val_samples=tst_batches.n)

# and back down for a few more
FC_model.optimizer.lr=0.01
FC_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=4,
        validation_data=tst_batches, nb_val_samples=tst_batches.n)

# NOTE: at this point, JH got acc/valacc:0.9474/0.9397. You wouldn't expect a
#       FC network to do that well, so let's create a CNN.

# let's create an architecture that looks like VGG, but is much simpler, since
# we're dealing w/ 28x28 grayscale images.

# 3. Basic 'VGG-style' CNN    #### #### #### #### #### #### #### #### #### ####

# I think Convolution2D has been renamed to Conv2D
# keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1,1),
#       padding='valid', data_format=None, dilation_rate=(1,1), activation=None,
#       use_bias=True, kernel_initializer='glorot_uniform',
#       bias_initializer='zeros', kernel_regularizer=None,
#       bias_regularizer=None, activity_regularizer=None,
#       kernel_constraint=None, bias_constraint=None)

# NOTE: aha!        - must be Keras 1.2.2 vs Keras 2
# Convolution2D(self, nb_filter, nb_row, nb_col, init='glorot_uniform',
#       activation=None, weights=None, border_mode='valid', subsample=(1,1),
#       dim_ordering='default', W_regularizer=None, b_regularizer=None,
#       activity_regularizer=None, W_constraint=None, b_constraint=None,
#       bias=True, **kwargs)

# keras.layers.pooling.MaxPooling2D(pool_size=(2,2), strides=None,
#       padding='valid, data_format=None)

def get_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1, 28, 28)),
        # Flatten(),
        # VGG typically has a couple CNN layers of 3x3
        Convolution2D(32, 3, 3, activation='relu'),
        Convolution2D(32, 3, 3, activation='relu'),
        # then a MaxPooling layer
        MaxPooling2D(),
        # & then a couple more CNNs w/ twice as many filters
        Convolution2D(64, 3, 3, activation='relu'),
        Convolution2D(64, 3, 3, activaiton='relu'),
        MaxPooling2D(),
        Flatten(),
        # Finally adding the 2 dense (FC) layers
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# NOTE: JH's logic w/ above: after 2 lots of MaxPooling, img-size will go from
#       28x28 to 14x14 to 7x7, and that's probably enough. (filter size? img?)
#       Above written just by intuition by prof.

# Same training pattern as before:

CNN_model = get_model()
CNN_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=1,
        validation_data = tst_batches, nb_val_samples = tst_batches.n)

CNN_model.optimizer.lr=0.1
CNN_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=1,
        validation_data = tst_batches, nb_val_samples = tst_batches.n)

CNN_model.optimizer.lr=0.01
CNN_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=8,
        validation_data = tst_batches, nb_val_samples = tst_batches.n)

# NOTE: at this point JH had an acc/valacc:0.9975/0.9926
#       at this point the model is overfitting. acc > valacc
#       Once you're overfitting, you know you have a model that's complex
#       enough to handle your data.

# So here's an architecture capable of overfitting. Let's now try to use the
# same Architecture and reduce overfitting, but reduce complexity of the model
# no more than necessary.

# 4. Data Augmentation    #### #### #### #### #### #### #### #### #### #### ####
CNN_model = get_model()

# keras.preprocessing.image.ImageDataGenerator(self, featurewise_center=False,
#       samplewise_center=False, featurewise_std_normalization=False,
#       samplewise_std_normalization=False, zca_whitening=False,
#       rotation_range=0.0, channel_shift_range=0.0, fill_mode='nearest',
#       cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None,
#       preprocessing_function=None, dim_ordering='default')

# add a bit of data augmentation
gen = image.ImageDataGenerator(rotation_range=8, shift_range=0.08,
        height_shift_range=0.08, zoom_range=0.08)

trn_batches = gen.flow(x_train, y_train, batch_size=64)
tst_batches = gen.flow(x_test, y_test, batch_size=64)

# now using exactly the same CNN model as before:
CNN.model.fit_generator(trn_batches, trn_batches.n, nb_epoch=1,
        validation_data=tst_batches, nb_val_samples=tst_batches.n)

CNN_model.optimizer.lr=0.1
CNN.model.fit_generator(trn_batches, trn_batches.n, nb_epoch=1,
        validation_data=tst_batches, nb_val_samples=tst_batches.n)

CNN_model.optimizer.lr=0.01
CNN_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=8,
        validation_data=tst_batches, nb_val_samples=tst_batches.n)

# JH acc/valacc:0.9925/0.9920

CNN_model.optimizer.lr=0.001
CNN_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=12,
        validation_data=tst_batches, nb_val_samples=tst_batches.n)

# NOTE: at this point in lecture, JH got acc/valacc:0.9961/0.9911

# Data Augmentation alone is not enough. And we're always going to use Batch-
# Norm anyway.

# 5. BatchNormalization + Data Augmentation  #### #### #### #### #### #### ####

# Keras 2?
# keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99,
#       epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
#       gamma_initializer='ones', moving_mean_initializer='zeros',
#       moving_variance_initializer='ones', beta_regularizer=None,
#       gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)

# Keras 1.2.2 init signature:
# ...BatchNormalization(self, epsilon=0.001, mode=0, axis=-1, momentum=0.99,
#       weights=None, beta_init='zero', gamma_init='one',
#       gamma_regularizer=None, beta_regularizer=None, **kwargs)

# axis: integer, axis along which to normalize in mode 0. For instance,
#   if your input tensor has shape (samples, channels, rows, cols),
#   set axis to 1 to normalzie per feature map (channels axis).

# mode: integer, 0, 1, or 2.
#   - 0: feature-wise normalization.
#       Each feature map in the input will be normalized seperately.
#       The axis on which to normalize is specified by the `axis` argument.
#       Note that if the input is a 4D image tensor using Theano conventions
#       (samples, channels, rows, cols) then you should set `axis` to `1` to
#       normalize along the channels axis.
#       During training we use per-batch statistics to normalize the data,
#       and during testing we use runnign averages computed during the
#       training phase.
#   - 1: sample-wise normalization. This mode assumes a 2D input.
#   - 2: feature-wise normalization, like mode 0, but using per-batch
#       statistics to normalize the data during both testing and training.

# NOTE: you want the batchnorm AFTER the nonlinearity and BEFORE the dropout

def get_model_bn():
    model = Sequential([
        # normalize input data
        Lambda(norm_input, input_shape=(1, 28, 28)),
        Convolution2D(32, 3, 3, activation='relu'),
        # when BN used on Conv layers: must add <axis=1>
        BatchNormalization(axis=1),
        Convolution2D(32, 3, 3, activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        MaxPooling2D(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax'),
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# NOTE: BatchNorm is used on every layer
#       use axis=1 for 2D Conv Layers w/ dim_ordering='th'
#       use axis=-1 for 2D Conv Layers w/ dim_ordering='tf'
#       use axis=-1 for FC, Rec, & most other layers
# http://forums.fast.ai/t/batchnormalization-axis-1-when-used-on-convolutional-layers/214/3

CNN_model = get_model_bn()
CNN_model.fit_gnerator(trn_batches, trn_batches.n, nb_epoch=1,
        validation_data=tst_batches, nb_val_samples=tst_batches.n)

CNN_model.optimizer.lr=0.1
CNN_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=4,
        validation_data=tst_batches, nb_val_samples=tst_batches.n)

CNN_model.optimizer.lr=0.01
CNN_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=12,
        validation_data=tst_batches, nb_val_samples=tst_batches.n)

CNN_model.optimizer.lr=0.001
CNN_model.fit_generator(trn_batches, trn_batches.n, nb_epoch=12,
        validation_data=tst_batches, nb_val_samples=tst_batches.n)

# NOTE: at this point, JH got acc/val: 0.9964/0.9945 at Epoch 12,
#       0.9954/0.9949 at Epoch 6.

# Still starting to overfit by the end. So we add a little Dropout.

# Keras 1.2.2 Dropout:
# keras.layers.core.Dropout(self, p, noise_shape=None, seed=None, **kwargs)
# Keras 2 Dropout:
# keras.layers.core.Dropout(rate, noise_shape=None, seed=None)

# <rate> and <p> seem to be the same thing: float between 0 & 1.
# ~~ actually looking at the source code, the initializers are exactly the same

# 6. BatchNormalization + DataAugmentation + Dropout   #### #### #### #### ####
# The general rule is to gradually increase Dropout. In Lecture, JH added only
# one layer of p=0.5 at the end:
def get_model_bn_do():
    model = Sequential([
        # normalize input data
        Lambda(norm_input, input_shape=(1, 28, 28)),
        Convolution2D(32, 3, 3, activation='relu'),
        # when BN used on Conv layers: must add <axis=1>
        BatchNormalization(axis=1),
        Convolution2D(32, 3, 3, activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64, 3, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax'),
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

CNN_model = get_model_bn_do()
# CNN_model.fit_generator(...)
# CNN_model.optimizer.lr=0.1
# ... same as above

# 1 epoch at default η
# 4 epochs at η = 0.1
# 12 epochs at η = 0.01
# 12 epochs at η = 0.001

# NOTE: at this point, JH got acc/val of 0.9944/0.9948; 0.9943/0.9956 epoch 10

# Now, there is one more trick you can do which makes every model better.

# 7. Ensembling     #### #### #### #### #### #### #### #### #### #### #### ####
# basically take all the code from the last section and put it into 1 function
def fit_model():
    model = get_model_bn_do()
    model.fit_generator(trn_batches, trn_batches.n, nb_epoch=1, verbose=0,
            validation_data=tst_batches, nb_val_samples=tst_batches.n)
    model.optimizer.lr=0.1
    model.fit_generator(trn_batches, trn_batches.n, nb_epoch=4, verbose=0,
            validation_data=tst_batches, nb_val_samples=tst_batches.n)
    model.optimizer.lr=0.01
    model.fit_generator(trn_batches, trn_batches.n, nb_epoch=12, verbose=0,
            validation_data=tst_batches, nb_val_samples=tst_batches.n)
    model.optimizer.lr=0.001
    model.fit_generator(trn_batches, trn_batches.n, nb_epoch=12, verbose=0,
            validation_data=tst_batches, nb_val_samples=tst_batches.n)
    return model

# so now, say, 6 times, fit a model, and make an array of them:
models = [fit_model() for i in xrange(6)]
# <models> will be an array of 6 trained models of our preferred network

# Save the models
path = "data/mnist/"
model_path = path + 'models/'
for i,m in enumerate(models):
    # I think you'll need utils.py for this:
    m.save_weights(model_path + 'cnn-mnist-' + str(i) + '.pkl')

# I think JH did this to see what the outputs were:
# evals = np.array([m.evaluate(x_test, y_test, batch_size=256) for m in models)
# evals.mean(axis=0)

# go through all 6 models, and predict the output in our test-set
all_preds = np.stack([m.predict(x_test, batch_size=256) for m in models])

# all_preds.shape
# returns: (6, 10000, 10); 6 models x 10000 imgs x 10 outputs

# Now we can take the average across the 6 models
avg_preds = all_preds.mean(axis=0)

# basically saying: here are 6 models, they've been trained in the same way
# from 6 different random starting points; & hopefully they'll have errors in
# different places.

# now we can check the average against the test labels:
keras.metrics.categorical_accuracy(y_test, avg_preds).eval()
# returns: array(0.996999979.., dtype=float32)
# NOTE: here JH got an accuracy of 0.996999979 ~ 99.7%
#       that's just behind the 5th best academic result for MNIST.













################################################################################
# def get_fc_model():
#     model = Sequential()
#     model.add(Lambda(norm_input, input_shape = (1, 28, 28))
#     model.add(Flatten())
#     model.add(Dense(10, activation='softmax'))
#     model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
#     return model


# older method used in Lecture 3:
# opt = RMSprop(lr=0.00001, rho=0.7)
# def get_fc_model():
#     model = Sequential([
#         Dense(4096, activation='relu', input_dim=conv_layers[-1].output_shape[1]),
#         Dropout(0.),
#         Dense(4096, activation='relu'),
#         Dropout(0.),
#         Dense(2, activation='softmax')
#         ])
#
#     for λ1, λ2 in zip(model.layers, fc_layers): λ1.set_weights(proc_wgts(λ2))
#
#     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
