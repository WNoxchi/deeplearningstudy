# Wayne H Nixalo - 2017-Jun-30 22:40
# FAI01 - Practical Deep Learning I - Lesson 7
# code from vgg16bn.py. See https//github.com/fastai/courses/nbs


# How to make a Convolutional Neural Network with variable-size input
# If the input size is off-default: do not add any Fully-Connected layers.

# (method of VGG16BN class)

def create(self, size, include_top):
    # this is the flag to add/not FC layers:
    if size != (224, 224):
        include_top = False

    model = self.model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(3,)+size, output_shape=(3,)+size))

    self.ConvBlock(2, 64)
    self.ConvBlock(2, 128)
    self.ConvBlock(2, 256)
    self.ConvBlock(2, 512)
    self.ConvBlock(2, 512)

    # if flag set to False: just create the Convolutional model and return
    if not include_top:
        fname = 'vgg16_bn_conv.h5'
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
        return

    # otherwise add the FC layers
    model.add(Flatten())
    self.FCBlock()
    self.FCBlock()
    model.add(Dense(1000, activation='softmax')

    fname = 'vgg16_bn.h5'
    model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
