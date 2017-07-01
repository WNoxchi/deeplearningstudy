# Wayne H Nixalo - 2017-Jul-01 00:26
# FAI01 - Practical Deep Learning I - Lesson 7
# code from vgg16bn.py. See https//github.com/fastai/courses/nbs

# from J.Howard -- "world's tiniest Inception Network" 

def incep_block(x):
    branch1x1 = conv2d_bn(x, 32, 1, 1, subsample=(2, 2))    # do 1x1 Conv
    branch5x5 = conv2d_bn(x, 24, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 32, 5, 5, subsample=(2, 2))    # do 5x5 Conv

    branch3x3dbl = conv2d_bn(x, 32, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 48, 3, 3)    # do 2 3x3 Convs|
    branch3x3dbl = conv2d_bn(branch3x3dbl, 48, 3, 3, subsample=(2, 2))  #   |

    branch_pool = AveragePooling2D(                     # Avg Pool the input
        (3, 3), strides=(2,2), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 16, 1, 1)
    return merge([branch1x1, branch5x5, branch3x3dbl, branch_pool], # concat all
                mode='concat', concat_axis=1)

inp = Input(vgg640.layers[-1].output_shape[1:])
x = BatchNormalization
x = incep_block(x)
x = incep_block(x)
x = incep_block(x)
x = Dropout(0.75)(x)
x = Convolution2D(8,3,3, border_mode='same')(x)
x = GlobalAveragePooling2D()(x)
outp = Activation('softmax')(x)
