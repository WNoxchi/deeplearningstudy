# WNX - 2017-May-10 21:00

from Vgg16 import vgg16
from keras.models import Sequential

model = vgg16()

model.pop()

LM = Sequential([Dense(Num_Outputs, input_shape=(Num_Inputs,))])
