# Just a test of getting GPU working in TensorFlow
# WNixalo 26 AUG 2017

import keras.backend as K
import tensorflow as tf

# from: https://www.tensorflow.org/tutorials/using_gpu

# Creates a graph
with tf.device('/gpu:0'):
	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2,3], name='a')
	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3,2], name='b')
c = tf.matmul(a,b)

# Creates a session with log_device_placement set to True
sess = tf.Session(config=tf.ConfigProto(
	allow_soft_placement=True, log_device_placement=True))

# Runs the op
print(sess.run(c))
