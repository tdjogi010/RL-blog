import tensorflow as tf
import numpy as np


#common components
fc = tf.layers.dense #fully connected network
relu = tf.nn.relu
tanh = tf.tanh
conv2d = tf.layers.conv2d
xavier_init = tf.contrib.layers.xavier_initializer # same as tf.contrib.layers.xavier_initializer_conv2d

def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x

#obs is 4D vector (batch, height, breadth, channel/image)
def get_mlp_network(obs):
    # flatten
    preprocessed = tf.layers.flatten(obs) # preserves the batch axis

    # network
    h1 = fc(preprocessed, 64, activation=tanh, use_bias=True, name='h1')
    h2 = fc(h1, 64, activation=tanh, use_bias=True, name='h2')
    return h2 # will be fed to get pi(s,a) and vf(s)

#obs is 4D vector, (batch, height, breadth, channel/image)
def get_cnn_network(obs):
    # preprocess
    preprocessed = tf.cast(obs, tf.float32) / 255
    
    # network
    h = conv2d(preprocessed, filters=32, kernel_size=[8,8], strides=4, padding='valid',activation=relu, name='h1', kernel_initializer=xavier_init()) 
    h2 = conv2d(h, 64, [4,4], 2, activation=relu, name='h2', kernel_initializer=xavier_init())
    h3 = conv2d(h2, 64, [3,3], 1, activation=relu, name='h3', kernel_initializer=xavier_init())
    h3 = conv_to_fc(h)
    return fc(h3, 512, activation=None, kernel_initializer=xavier_init()) # will be fed to get pi(s,a) and vf(s)
