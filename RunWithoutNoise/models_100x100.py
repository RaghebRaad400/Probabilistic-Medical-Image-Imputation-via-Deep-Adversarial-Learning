from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from config_cect import argparser
from functools import partial

PARAMS = argparser()

conv = partial(slim.conv2d, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
dconv = partial(slim.conv2d_transpose, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
fc = partial(ops.flatten_fully_connected, activation_fn=None, weights_initializer=tf.random_normal_initializer(stddev=0.02))
relu = tf.nn.relu
lrelu = partial(ops.leak_relu, leak=0.2)
batch_norm = partial(slim.batch_norm, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None)
ln = slim.layer_norm
noise1 = np.load('std_normal_1_100_100_16.npy')


def generator(z, dim=16, reuse=True, training=True):
    bn = partial(batch_norm, is_training=training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
    conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None) #(add)
    fc_bn_relu = partial(fc, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

    with tf.variable_scope('generator', reuse=reuse):
        y = fc_bn_relu(z, 3 * 3 * dim * 14)
        y = tf.reshape(y, [-1, 3, 3, dim * 14])
        y = tf.image.resize_bilinear(y, [6,6])
        y = conv_bn_relu(y, dim*12, 3, 1)
        y = tf.image.resize_bilinear(y, [12,12])
        y = conv_bn_relu(y, dim*10, 3, 1)
        y = tf.image.resize_bilinear(y, [24,24])
        y = conv_bn_relu(y, dim*8, 3, 1)
        y = tf.image.resize_bilinear(y, [48,48])
        y = conv_bn_relu(y, dim*6, 3, 1)
        y = tf.image.resize_bilinear(y, [96,96])
        y = conv_bn_relu(y, dim*4, 2, 1)
        y = tf.image.resize_bilinear(y, [98,98])
        y = conv_bn_relu(y, dim, 2, 1)
        y = tf.image.resize_bilinear(y, [100,100])
       # if PARAMS.phase=='learning_prior':
        #    y = y + tf.tile(tf.constant(noise1, dtype=tf.float32)/tf.reduce_max(y), (PARAMS.train_batch_size, 1, 1, 1))  #comment lines 46 47 48 49to turn off noise
        #elif PARAMS.phase=='inference':
         #   y = y + tf.tile(tf.constant(noise1, dtype=tf.float32)/tf.reduce_max(y), (PARAMS.batch_size, 1, 1, 1))
        img = tf.tanh(tf.layers.conv2d(y, filters=4, kernel_size=3, padding='SAME',kernel_initializer=tf.initializers.glorot_normal()))                                      
        return img

def discriminator_wgan_gp(img, dim=16, reuse=True, training=True):
    conv_ln_lrelu = partial(conv, normalizer_fn=ln, activation_fn=lrelu, biases_initializer=None)

    with tf.variable_scope('discriminator', reuse=reuse):
        f0 = tf.reshape(img, [-1, img.shape[1]*img.shape[2], img.shape[3]])
        g0 = tf.matmul(tf.transpose(f0, (0,2,1)), f0)/(tf.cast(f0.shape[2]*f0.shape[1], tf.float32)) 
        
        y = lrelu(conv(img, dim, 3, 1))
        f1 = tf.reshape(y, [-1, y.shape[1]*y.shape[2], y.shape[3]])
        g1 = tf.matmul(tf.transpose(f1, (0,2,1)), f1)/(tf.cast(f1.shape[2]*f1.shape[1], tf.float32)) 
        
        y = conv_ln_lrelu(y, dim * 2, 3, 1, padding='VALID')
        f2 = tf.reshape(y, [-1, y.shape[1]*y.shape[2], y.shape[3]])
        g2 = tf.matmul(tf.transpose(f2, (0,2,1)), f2)/(tf.cast(f2.shape[2]*f2.shape[1], tf.float32))
        y = conv_ln_lrelu(y, dim * 4, 3, 1, padding='VALID')
        f3 = tf.reshape(y, [-1, y.shape[1]*y.shape[2], y.shape[3]])
        g3 = tf.matmul(tf.transpose(f3, (0,2,1)), f3)/(tf.cast(f3.shape[2]*f3.shape[1], tf.float32))
        y = conv_ln_lrelu(y, dim * 6, 3, 2)
        f4 = tf.reshape(y, [-1, y.shape[1]*y.shape[2], y.shape[3]])
        g4 = tf.matmul(tf.transpose(f4, (0,2,1)), f4)/(tf.cast(f4.shape[2]*f4.shape[1], tf.float32))
        y = conv_ln_lrelu(y, dim * 8, 3, 2)
        f5 = tf.reshape(y, [-1, y.shape[1]*y.shape[2], y.shape[3]])
        g5 = tf.matmul(tf.transpose(f5, (0,2,1)), f5)/(tf.cast(f5.shape[2]*f5.shape[1], tf.float32))
        y = conv_ln_lrelu(y, dim * 10, 3, 2)
        f6 = tf.reshape(y, [-1, y.shape[1]*y.shape[2], y.shape[3]])
        g6 = tf.matmul(tf.transpose(f6, (0,2,1)), f6)/(tf.cast(f6.shape[2]*f6.shape[1], tf.float32))
        y = conv_ln_lrelu(y, dim * 12, 2, 2)
        y = conv_ln_lrelu(y, dim * 14, 2, 2)
        logit = fc(y, 1)
        return logit, g0, g1, g2, g3, g4, g5, g6
