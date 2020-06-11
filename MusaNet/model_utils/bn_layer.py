"""Implementation of masked self-attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from MusaNet.model_utils.nn_layer import Linear


# Batch normalization dense layer
class BatchNormalDenseLayer(keras.layers.Layer):
    def __init__(self,output_size,activation='relu',enable_bn=True,is_train=None, **kwargs):
        super(BatchNormalDenseLayer, self).__init__(**kwargs)
        self.is_train = is_train
        if self.is_train is None:
            self.is_train = False
        # activation
        if activation == 'linear':
            self.activation_func = tf.identity
        elif activation == 'relu':
            self.activation_func = tf.nn.relu
        elif activation == 'elu':
            self.activation_func = tf.nn.elu
        elif activation == 'sigmoid':
            self.activation_func = tf.nn.sigmoid
        elif activation == 'tanh':
            self.activation_func = tf.nn.tanh
        else:
            raise AttributeError('no activation function named as %s' % activation)
        self.enable_bn = enable_bn
        self.output_size = output_size
        self.linear = Linear(self.output_size)

    def call(self, inputs):
        x = self.linear(inputs)
        if self.enable_bn:
            x= keras.layers.LayerNormalization()(x)
        return self.activation_func(x)