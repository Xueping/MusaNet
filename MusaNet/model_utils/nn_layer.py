"""Implementation of masked self-attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from functools import reduce
from operator import mul


VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


@tf.function
def scaled_tanh(x, scale=5.):
    return scale * tf.nn.tanh(1. / scale * x)


@tf.function
def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER,
                  name=name or 'exp_mask_for_high_rank')


@tf.function
def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32), name=name or 'mask_for_high_rank')


class Reconstruct(keras.layers.Layer):
    def __init__(self, keep):
        super(Reconstruct, self).__init__()
        self.keep = keep

    def call(self, inputs):
        tensor, ref = inputs[0],inputs[1]
        dim_reduced_keep = self.keep
        ref_shape = ref.get_shape().as_list()  # original shape
        tensor_shape = tensor.get_shape().as_list()  # current shape
        ref_stop = len(ref_shape) - self.keep  # flatten dims list
        tensor_start = len(tensor_shape) - dim_reduced_keep  # start
        pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
        keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
        target_shape = pre_shape + keep_shape
        out = tf.reshape(tensor, target_shape)
        return out


# reshape the original tensor to (mul of other axises,  last keep dims)
class Flatten(keras.layers.Layer):
    def __init__(self, keep, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.keep = keep

    def call(self, inputs):
        fixed_shape = inputs.get_shape().as_list()
        # print('shapeo o f inputs ', fixed_shape, self.keep)
        start = len(fixed_shape) - self.keep
        left = reduce(mul, [fixed_shape[i] or tf.shape(inputs)[i] for i in range(start)])
        out_shape = [left] + [fixed_shape[i] or tf.shape(inputs)[i] for i in range(start, len(fixed_shape))]
        flat = tf.reshape(inputs, out_shape)
        # print('shapeo o f outputs ', flat.get_shape().as_list())
        return flat


class Linear(keras.layers.Layer):
    def __init__(self,output_size, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.output_size = output_size
        self.flatten = Flatten(1)
        self.linear = keras.layers.Dense(output_size, activation='linear')
        self.reconstruct = Reconstruct(1)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.linear(x)
        x = self.reconstruct([x,inputs])
        return x