"""Implementation of masked self-attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from MusaNet.utils.model_utils import DenseActivation
from nn_utils.attentionLayers import exp_mask_for_high_rank


class AttentionPooling(keras.layers.Layer):
    def __init__(self, embedding_size, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.dense = DenseActivation(embedding_size, activation='relu')
        self.linear = DenseActivation(embedding_size)

    def call(self, inputs):
        tensor, mask = inputs
        x = self.dense(tensor)
        x = self.linear(x)
        x = exp_mask_for_high_rank(x, mask)

        soft = tf.nn.softmax(x, 1)  # bs,skip_window,vec
        attn_output = tf.reduce_sum(soft * tensor, 1)  # bs, vec

        return attn_output