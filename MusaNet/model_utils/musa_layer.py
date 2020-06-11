"""Implementation of masked self-attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import math_ops
from MusaNet.model_utils.normalization_layer import LayerNormalization
from MusaNet.model_utils.nn_layer import exp_mask_for_high_rank, mask_for_high_rank, Linear
from MusaNet.model_utils.bn_layer import BatchNormalDenseLayer


# The second level self-attention
class DirectionalVisitSelfAttnResnetLayer(keras.layers.Layer):
    def __init__(self, direction, train, dropout):
        self.direction = direction
        self.train = train
        self.dropout = dropout
        super(DirectionalVisitSelfAttnResnetLayer, self).__init__()

    def build(self, input_shape):
        self.batch_size  = input_shape[0][0]
        self.skip_window = input_shape[0][1]
        self.embedding_size = input_shape[0][2]
        self.dense =  BatchNormalDenseLayer(self.embedding_size, 'relu', False, None)
        self.linear = Linear(self.embedding_size)
        self.layer_norm = LayerNormalization()

    def call(self, inputs):
        tensor, mask = inputs

        sw_indices = tf.range(self.skip_window, dtype=tf.int32)
        sw_col, sw_row = tf.meshgrid(sw_indices, sw_indices)
        if self.direction is None:
            # shape of (skip_window, skip_window)
            direct_mask = tf.cast(tf.linalg.diag(- tf.ones([self.skip_window], tf.int32)) + 1, tf.bool)
        else:
            if self.direction == 'forward':
                direct_mask = tf.greater(sw_row, sw_col)  # shape of (skip_window, skip_window)
            else:
                direct_mask = tf.greater(sw_col, sw_row)  # shape of (skip_window, skip_window)

        # non-linear for context
        rep_map = self.dense(tensor)
        rep_map_tile = keras.layers.Reshape((1, self.skip_window, self.embedding_size))(rep_map)
        rep_map_tile = tf.tile(rep_map_tile, [1, self.skip_window, 1, 1]) # bs,sl,sl,vec

        dependent = self.linear(rep_map)
        # batch_size,1,sw,vec_size
        dependent_etd = keras.layers.Reshape((1,self.skip_window,self.embedding_size))(dependent)

        head = self.linear(rep_map)
        # batch_size,sw,1,vec_size
        head_etd = keras.layers.Reshape((self.skip_window,1,self.embedding_size))(head)

        # batch_size,sw,sw,vec_size
        attention_fact = keras.layers.Add()([dependent_etd, head_etd])
        bias_1 = self.add_weight(shape=(self.embedding_size,),
                                 initializer='zeros', dtype='float32',
                                 trainable=True)
        attention_fact = tf.nn.bias_add(attention_fact, bias_1)

        attention_fact = keras.layers.Activation('tanh')(attention_fact)
        attention_fact = self.linear(attention_fact)
        bias_2 = self.add_weight(shape=(self.embedding_size,),
                                 initializer='zeros', dtype='float32',
                                 trainable=True)
        attention_fact = tf.nn.bias_add(attention_fact, bias_2)

        logits_masked = exp_mask_for_high_rank(attention_fact, direct_mask)
        attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sw,sw,vec_size
        attn_score = mask_for_high_rank(attn_score, direct_mask)

        # Dropouts
        if self.train:
            attn_score = tf.nn.dropout(attn_score, rate=self.dropout)

        attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec_size
        attn_result = mask_for_high_rank(attn_result, mask)


        # fusion gate
        # gate_add = keras.layers.Add()([attn_result, rep_map])
        # bias_2 = self.add_weight(shape=(self.embedding_size,),
        #                          initializer='zeros', dtype='float32',
        #                          trainable=True)
        # gate_add = tf.nn.bias_add(gate_add, bias_2)
        # gate_add = keras.layers.Activation('sigmoid')(gate_add)
        # # input gate
        # output = gate_add * rep_map + (1 - gate_add) * attn_result

        # resnet
        output = keras.layers.Add()([attn_result, rep_map])
        # layer normalization
        output = self.layer_norm(output)

        return output

# the first level self-attention with ICDM19 layer
class FirstLevelSA(keras.layers.Layer):
    def __init__(self, vocabulary_size, embedding_size, activation, task_type, **kwargs):
        super(FirstLevelSA, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.task_type = task_type
        self.embedding = keras.layers.Embedding(vocabulary_size, embedding_size, mask_zero=True)
        self.layer_norm = LayerNormalization()
        self.bn_relu = keras.layers.Dense(self.embedding_size, 'relu')
        self.bn_linear = keras.layers.Dense(self.embedding_size, 'linear')
        self.digit = keras.layers.Dense(1, 'sigmoid')

    def call(self, inputs):
        inputs_mask = math_ops.not_equal(inputs, 0)
        inputs_embed = self.embedding(inputs)
        valid_inputs_embed = mask_for_high_rank(inputs_embed, inputs_mask)

        x = self.bn_relu(inputs_embed)
        x = self.bn_linear(x)
        map2_masked = exp_mask_for_high_rank(x, inputs_mask)
        soft = tf.nn.softmax(map2_masked, 2)  # bs,sk,code_len,vec
        attn_output = soft * inputs_embed

        if self.task_type == 'dx':
            attn_output += inputs_embed
        else:
            attn_output = keras.layers.Add()([attn_output, valid_inputs_embed])

        attn_output = tf.reduce_sum(attn_output, 2)  # bs, sk, vec
        attn_output = self.layer_norm(attn_output)

        soft_output = tf.reduce_sum(map2_masked, -1)  # bs,skip_window,
        soft_output = tf.nn.softmax(soft_output, 2)  # bs,skip_window

        return soft_output, attn_output


# the third level Attention pooling
class VisitMultiDimAttn(keras.layers.Layer):

    def __init__(self, embedding_size, **kwargs):
        super(VisitMultiDimAttn, self).__init__(**kwargs)
        self.dense = BatchNormalDenseLayer(embedding_size, 'relu', False, None)
        self.linear = Linear(embedding_size)
        self.layer_norm = LayerNormalization()

    def call(self, inputs):
        tensor, mask = inputs
        x = self.dense(tensor)
        x = self.linear(x)
        x = exp_mask_for_high_rank(x, mask)
        soft = tf.nn.softmax(x, 1)  # bs,skip_window,vec
        soft_output = tf.reduce_sum(x, 2)  #bs,skip_window,
        soft_output = tf.nn.softmax(soft_output, 1)  # bs,skip_window

        attn_output = soft * tensor
        attn_output = tf.reduce_sum(attn_output, 1)  # bs, vec

        return soft_output, attn_output