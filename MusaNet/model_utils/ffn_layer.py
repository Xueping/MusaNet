
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from MusaNet.model_utils.normalization_layer import LayerNormalization


class FeedForwardNetwork(keras.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, train, dropout,  **kwargs):
    super(FeedForwardNetwork, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.dropout = dropout
    self.train = train
    self.layer_norm = LayerNormalization()

    self.filter_dense_layer = keras.layers.Dense(
        filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
    self.output_dense_layer = keras.layers.Dense(
        hidden_size, use_bias=True, name="output_layer")

  def call(self, x):
    """Return outputs of the feedforward network.

    Args:
      x: tensor with shape [batch_size, length, hidden_size]

    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, hidden_size]
    """
    output = self.filter_dense_layer(x)
    # if self.train:
    #   output = tf.nn.dropout(output, self.dropout)
    output = self.output_dense_layer(output)

    output = keras.layers.Add()([output, x])
    output = self.layer_norm(output)

    return output
