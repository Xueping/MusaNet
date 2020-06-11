from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):
  """Applies layer normalization."""

  def __init__(self, **kwargs):
    # Pass dtype=float32, as we have not yet tested if layer norm is numerically
    # stable in float16 and bfloat16.
    super(LayerNormalization, self).__init__(dtype="float32", **kwargs)

  def build(self, input_shape):
    self.hidden_size = input_shape[-1]
    """Builds the layer."""
    self.scale = self.add_weight(
        "layer_norm_scale",
        shape=[self.hidden_size],
        initializer=tf.ones_initializer())
    self.bias = self.add_weight(
        "layer_norm_bias",
        shape=[self.hidden_size],
        initializer=tf.zeros_initializer())
    super(LayerNormalization, self).build(input_shape)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
    }

  def call(self, x, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
    return norm_x * self.scale + self.bias