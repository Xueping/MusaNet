# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


class EmbeddingSharedWeights(keras.layers.Layer):
    """Calculates input embeddings"""

    def __init__(self, vocab_size, hidden_size, **kwargs):
        """Specify characteristic parameters of embedding layer.

        Args:
          vocab_size: Number of tokens in the embedding. (Typically ~32,000)
          hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
        """
        super(EmbeddingSharedWeights, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, _):

          # Create and initialize weights. The random normal initializer was chosen
          # randomly, and works well.
          self.shared_weights = self.add_weight(
              name="weights", shape = (self.vocab_size, self.hidden_size),
              dtype='float32', trainable=True,
              initializer=tf.random_normal_initializer(0.0, self.hidden_size ** -0.5))
          # self.built = True

    def call(self, x):
        """Get token embeddings of x.

        Args:
          x: An int64 tensor with shape [batch_size, n_visits, n_codes]
        Returns:
          embeddings: float32 tensor with shape [batch_size, n_visits, n_codes, embedding_size]
          padding: float32 tensor with shape [batch_size, n_visits, n_codes] indicating the
            locations of the padding tokens in x.
        """
        # Create binary mask of size [batch_size, n_visits, n_codes]
        mask = tf.cast(tf.not_equal(x, 0), tf.float32)

        embeddings = tf.gather(self.shared_weights, x)
        embeddings *= tf.expand_dims(mask, -1)

        # Scale embedding by the sqrt of the hidden size
        embeddings *= self.hidden_size ** 0.5

        return embeddings
