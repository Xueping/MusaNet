import tensorflow as tf
from tensorflow import keras
from nn_utils.attentionLayers import Flatten, Reconstruct


class Reshape(keras.layers.Layer):
    def __init__(self):
        super(Reshape, self).__init__()

    def call(self, inputs):
        v, ref, embedding_size = inputs
        batch_size = tf.shape(ref)[0]
        n_visits = tf.shape(ref)[1]
        out = tf.reshape(v, [batch_size, n_visits, embedding_size])
        return out


class DenseActivation(keras.layers.Layer):
    def __init__(self, output_size, activation=None):
        super(DenseActivation, self).__init__()
        self.output_size = output_size
        self.flatten = Flatten(1)
        self.linear = keras.layers.Dense(output_size, activation=activation)
        self.reconstruct = Reconstruct(1)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.linear(x)
        x = self.reconstruct([x,inputs])
        return x

