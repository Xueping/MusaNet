import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.ops import math_ops
from functools import reduce
from operator import mul
from src.AAAI20.baselines.data_transformer import med2vec_trans, mce_trans
from src.BiteNet.model_utils.normalization_layer import LayerNormalization

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


# the first level self-attention with ICDM19 layer
class FirstLevelSA(keras.layers.Layer):
    def __init__(self, vocabulary_size, embedding_size, activation, **kwargs):
        super(FirstLevelSA, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding = keras.layers.Embedding(vocabulary_size, embedding_size, mask_zero=True)
        self.learner = FirstLevelRepresentation(embedding_size, activation)

    def call(self, inputs):
        inputs_mask = math_ops.not_equal(inputs, 0)
        inputs_embed = self.embedding(inputs)
        x = BatchNormalDenseLayer(self.embedding_size, 'relu', False, None)(inputs_embed)
        x = BatchNormalDenseLayer(self.embedding_size, 'linear', False, None)(x)
        map2_masked = exp_mask_for_high_rank(x, inputs_mask)
        soft = tf.nn.softmax(map2_masked, 2)  # bs,sk,code_len,vec
        attn_output = tf.reduce_sum(soft * inputs_embed, 2)  # bs, sk, vec
        attn_output = self.learner(attn_output)

        soft_output = tf.reduce_sum(map2_masked, -1)  # bs,skip_window,
        soft_output = tf.nn.softmax(soft_output, 2)  # bs,skip_window

        # visit_mask = tf.reduce_sum(tf.cast(inputs_mask, tf.int32), -1)  # [bs,max_visits]
        # visit_mask = tf.cast(visit_mask, tf.bool)
        # attn_output = mask_for_high_rank(attn_output, visit_mask)

        return soft_output, attn_output


# The first level sum with ICDM19 layer
class FirstLevelSumDence(keras.layers.Layer):
    def __init__(self, vocabulary_size, embedding_size, activation):
        super(FirstLevelSumDence, self).__init__()
        self.embedding = keras.layers.Embedding(vocabulary_size, embedding_size, mask_zero=True)
        self.learner = FirstLevelRepresentation(embedding_size, activation)

    def call(self, inputs):
        inputs_embed = self.embedding(inputs)
        inputs_mask = math_ops.not_equal(inputs, 0)
        valid_inputs_embed = mask_for_high_rank(inputs_embed, inputs_mask)  # batch_size,skip_window,visit_len,vec
        inputs_merged = tf.reduce_sum(valid_inputs_embed, 2)  # batch_size,skip_window,vec
        inputs_merged = self.learner(inputs_merged)
        return inputs_merged


# The first level sum with ICDM19 layer
class FirstLevelSumDencePreTrain(keras.layers.Layer):
    def __init__(self, vocabulary_size, embedding_size, activation, pre_train, dictionary):
        super(FirstLevelSumDencePreTrain, self).__init__()
        if pre_train == 'med2vec':
            weights = med2vec_trans(dictionary)
            weights = keras.layers.Dense(embedding_size, activation='linear')(
                tf.convert_to_tensor(weights, dtype=tf.float32))
            weights = np.array(weights)
        else:
            weights = mce_trans(dictionary)
        self.embedding = keras.layers.Embedding(vocabulary_size, embedding_size,
                                                mask_zero=True,
                                                embeddings_initializer=keras.initializers.Constant(weights),
                                                trainable=False)
        self.learner = FirstLevelRepresentation(embedding_size, activation)

    def call(self, inputs):
        inputs_embed = self.embedding(inputs)
        inputs_mask = math_ops.not_equal(inputs, 0)
        valid_inputs_embed = mask_for_high_rank(inputs_embed, inputs_mask)  # batch_size,skip_window,visit_len,vec
        inputs_merged = tf.reduce_sum(valid_inputs_embed, 2)  # batch_size,skip_window,vec
        inputs_merged = self.learner(inputs_merged)
        return inputs_merged


class FirstLevelSum(keras.layers.Layer):
    def __init__(self, vocabulary_size, embedding_size):
        super(FirstLevelSum, self).__init__()
        self.embedding = keras.layers.Embedding(vocabulary_size, embedding_size, mask_zero=True)

    def call(self, inputs):
        inputs_embed = self.embedding(inputs)
        inputs_mask = math_ops.not_equal(inputs, 0)
        valid_inputs_embed = mask_for_high_rank(inputs_embed, inputs_mask)  # batch_size,skip_window,visit_len,vec
        inputs_merged = tf.reduce_sum(valid_inputs_embed, 2)  # batch_size,skip_window,vec
        return inputs_merged


# the third level Attention pooling
class VisitMultiDimAttn(keras.layers.Layer):

    def __init__(self, embedding_size, **kwargs):
        super(VisitMultiDimAttn, self).__init__(**kwargs)
        self.dense = BatchNormalDenseLayer(embedding_size, 'relu', False, None)
        self.linear = Linear(embedding_size)

    def call(self, inputs):
        tensor, mask = inputs
        x = self.dense(tensor)
        x = self.linear(x)
        x = exp_mask_for_high_rank(x, mask)
        soft = tf.nn.softmax(x, 1)  # bs,skip_window,vec
        soft_output = tf.reduce_sum(x, 2)  #bs,skip_window,
        soft_output = tf.nn.softmax(soft_output, 1)  # bs,skip_window
        attn_output = tf.reduce_sum(soft * tensor, 1)  # bs, vec
        return soft_output, attn_output


# the third level summing
class VisitMultiDimSum(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(VisitMultiDimSum, self).__init__(**kwargs)

    def call(self, inputs):
        tensor, mask = inputs
        valid_inputs_embed = mask_for_high_rank(tensor, mask)  # batch_size,skip_window,vec
        inputs_merged = tf.reduce_sum(valid_inputs_embed, 1)  # batch_size,vec
        return inputs_merged



# The first level sum with ICDM19 layer
class FirstLevelRepresentation(keras.layers.Layer):
    def __init__(self, output_size, activation, **kwargs):
        super(FirstLevelRepresentation, self).__init__(**kwargs)
        self.output_size = output_size
        self.flatten = Flatten(1)
        self.leaner = keras.layers.Dense(output_size, activation=activation)
        self.reconstruct = Reconstruct(1)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.leaner(x)
        x = self.reconstruct([x, inputs])
        return x


# Visit Length layer
class VisitLengthLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VisitLengthLayer, self).__init__(**kwargs)

    def call(self, inputs):
        visit_mask = tf.reduce_sum(tf.cast(inputs, tf.int32), -1)  # [bs,max_visits]
        visit_mask = tf.cast(visit_mask, tf.bool)
        tensor_len = tf.reduce_sum(tf.cast(visit_mask, tf.int32), -1)  # [bs]

        return tensor_len, visit_mask


class EuclideanDistance(keras.layers.Layer):
    def __init__(self):
        super(EuclideanDistance, self).__init__()

    def call(self, inputs):
        A, B = inputs
        l2_norm = tf.norm(A - B, axis=0, ord='euclidean')
        return l2_norm


class CosineDistance(keras.layers.Layer):
    def __init__(self):
        super(CosineDistance, self).__init__()

    def call(self, inputs):
        A, B= inputs
        normalize_a = tf.nn.l2_normalize(A, 0)
        normalize_b = tf.nn.l2_normalize(B, 0)
        cos_similarity = tf.multiply(normalize_a, normalize_b)
        # loss = tf.keras.losses.CosineSimilarity(A,B)
        return cos_similarity

# Mask Visit in patient journey
class VisitMaskedLayer(keras.layers.Layer):
    def __init__(self):
        super(VisitMaskedLayer, self).__init__()

    def call(self, inputs):
        inputs_merged, visit_mask = inputs
        masked_visit = mask_for_high_rank(inputs_merged, visit_mask)
        return masked_visit


# Last Relevant output
class LastRelevant(keras.layers.Layer):
    def __init__(self):
        super(LastRelevant, self).__init__()

    def call(self, inputs):
        outputs, tensor_len = inputs
        tensor_len = tf.cast(Flatten(0)(tensor_len), 'int32')
        batch_size = tf.shape(outputs)[0]
        max_length = tf.shape(outputs)[1]
        out_size = int(outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (tensor_len - 1)
        flat = tf.reshape(outputs, [-1, out_size])
        relevant = tf.gather(flat, index)
        # print('outputs shape:', outputs.get_shape())
        # print('last output shape:', relevant.get_shape())

        return relevant


# The second level self-attention
class VisitSelfAttnDenseLayer(keras.layers.Layer):
    def __init__(self, is_scale):
        self.is_scale = is_scale
        super(VisitSelfAttnDenseLayer, self).__init__()

    def build(self, input_shape):
        self.skip_window = input_shape[0][1]
        self.embedding_size = input_shape[0][2]
        self.dense =  BatchNormalDenseLayer(self.embedding_size, 'relu', False, None)
        self.linear = Linear(self.embedding_size)

    def call(self, inputs):
        tensor, mask = inputs
        # mask generation
        attn_mask = tf.cast(tf.linalg.diag(- tf.ones([self.skip_window], tf.int32)) + 1, tf.bool) # skip_window, skip_window
        # non-linear for context
        rep_map = self.dense(tensor)
        rep_map_tile = keras.layers.Reshape((1, self.skip_window, self.embedding_size))(rep_map)
        rep_map_tile = tf.tile(rep_map_tile, [1, self.skip_window, 1, 1]) # bs,sl,sl,vec

        dependent = self.linear(rep_map)
        dependent_etd = keras.layers.Reshape((1,self.skip_window,self.embedding_size))(dependent) # batch_size, 1,sw, vec_size

        head = self.linear(rep_map)
        head_etd = keras.layers.Reshape((self.skip_window,1,self.embedding_size))(head) # batch_size, sw,1, vec_size

        attention_fact = keras.layers.Add()([dependent_etd, head_etd])# batch_size, sw,sw, vec_size
        bias_1 = self.add_weight(shape=(self.embedding_size,),
                                 initializer='zeros', dtype='float32',
                                 trainable=True)
        attention_fact = tf.nn.bias_add(attention_fact, bias_1)
        if self.is_scale:
            attention_fact = scaled_tanh(attention_fact, 5.0)
        else:
            attention_fact = keras.layers.Activation('tanh')(attention_fact)
            attention_fact = self.linear(attention_fact)

        logits_masked = exp_mask_for_high_rank(attention_fact, attn_mask)
        attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
        attn_score = mask_for_high_rank(attn_score, attn_mask)
        attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec
        attn_result = mask_for_high_rank(attn_result, mask)

        linear_rep = self.linear(rep_map)
        linear_attn_result = self.linear(attn_result)

        gate_add = keras.layers.Add()([linear_rep, linear_attn_result])
        bias_2 = self.add_weight(shape=(self.embedding_size,),
                                  initializer='zeros', dtype='float32',
                                  trainable=True)
        gate_add = tf.nn.bias_add(gate_add, bias_2)
        gate_add = keras.layers.Activation('sigmoid')(gate_add)

        # input gate
        output = gate_add * rep_map + (1 - gate_add) * attn_result
        return output


# The second level self-attention
class VisitSelfAttnResnetLayer(keras.layers.Layer):
    def __init__(self):
        super(VisitSelfAttnResnetLayer, self).__init__()

    def build(self, input_shape):
        self.skip_window = input_shape[0][1]
        self.embedding_size = input_shape[0][2]
        self.dense =  BatchNormalDenseLayer(self.embedding_size, 'relu', False, None)
        self.linear = Linear(self.embedding_size)

    def call(self, inputs):
        tensor, mask = inputs
        # mask generation, # skip_window, skip_window
        attn_mask = tf.cast(tf.linalg.diag(- tf.ones([self.skip_window], tf.int32)) + 1,tf.bool)
        # non-linear for context
        rep_map = self.dense(tensor)
        rep_map_tile = keras.layers.Reshape((1, self.skip_window, self.embedding_size))(rep_map)
        rep_map_tile = tf.tile(rep_map_tile, [1, self.skip_window, 1, 1])  # bs,sl,sl,vec

        dependent = self.linear(rep_map)
        dependent_etd = keras.layers.Reshape((1, self.skip_window, self.embedding_size))(
            dependent)  # batch_size, 1,sw, vec_size

        head = self.linear(rep_map)
        head_etd = keras.layers.Reshape((self.skip_window, 1, self.embedding_size))(head)  # batch_size, sw,1, vec_size

        attention_fact = keras.layers.Add()([dependent_etd, head_etd])  # batch_size, sw,sw, vec_size
        bias_1 = self.add_weight(shape=(self.embedding_size,),
                                 initializer='zeros', dtype='float32',
                                 trainable=True)
        attention_fact = tf.nn.bias_add(attention_fact, bias_1)

        attention_fact = keras.layers.Activation('tanh')(attention_fact)
        attention_fact = self.linear(attention_fact)

        logits_masked = exp_mask_for_high_rank(attention_fact, attn_mask)
        attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
        attn_score = mask_for_high_rank(attn_score, attn_mask)
        attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec
        attn_result = mask_for_high_rank(attn_result, mask)

        output = keras.layers.Add()([attn_result,rep_map])
        output = keras.layers.Activation('relu')(output)

        return output


# The second level self-attention
class DirectionalVisitSelfAttnResnetLayer(keras.layers.Layer):
    def __init__(self, direction):
        self.direction = direction
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
        # generate directional mask, shape of (1,skip_window, skip_window)
        # direct_mask_tile = keras.backend.expand_dims(direct_mask, 0)
        # generate input tensor mask, shape of (bs,sw,sw)
        # rep_mask_tile = tf.tile(keras.backend.expand_dims(mask, 1), [1, self.skip_window, 1])
        # attention mask generation, shape of (bs,sw,sw)
        # attn_mask = tf.logical_and(direct_mask, rep_mask_tile)

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
        attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec_size
        attn_result = mask_for_high_rank(attn_result, mask)

        output = keras.layers.Add()([attn_result, rep_map])
        return output
