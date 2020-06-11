import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops import math_ops
from nn_utils.attentionLayers import VisitLengthLayer,  VisitSelfAttnResnetLayer, VisitMultiDimAttn
from MusaNet.model_utils.musa_layer import DirectionalVisitSelfAttnResnetLayer, FirstLevelSA
from MusaNet.model_utils import embedding_layer, position_encoding_layer as pos_layer
from template.model import ModelTemplate
from MusaNet.utils.configs import cfg


class VisitModel(ModelTemplate):

    def __init__(self, dataset, levels, direction):

        super(VisitModel, self).__init__(dataset)
        # ------ start ------
        self.embedding_size = cfg.embedding_size
        self.predict_type = cfg.predict_type
        self.train = cfg.train
        self.levels = levels
        self.lr = 0.0001
        self.dropout_rate = cfg.dropout
        self.direction = direction
        self.n_visits = cfg.valid_visits
        self.max_len_visit = dataset.max_len_visit
        self.vocabulary_size = len(dataset.dictionary)
        self.digit3_size = len(dataset.dictionary_3digit)
        print('length of digit3_size: ',self.digit3_size)
        self.model = None
        # ---- place holder -----
        self.inputs = keras.layers.Input(shape=(self.n_visits, self.max_len_visit,), dtype=tf.int32,
                                     name='train_inputs') #batch_size,skip_window,visit_len
        self.interval_inputs = keras.layers.Input(shape=(self.n_visits,), dtype=tf.int32, name='interval_inputs')
        self.inputs_mask = math_ops.not_equal(self.inputs, 0)
        self.interval_embedding = embedding_layer.EmbeddingSharedWeights(self.n_visits, self.embedding_size,
                                                                         name='interval_embedding')

        self.layer_model_b = None
        self.layer_model_f = None
        self.layer_model_n = None
        self.layer_model_visit_len = None
        self.layer_model_first_attn_pooling = None

    def build_network(self):

        # representation for visit
        # batch_size,skip_window,embedding_size
        weight, context_embed = FirstLevelSA(self.vocabulary_size, self.embedding_size,
                                             cfg.activation, cfg.predict_type, name='first_attn_pooling')(self.inputs)
        # context_embed = FirstLevelSumDence(self.vocabulary_size, self.embedding_size, cfg.activation)(self.inputs)
        print("first level shape: ", context_embed.get_shape())

        visit_len, visit_mask = VisitLengthLayer(name='layer_visit_len')(self.inputs_mask)
        if cfg.pos_encoding == 'embedding':
            # position embedding strategy
            e_p = self.interval_embedding(self.interval_inputs)
        else:
            # position encoding strategy
            e_p = pos_layer.PositionEncoding(self.embedding_size, True)(self.interval_inputs)
        context_embed = keras.layers.Add()([e_p, context_embed])

        if self.direction == 'None':
            attn_layers = []
            for i in range(self.levels):
                # self_attention Layer
                context_embed = VisitSelfAttnResnetLayer()((context_embed, visit_mask))
                attn_layers.append(context_embed)

            if len(attn_layers) <= 1:
                context_embed = context_embed
            else:
                context_embed = keras.layers.Add()(attn_layers)
            context_embed = keras.layers.Activation('relu')(context_embed)
            context_embed = keras.layers.Dense(self.embedding_size,'relu')(context_embed)
            context_embed = keras.layers.Dense(self.embedding_size,'linear')(context_embed)
            # Attention pooling Layer
            weight_n, context_fusion = VisitMultiDimAttn(self.embedding_size, name='weight_n')((context_embed,visit_mask))
        else:
            context_embed_forward = context_embed
            context_embed_backward = context_embed
            for i in range(self.levels):
                # self_attention Layer
                context_embed_forward = DirectionalVisitSelfAttnResnetLayer('forward', self.train, self.dropout_rate)(
                    (context_embed_forward, visit_mask))

                context_embed_backward = DirectionalVisitSelfAttnResnetLayer('backward', self.train, self.dropout_rate)(
                    (context_embed_backward, visit_mask))
            # Attention pooling Layer
            context_embed_forward = keras.layers.Add()([e_p, context_embed_forward])
            weight_f, context_fusion_forward = VisitMultiDimAttn(self.embedding_size, name='weight_f')\
                ((context_embed_forward, visit_mask))
            # Attention pooling Layer
            context_embed_backward = keras.layers.Add()([e_p, context_embed_backward])
            weight_b, context_fusion_backward = VisitMultiDimAttn(self.embedding_size, name='weight_b')\
                ((context_embed_backward, visit_mask))

            context_fusion = keras.layers.Concatenate()([context_fusion_forward, context_fusion_backward])

        context_fusion = keras.layers.Dense(self.embedding_size, 'relu')(context_fusion)
        context_fusion = keras.layers.Dense(self.embedding_size, 'linear')(context_fusion)
        context_fusion = keras.layers.Dropout(self.dropout_rate)(context_fusion)

        if self.predict_type == 'dx':
            logits = keras.layers.Dense(self.digit3_size, activation='sigmoid')(context_fusion)
            self.model = keras.Model(inputs=[self.inputs, self.interval_inputs],
                                     outputs=logits, name='hierarchicalSA')
            self.model.compile(optimizer=keras.optimizers.RMSprop(0.001),
                               loss='binary_crossentropy',
                               metrics=['accuracy'])

        elif self.predict_type == 're':
            logits = keras.layers.Dense(1, activation='sigmoid', use_bias=True)(context_fusion)
            self.model = keras.Model(inputs=[self.inputs, self.interval_inputs],
                                     outputs=logits, name='hierarchicalSA')
            self.model.compile(optimizer=keras.optimizers.RMSprop(0.001),
                               loss=keras.losses.BinaryCrossentropy(),
                               metrics=['accuracy'])

        # multiple tasks
        else:
            logits_dx = keras.layers.Dense(self.digit3_size, activation='sigmoid', name='los_dx')(context_fusion)
            logits_re = keras.layers.Dense(1, activation='sigmoid', use_bias=True, name='los_re')(context_fusion)
            self.model = keras.Model(inputs=[self.inputs, self.interval_inputs],
                                     outputs=[logits_dx, logits_re],
                                     name='hierarchicalSA')
            self.model.compile(keras.optimizers.RMSprop(0.001),
                               loss={'los_dx': 'binary_crossentropy',
                                     'los_re': 'binary_crossentropy'},
                               loss_weights={'los_dx': 2., 'los_re': 0.1},
                               metrics={'los_dx': 'accuracy', 'los_re': 'accuracy'})

        print(self.model.summary())
