# from src.utils.configs import cfg
from abc import ABCMeta


class ModelTemplate(metaclass=ABCMeta):
    def __init__(self, dataset):
        self.scope = None
        # self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
        #                                    initializer=tf.constant_initializer(0), trainable=False)

        # ----------- parameters -------------
        # self.is_scale = cfg.is_scale
        # self.is_plus_sa = cfg.is_plus_sa
        # self.is_plus_date = cfg.is_plus_date
        # self.predict_type = cfg.predict_type
        # self.activation = cfg.activation
        # self.embedding_size = cfg.embedding_size
        # self.max_epoch = cfg.max_epoch
        # self.num_samples = cfg.num_samples
        # self.valid_samples = cfg.valid_examples
        # self.gpu_device = '/gpu:' + str(cfg.gpu)
        # self.valid_visits = cfg.valid_visits
        # self.num_hidden_layers = cfg.num_hidden_layers
        # self.model_type = cfg.model_utils
        # self.data_src = cfg.data_source
        # self.top_k = cfg.top_k
        # self.verbose = cfg.verbose
        # self.is_date_encoding = cfg.is_date_encoding
        # self.hierarchical = cfg.hierarchical
        # self.train = cfg.train

        self.vocabulary_size = len(dataset.dictionary)
        self.dates_size = dataset.days_size
        self.reverse_dict = dataset.reverse_dictionary

        # ------ start ------
        self.tensor_dict = {}
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.summary = None
        self.opt = None
        self.train_op = None


    def build_network(self):
        pass


    def build_loss_optimizer(self):
        pass

