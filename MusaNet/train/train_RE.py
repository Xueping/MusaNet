from MusaNet.utils.configs import cfg
from MusaNet.utils.record_log import RecordLog
from template.evaluation import EvaluationTemplate as Evaluation
import numpy as np
from MusaNet.musaNet.model import VisitModel as Model
import os
from MusaNet.dataset.dataset_full import VisitDataset

import warnings
warnings.filterwarnings('ignore')
logging = RecordLog()


def train():

    visit_threshold = cfg.visit_threshold
    epochs = cfg.max_epoch
    batch_size = cfg.train_batch_size

    data_set = VisitDataset()
    data_set.prepare_data(visit_threshold)
    data_set.build_dictionary()
    data_set.load_data()
    # for i in range(1, 9):
    model = Model(data_set, cfg.num_hidden_layers, cfg.direction)
    model.build_network()
    model.model.fit([data_set.train_context_codes, data_set.train_intervals],
                    data_set.train_labels_2,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=([data_set.dev_context_codes, data_set.dev_intervals], data_set.dev_labels_2)
                    )

    metrics = model.model.evaluate([data_set.test_context_codes, data_set.test_intervals], data_set.test_labels_2)
    log_str = 'Single fold accuracy is {}'.format(metrics[1])
    logging.add(log_str)

    predicts = model.model.predict([data_set.test_context_codes, data_set.test_intervals])
    predict_classes = predicts > 0.5
    predict_classes = predict_classes.astype(np.int)
    metrics = Evaluation.metric_pred(data_set.test_labels_2, predicts, predict_classes)
    logging.add(metrics)
    logging.done()


def test():
    pass


def main():
    if cfg.train:
        train()
    else:
        test()


def output_model_params():
    logging.add()
    logging.add('==>model_title: ' + cfg.model_name[1:])
    logging.add()
    for key,value in cfg.args.__dict__.items():
        if key not in ['test','shuffle']:
            logging.add('%s: %s' % (key, value))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    main()