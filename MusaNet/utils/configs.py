import argparse
import os
from os.path import join


class Configs(object):
    def __init__(self):
        root_dir, _ = os.path.split(os.path.abspath(__file__))
        root_dir = os.path.dirname(root_dir)
        root_dir = os.path.dirname(root_dir)
        root_dir = os.path.dirname(root_dir)
        self.project_dir = root_dir
        self.icd_file = join(self.project_dir, 'src/utils/ontologies/D_ICD_DIAGNOSES.csv')
        self.ccs_file = join(self.project_dir, 'src/utils/ontologies/SingleDX-edit.txt')
        self.icd_hierarchy = join(self.project_dir, 'src/utils/ontologies/codes_2L.json')

        self.dataset_dir = join(self.project_dir, 'dataset', 'processed')
        self.standby_log_dir = self.mkdir(self.project_dir, 'logs')
        self.result_dir = self.mkdir(self.project_dir, 'outputs')
        self.all_model_dir = self.mkdir(self.result_dir, 'tasks')

        # ------parsing input arguments"--------
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', (lambda x: x.lower() in ('True', "yes", "true", "t", "1")))

        parser.add_argument('--data_source', type=str, default='mimic3', help='mimic3 or cms')
        parser.add_argument('--task', type=str, default='AAAI20', help='ICDM19 or AAAI20')
        parser.add_argument('--model', type=str, default='tesa', help='tesa, vanila_sa, or cbow')
        parser.add_argument('--verbose', type='bool', default=False, help='print ...')
        parser.add_argument('--predict_type', type=str, default='re',
                            help='dx:diagnosis; re:readmission,death: mortality, los: length of stay')
        parser.add_argument('--pos_encoding', type=str, default='embedding', help='None, embedding, or encoding')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        parser.add_argument('--version', type=str, default='original', help='version of original or latest.')

        # @ ----- control ----
        parser.add_argument('--train', type='bool', default=True, help='whether run train or test')
        parser.add_argument('--debug', type='bool', default=False, help='whether run as debug mode')
        parser.add_argument('--gpu', type=int, default=0, help='eval_period')
        parser.add_argument('--gpu_mem', type=float, default=None, help='eval_period')
        parser.add_argument('--save_model', type='bool', default=True, help='save_model')
        parser.add_argument('--load_model', type='bool', default=False, help='load_model')

        # @ ------------------multi-head self-attention------------------
        parser.add_argument('--num_heads', type=int, default=5, help='multi-head self-attention')

        # @ ------------------RNN------------------
        parser.add_argument('--cell_type', type=str, default='gru', help='cell unit')
        parser.add_argument('--hn', type=int, default=100, help='number of hidden units')

        # @ ------------------Dipole------------------
        parser.add_argument('--Dipole', type=str, default='location', help='location, general, concat')
        # @ ------------------Deepr------------------
        parser.add_argument('--max_len_codes', type=int, default=100, help='length of sentence')


        # @ ----------training ------
        parser.add_argument('--max_epoch', type=int, default=20, help='Max Epoch Number')
        parser.add_argument('--train_batch_size', type=int, default=128, help='Train Batch Size')
        parser.add_argument('--activation', type=str, default='relu', help='activation function')

        parser.add_argument('--valid_visits', type=int, default=10, help='eval_period')
        parser.add_argument('--num_hidden_layers', type=int, default=5, help='num_hidden_layers in transformer')


        # @ ----- code Processing ----
        parser.add_argument('--embedding_size', type=int, default=100, help='code ICDM19 size')
        parser.add_argument('--only_dx_flag', type='bool', default=True, help='only_dx_flag')
        parser.add_argument('--visit_threshold', type=int, default=4, help='visit_threshold')
        parser.add_argument('--min_cut_freq', type=int, default=5, help='min code frequency')

        # @ ------validatation-----
        parser.add_argument('--valid_size', type=int, default=500, help='evaluate similarity size')
        parser.add_argument('--top_k', type=int, default=1, help='number of nearest neighbors')

        # -------Hierarchical Self-Attention for AAAI20 ----------
        parser.add_argument('--direction', type=str, default='Bi-Di', help='None or Bi-Direction')

        # ------- ablation --------
        parser.add_argument('--self_attn', type='bool', default=True, help='whether have middle layer of self-attn')
        parser.add_argument('--attn_pooling', type='bool', default=True, help='whether have last layer of attn pooling')

        parser.add_argument('--pre_train', type=str, default=None, help='med2vec or MCE')

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        ## ---- to member variables -----
        for key, value in self.args.__dict__.items():
            if key not in ['test', 'shuffle']:
                exec('self.%s = self.args.%s' % (key, key))

        #------------------path-------------------------------
        self.model_dir = self.mkdir(self.all_model_dir, self.task, self.model)
        self.summary_dir = self.mkdir(self.model_dir, 'summary')
        self.ckpt_dir = self.mkdir(self.model_dir, 'ckpt')
        self.log_dir = self.mkdir(self.model_dir, 'log_files')
        self.saved_vect_dir = self.mkdir(self.model_dir, 'vects')
        self.dict_dir = self.mkdir(self.result_dir, 'dict')
        self.processed_dir = self.mkdir(self.result_dir, 'processed_data')
        self.processed_task_dir = self.mkdir(self.processed_dir, self.task)

        self.processed_name = '_'.join([self.data_source, self.task]) + '.pickle'
        self.processed_path = join(self.processed_task_dir, self.processed_name)
        # self.dict_path = join(self.dict_dir, self.dict_name)
        # self.ckpt_path = join(self.ckpt_dir, self.model_ckpt_name)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        self.log_name = self.get_params_str(['data_source', 'model', 'predict_type', 'direction', 'pos_encoding',
                                             'num_hidden_layers', 'num_heads', 'visit_threshold', 'valid_visits', 'max_epoch',
                                             'version', 'Dipole', 'only_dx_flag', 'dropout', 'train_batch_size'])

    def get_params_str(self, params):
        def abbreviation(name):
            words = name.strip().split('_')
            abb = ''
            for word in words:
                abb += word[0]
            return abb

        abbreviations = map(abbreviation, params)
        model_params_str = ''
        for paramsStr, abb in zip(params, abbreviations):
            model_params_str += '_' + str(eval('self.args.' + paramsStr))
        return model_params_str

    def mkdir(self, *args):
        dir_path = join(*args)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    def get_file_name_from_path(self, path):
        assert isinstance(path, str)
        file_name = '.'.join((path.split('/')[-1]).split('.')[:-1])
        return file_name

cfg = Configs()
