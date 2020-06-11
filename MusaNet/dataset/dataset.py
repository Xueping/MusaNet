from abc import ABCMeta
import collections
from os.path import join
import json
from MusaNet.utils.configs import cfg
from utils.icd9_ontology import SecondLevelCodes


class DatasetTemplate(metaclass=ABCMeta):

    def __init__(self):

        self.min_freq = cfg.min_cut_freq
        self.dx_only = cfg.only_dx_flag
        self.batch_size = cfg.train_batch_size
        self.visit_threshold = cfg.visit_threshold
        self.valid_visits = cfg.valid_visits
        self.code_to_second_level_code_dict = dict()

        self.word_sample = dict()
        self.reverse_dictionary = dict()
        self.dictionary = dict()
        self.dictionary_3digit = dict()
        self.words_count = 0
        self.total_visits = 0
        self.code_no_per_visit = 0
        self.max_len_visit = 0
        self.max_visits = 0
        self.train_size = 0

        self.patients_file = None
        self.dict_file = None
        self.days_size = None
        self.patients = None
        self.patients_codes_file = None

    def prepare_data(self, visit_threshold):
            if cfg.data_source == 'mimic3':
                self.patients_file = join(cfg.dataset_dir, 'patients_mimic3_full.json')
                self.dict_file = join(cfg.dataset_dir, 'mimic3_dict_' + str(cfg.only_dx_flag) +
                                      '_' + str(cfg.min_cut_freq))
                self.patients_codes_file = join(cfg.dataset_dir, 'patients_mimic3_codes')
                self.days_size = 12 * 365 + 1
            else:
                self.patients_file = join(cfg.dataset_dir, 'patients_cms_full.json')
                self.patients_codes_file = join(cfg.dataset_dir, 'patients_cms_codes')
                self.dict_file = join(cfg.dataset_dir, 'cms_dict')
                self.days_size = 4 * 365 + 1

            # logging.info('source data path:%s' % patients_file)
            with open(self.patients_file) as read_file:
                self.patients = json.load(read_file)

            # if not cfg.data_source == 'mimic3':
            #     self.patients = [patient for patient in self.patients if len(patient['visits']) >= visit_threshold]
            self.patients = [patient for patient in self.patients if len(patient['visits']) >= visit_threshold]

    def build_dictionary(self):

        all_codes = []  # store all diagnosis codes
        all_cpt_codes = []

        for patient in self.patients:
            for visit in patient['visits']:
                self.total_visits += 1
                dxs = visit['DXs']
                for dx in dxs:
                    all_codes.append('D_' + dx)
                if not self.dx_only:
                    txs = visit['CPTs']
                    all_cpt_codes.extend(txs)
                    for tx in txs:
                        all_codes.append('T_' + tx)

        # store all codes and corresponding counts
        count_org = []
        count_org.extend(collections.Counter(all_codes).most_common())

        # store filtering codes and counts
        count = []
        for word, c in count_org:
            word_tuple = [word, c]
            if c >= self.min_freq:
                count.append(word_tuple)
                self.words_count += c

        self.code_no_per_visit = self.words_count / self.total_visits

        slc = SecondLevelCodes(cfg.icd_hierarchy)
        second_level_codes = []
        # add padding
        self.dictionary['PAD'] = 0
        for word, cnt in count:
            index = len(self.dictionary)
            self.dictionary[word] = index
            if word[:2] == 'D_':
                digit = slc.second_level_codes_icd9(word[2:])
                second_level_codes.append(digit)
                self.code_to_second_level_code_dict[word] = digit

        count_second_level_codes = []
        count_second_level_codes.extend(collections.Counter(second_level_codes).most_common())
        for word, cnt in count_second_level_codes:
            index = len(self.dictionary_3digit)
            self.dictionary_3digit[word] = index
        print('Number of second level category: ', len(self.dictionary_3digit))


        # encoding patient using index
        for patient in self.patients:
            visits = patient['visits']
            len_visits = len(visits)
            if len_visits > self.max_visits:
                self.max_visits = len_visits
            for visit in visits:
                visit['Drugs'] = []
                dxs = visit['DXs']
                if len(dxs) == 0:
                    continue
                else:
                    visit['DXs'] = [self.dictionary['D_' + dx] for dx in dxs if 'D_' + dx in self.dictionary]
                len_current_visit = len(visit['DXs'])

                if not self.dx_only:
                    txs = visit['CPTs']
                    visit['CPTs'] = [self.dictionary['T_' + tx] for tx in txs if 'T_' + tx in self.dictionary]
                    # len_current_visit = len(visit['DXs']+visit['CPTs'])
                if len_current_visit > self.max_len_visit:
                    self.max_len_visit = len_current_visit

        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

        with open(self.dict_file + '.json', 'w') as fp:
            json.dump(self.reverse_dictionary, fp)

        # with open(self.patients_codes_file + '.json', 'w') as fp:
        #     json.dump(self.patients, fp)

        print('number of CPTs: ', len(all_cpt_codes))
        print('avg number of CPTs: ', len(all_cpt_codes) / self.total_visits)

    # @abstractmethod
    def generate_batch(self):
        pass

    def process_data(self):
        pass

    def load_data(self):
        pass

    def get_words_count(self):
        return self.words_count

    def get_dictionary(self):
        return self.dictionary

    def get_reverse_dictionary(self):
        return self.reverse_dictionary
