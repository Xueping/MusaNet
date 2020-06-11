import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from MusaNet.dataset.dataset import DatasetTemplate


class VisitDataset(DatasetTemplate):

    def __init__(self):
        super(VisitDataset, self).__init__()
        self.train_context_codes = None
        self.train_labels_1 = None
        self.train_labels_2 = None
        self.train_context_onehots = None
        self.dev_context_codes = None
        self.dev_labels_1 = None
        self.dev_labels_2 = None
        self.dev_context_onehots = None
        self.test_context_codes = None
        self.test_labels_1 = None
        self.test_labels_2 = None
        self.test_context_onehots = None
        self.train_size = 0
        self.train_pids = None
        self.dev_pids = None
        self.test_pids = None
        self.train_intervals = None
        self.dev_intervals = None
        self.test_intervals = None

    def process_data(self):
        if not self.dx_only:
            self.max_len_visit += 10
            print(self.max_len_visit)
        batches = []
        n_zeros = 0
        for patient in self.patients:
            pid = patient['pid']
            # get patient's visits
            visits = patient['visits']
            # sorting visits by admission date
            sorted_visits = sorted(visits, key=lambda visit: visit['admsn_dt'])
            valid_visits = []
            for v in sorted_visits:
                if len(v['DXs']) > 0 and sum(v['DXs']) > 0:
                    valid_visits.append(v)

            if (len(valid_visits)) < 2:
                continue

            # number of visits and only use 10 visits to predict last one if number of visits is larger than 11
            no_visits = len(valid_visits)
            last_visit = valid_visits[no_visits - 1]
            second_last_visit = valid_visits[no_visits - 2]

            ls_codes = []
            ls_intervals = []
            # only use 10 visits to predict last one if number of visits is larger than 11
            if no_visits > self.valid_visits+1:
                feature_visits = valid_visits[no_visits-(self.valid_visits+1):no_visits-1]
            else:
                feature_visits = valid_visits[0:no_visits - 1]

            n_visits = len(feature_visits)
            # if n_visits == 0:
            #     n_zeros += 1
            first_valid_visit_dt = datetime.datetime.strptime(feature_visits[0]['admsn_dt'], "%Y%m%d")
            for i in range(n_visits):
                visit = feature_visits[i]
                codes = visit['DXs']
                if not self.dx_only:
                    length = len(visit['CPTs'])
                    if length < 11:
                        codes.extend(visit['CPTs'])
                    else:
                        codes.extend(visit['CPTs'][:10])

                if sum(codes) == 0:
                    n_zeros += 1

                current_dt = datetime.datetime.strptime(visit['admsn_dt'], "%Y%m%d")
                interval = (current_dt - first_valid_visit_dt).days + 1
                ls_intervals.append(interval)
                code_size = len(codes)
                # code padding
                if code_size < self.max_len_visit:
                    list_zeros = [0] * (self.max_len_visit - code_size)
                    codes.extend(list_zeros)
                ls_codes.append(codes)

            # visit padding
            if n_visits < self.valid_visits:
                for i in range(self.valid_visits - n_visits):
                    list_zeros = [0] * self.max_len_visit
                    ls_codes.append(list_zeros)
                    ls_intervals.append(0)


            last_dt = datetime.datetime.strptime(last_visit['admsn_dt'], "%Y%m%d")
            second_last_dt = datetime.datetime.strptime(second_last_visit['admsn_dt'], "%Y%m%d")
            days = (last_dt - second_last_dt).days
            if days <= 30:
                adm_label = 1
            else:
                adm_label = 0
            # --------- end readmission label --------------------

            # --------- second level category --------------------
            one_hot_labels = np.zeros(len(self.dictionary_3digit)).astype(int)
            last_codes = last_visit['DXs']
            for code in last_codes:
                code_str = self.reverse_dictionary[code]
                cat_code = self.code_to_second_level_code_dict[code_str]
                index = self.dictionary_3digit[cat_code]
                one_hot_labels[index] = 1
            # --------- end diagnosis label --------------------

            # --------- high level icd9 diagnosis label --------------------
            # one_hot_labels = np.zeros(19).astype(int)
            # last_codes = last_visit['DXs']
            # for code in last_codes:
            #     code_str = self.reverse_dictionary[code]
            #     index = convert_to_high_level_icd9(code_str[2:])
            #     one_hot_labels[index] = 1
            # --------- end diagnosis label --------------------
            batches.append(
                [np.array(ls_codes, dtype=np.int32), one_hot_labels, np.array([adm_label], dtype=np.int32), pid,
                 np.array(ls_intervals, dtype=np.int32)])

        print('number of non-context ', n_zeros)
        codes = []
        dx_labels = []
        re_labels = []
        pids = []
        intervals = []
        for batch in batches:
            codes.append(batch[0])
            dx_labels.append(batch[1])
            re_labels.append(batch[2])
            pids.append(batch[3])
            intervals.append(batch[4])

        return codes, dx_labels, re_labels, pids, intervals

    def load_data(self):

        data = self.process_data()
        context_codes = data[0]
        labels_1 = data[1]
        labels_2 = data[2]
        pids = data[3]
        intervals = data[4]

        context_codes = np.array(context_codes, dtype=np.int32)
        intervals = np.array(intervals, dtype=np.int32)
        labels_1 = np.array(labels_1, dtype=np.int32)
        labels_2 = np.array(labels_2, dtype=np.int32)

        self.train_context_codes, vt_context_codes, self.train_labels_1, vt_labels_1, \
        self.train_labels_2, vt_labels_2, self.train_pids, vt_pids, self.train_intervals, vt_intervals\
            = train_test_split(context_codes, labels_1, labels_2, pids, intervals, test_size=0.2, random_state=42)

        self.train_size = len(self.train_context_codes)

        self.dev_context_codes, self.test_context_codes, self.dev_labels_1, self.test_labels_1, self.dev_labels_2, \
        self.test_labels_2, self.dev_pids, self.test_pids, self.dev_intervals, self.test_intervals \
            = train_test_split(vt_context_codes, vt_labels_1, vt_labels_2, vt_pids,vt_intervals,
                               test_size=0.5, random_state=42)
