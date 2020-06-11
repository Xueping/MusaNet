from abc import ABCMeta
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, auc
import numpy as np


class EvaluationTemplate(metaclass=ABCMeta):

    def __init__(self, model, logging):
        self.logging = logging
        if model is not None:
            self.model = model
            self.reverse_dict = model.reverse_dict
            self.dictionary = dict(zip(model.reverse_dict.values(), model.reverse_dict.keys()))
            self.verbose = model.verbose
            self.valid_samples = model.valid_samples
            self.top_k = model.top_k

    # @abstractmethod
    def get_clustering_nmi(self, sess, ground_truth):
        pass

    # @abstractmethod
    def get_nns_p_at_top_k(self, sess, ground_truth):
        pass

    # @abstractmethod
    def get_nns_pairs_count(self, ground_truth):
        pass

    @staticmethod
    def metric_pred(y_true, probs, y_pred):
        [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
        # print(TN, FP, FN, TP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (FP + TN)
        precision = TP / (TP + FP)
        sensitivity = recall = TP / (TP + FN)
        # f_score = 2 * TP / (2 * TP + FP + FN)

        # calculate AUC
        roc_auc = roc_auc_score(y_true, probs)
        # print('roc_auc: %.4f' % roc_auc)
        # calculate roc curve
        # fpr, tpr, thresholds = roc_curve(y_true, probs)

        # calculate precision-recall curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, probs)
        # calculate F1 score
        f_score = f1_score(y_true, y_pred)
        # calculate precision-recall AUC

        pr_auc = auc(recall_curve, precision_curve)
        accuracy = round(accuracy, 4)
        precision = round(precision, 4)
        sensitivity = round(sensitivity, 4)
        specificity = round(specificity, 4)
        f_score = round(f_score, 4)
        pr_auc = round(pr_auc, 4)
        roc_auc = round(roc_auc, 4)

        return [accuracy, precision, sensitivity, specificity, f_score, pr_auc, roc_auc]

    @staticmethod
    def recall_top(y_true, y_pred, rank=[5, 10, 15, 20, 25, 30]):
        recall = list()
        for i in range(len(y_pred)):
            this_one = list()
            codes = y_true[i]
            tops = y_pred[i]
            for rk in rank:
                length = len(set(codes))
                if length > rk:
                    length = rk
                this_one.append(len(set(codes).intersection(set(tops[:rk])))*1.0/length)
            recall.append(this_one)
        return np.round((np.array(recall)).mean(axis=0), decimals=4).tolist(), \
               np.round((np.array(recall)).std(axis=0), decimals=4).tolist()

    @staticmethod
    def code_level_top(y_true, y_pred, rank=[5, 10, 15, 20, 25, 30]):
        recall = list()
        for i in range(len(y_pred)):
            this_one = list()
            codes = y_true[i]
            tops = y_pred[i]
            for rk in rank:
                this_one.append(len(set(codes).intersection(set(tops[:rk]))) * 1.0 / rk)
            recall.append(this_one)
        return np.round((np.array(recall)).mean(axis=0), decimals=4).tolist()
