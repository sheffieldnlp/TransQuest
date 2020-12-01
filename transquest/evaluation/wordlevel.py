from __future__ import division, print_function

import numpy as np

from sklearn.metrics import f1_score, matthews_corrcoef

"""
Scoring programme for WMT'20 Task 2 HTER **word-level**
"""

# -------------PREPROCESSING----------------
def list_of_lists(a_list):
    """
    check if <a_list> is a list of lists
    """
    if (
        isinstance(a_list, (list, tuple, np.ndarray))
        and len(a_list) > 0
        and all([isinstance(l, (list, tuple, np.ndarray)) for l in a_list])
    ):
        return True
    return False


def flatten(lofl):
    """
    convert list of lists into a flat list
    """
    if list_of_lists(lofl):
        return [item for sublist in lofl for item in sublist]
    elif type(lofl) == dict:
        return lofl.values()


def compute_scores(true_tags, test_tags):
    flat_true = flatten(true_tags)
    flat_pred = flatten(test_tags)

    f1_bad, f1_good = f1_score(flat_true, flat_pred, average=None, pos_label=None)
    mcc = matthews_corrcoef(flat_true, flat_pred)
    # Matthews correlation coefficient (MCC)
    # true/false positives/negatives
    # tp = tn = fp = fn = 0
    # for pred_tag, gold_tag in zip(flat_pred, flat_true):
    #     if pred_tag == 1:
    #         if pred_tag == gold_tag:
    #             tp += 1
    #         else:
    #             fp += 1
    #     else:
    #         if pred_tag == gold_tag:
    #             tn += 1
    #         else:
    #             fn += 1

    # mcc_numerator = (tp * tn) - (fp * fn)
    # mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    # mcc = mcc_numerator / mcc_denominator

    return f1_bad, f1_good, mcc
