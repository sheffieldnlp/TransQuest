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


def flatten_with_sanity_check(true_tags, test_tags):
    assert len(true_tags) == len(test_tags)
    flat_true, flat_test = [], []
    for i, (true_tags_i, test_tags_i) in enumerate(zip(true_tags, test_tags)):
        if len(true_tags_i) != len(test_tags_i):
            print('Warning inconsistent number of labels and predictions for line {}: {} and {}. Skipping'.format(
                i, len(true_tags_i), len(test_tags_i)
            ))
            continue
        else:
            flat_true.extend(true_tags_i)
            flat_test.extend(test_tags_i)
    return flat_true, flat_test


def compute_scores(true_tags, test_tags):
    flat_true, flat_pred = flatten_with_sanity_check(true_tags, test_tags)
    if len(list(set(flat_true + flat_pred))) > 1:
        f1_bad, f1_good = f1_score(flat_true, flat_pred, average=None, pos_label=None)
    else:
        print('Warning! All predictions and gold labels are the same class.')
        f1_bad, f1_good = 0, 0
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
