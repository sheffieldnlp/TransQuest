import numpy as np

from sklearn.metrics import mean_absolute_error, accuracy_score, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr, spearmanr


def pearson_corr(labels, preds):
    return pearsonr(labels, preds)[0]


def spearman_corr(labels, preds):
    return spearmanr(labels, preds)[0]


def rmse(labels, preds):
    return np.sqrt(((np.asarray(preds, dtype=np.float32) - np.asarray(labels, dtype=np.float32)) ** 2).mean())


def _flatten_confusion_matrix(labels, preds):
    return ' '.join(['{}'.format(s) for s in confusion_matrix(labels, preds).ravel()])


def define_evaluation_metrics(config):
    if config['regression']:
        metrics = {
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr,
            'mae': mean_absolute_error,
            'rmse': rmse,
        }
    else:
        metrics = {
            'accuracy': accuracy_score,
            'mcc': matthews_corrcoef,
        }
        if config['num_labels'] == 2:  # binary classification
            metrics.update({'tn, fp, fn, tp': _flatten_confusion_matrix})
    return metrics
