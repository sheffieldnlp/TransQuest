import argparse
import numpy as np
import os
import shutil

from collections import defaultdict

import torch

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from transquest.algo.transformers.evaluation import pearson_corr, spearman_corr
from transquest.algo.transformers.run_model import QuestModel

from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetWordLevel


def train_model(train_set, config, n_fold=None, test_size=None, return_model=False):
    config['running_seed'] = config['SEED'] * n_fold if n_fold is not None else config['SEED']
    model = QuestModel(config['model_type'], config['model_name'], num_labels=1, use_cuda=torch.cuda.is_available(), args=config)
    if test_size:
        train_n, eval_df_n = train_test_split(train_set, test_size=test_size, random_state=config['running_seed'])
    else:
        train_n = train_set
        eval_df_n = None
    if config['regression']:
        model.train_model(train_n, eval_df=eval_df_n, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
    else:
        model.train_model(train_n, eval_df=eval_df_n)
    if return_model:
        return model


def evaluate_model(test_set, config, model=None):
    if model is None:
        model = QuestModel(config['model_type'], config['best_model_dir'], num_labels=1, use_cuda=torch.cuda.is_available(), args=config)
    if config['regression']:  # TODO: this is repeated
        _, model_outputs = model.eval_model(test_set, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
    else:
        _, model_outputs = model.eval_model(test_set)
    return model_outputs


def train_cycle(train, test, config, test_size):
    train_model(train, config, test_size=test_size)
    model_outputs = evaluate_model(test, config)
    return model_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--features_pref', default=None, required=False)
    parser.add_argument('--test_size', default=0.1, type=float)
    args = parser.parse_args()
    config = load_config(args)
    train_set = DatasetWordLevel(config, evaluate=False)
    test_set = DatasetWordLevel(config, evaluate=True)
    train_data = train_set.make_dataset(
        os.path.join(args.data_dir, 'train', 'train.src'),
        os.path.join(args.data_dir, 'train', 'train.mt'),
        os.path.join(args.data_dir, 'train', 'train.tags'),
    )
    test_data = test_set.make_dataset(
        os.path.join(args.data_dir, 'dev', 'dev.src'),
        os.path.join(args.data_dir, 'dev', 'dev.mt'),
        os.path.join(args.data_dir, 'dev', 'dev.tags'),
    )
    train_cycle(train_data, test_data, config, args.test_size)


if __name__ == '__main__':
    main()
