import argparse
import numpy as np
import os
import shutil

from scipy.stats import pearsonr, spearmanr

from collections import defaultdict

import torch

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from transquest.algo.transformers.evaluation import pearson_corr, spearman_corr
from transquest.algo.transformers.run_model import QuestModel
from transquest.util.draw import draw_scatterplot

from transquest.data.load_config import load_config
from transquest.data.normalizer import un_fit
from transquest.data.dataset import DatasetSentLevel


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


def predictions_column(training_run):
    return 'predictions_{}'.format(training_run)


def train_cycle(train, test, test_tsv, config, output_dir, test_size):
    run = 1
    if config['evaluate_during_training']:
        if config['n_fold'] > 1:
            for i in range(config['n_fold']):
                print('Training with N folds. Now N is {}'.format(i))
                run = i
                if os.path.exists(config['output_dir']) and os.path.isdir(config['output_dir']):
                    shutil.rmtree(config['output_dir'])
                train_model(train, config, n_fold=i, test_size=test_size)
                model_outputs = evaluate_model(test, config)
                test_tsv[predictions_column(run)] = model_outputs
        else:
            train_model(train, config, test_size=test_size)
            model_outputs = evaluate_model(test, config)
            test_tsv[predictions_column(run)] = model_outputs
    else:
        model = train_model(train, config, return_model=True)
        model_outputs = evaluate_model(test, config, model=model)
        test_tsv[predictions_column(run)] = model_outputs

    runs = 1 if config['n_fold'] < 2 else config['n_fold']
    correlations = defaultdict(list)

    test_tsv = un_fit(test_tsv, 'labels')

    for r in range(runs):
        test_tsv = un_fit(test_tsv, predictions_column(r))
        correlations['pearson'].append(pearsonr(test_tsv[predictions_column(r)], test_tsv['labels'])[0])
        correlations['spearman'].append(spearmanr(test_tsv[predictions_column(r)], test_tsv['labels'])[0])
        plot_path = os.path.join(output_dir, 'results_{}.png'.format(r))
        draw_scatterplot(test_tsv, 'labels', predictions_column(r), plot_path, config['model_type'] + ' ' + config['model_name'])

    preds_path = os.path.join(output_dir, 'results.tsv')
    test_tsv.to_csv(preds_path, header=True, sep='\t', index=False, encoding='utf-8')

    for corr in ('pearson', 'spearman'):
        print(corr)
        print(correlations[corr])
        print(np.mean(correlations[corr]))
        print(np.std(correlations[corr]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True)
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--train_features_path', required=False, default=None)
    parser.add_argument('--test_features_path', required=False, default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--test_size', default=0.1, type=float)
    args = parser.parse_args()
    config = load_config(args)
    train_set = DatasetSentLevel(config, evaluate=False)
    test_set = DatasetSentLevel(config, evaluate=True)
    train_data = train_set.make_dataset(args.train_path, )
    test_data = test_set.make_dataset(args.test_path, features_path=args.train_features_path)
    test_tsv = test_set.read(args.test_path, features_path=args.test_features_path)
    train_cycle(train_data, test_data, test_tsv, config, args.output_dir, args.test_size)


if __name__ == '__main__':
    main()
