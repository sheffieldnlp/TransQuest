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

from transquest.data.read_dataframe import read_data_files
from transquest.data.load_config import load_config
from transquest.data.normalizer import un_fit


def train_model(train_set, config, n_fold=None, test_size=None, return_model=False):
    config['running_seed'] = config['SEED'] * n_fold if n_fold is not None else config['SEED']
    model = QuestModel(config['MODEL_TYPE'], config['MODEL_NAME'], num_labels=1, use_cuda=torch.cuda.is_available(), args=config)
    if test_size:
        train_n, eval_df_n = train_test_split(train_set, test_size=test_size, random_state=config['running_seed'])
    else:
        train_n = train_set
        eval_df_n = None
    model.train_model(train_n, eval_df=eval_df_n, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
    if return_model:
        return model


def evaluate_model(test_set, config, model=None):
    if model is None:
        model = QuestModel(config['MODEL_TYPE'], config['best_model_dir'], num_labels=1, use_cuda=torch.cuda.is_available(),args=config)
    _, model_outputs, _ = model.eval_model(test_set, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
    return model_outputs


def predictions_column(training_run):
    return 'predictions_{}'.format(training_run)


def train_cycle(train, test, config, output_dir, test_size):
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
                test[predictions_column(run)] = model_outputs
        else:
            train_model(train, config, test_size=test_size)
            model_outputs = evaluate_model(test, config)
            test[predictions_column(run)] = model_outputs
    else:
        model = train_model(train, config, return_model=True)
        model_outputs = evaluate_model(test, config, model=model)
        test[predictions_column(run)] = model_outputs

    runs = 1 if config['n_fold'] < 2 else config['n_fold']
    correlations = defaultdict(list)

    test = un_fit(test, 'labels')

    for r in range(runs):
        test = un_fit(test, predictions_column(r))
        correlations['pearson'].append(pearsonr(test[predictions_column(r)], test['labels'])[0])
        correlations['spearman'].append(spearmanr(test[predictions_column(r)], test['labels'])[0])
        plot_path = os.path.join(output_dir, 'results_{}.png'.format(r))
        draw_scatterplot(test, 'labels', predictions_column(r), plot_path, config['MODEL_TYPE'] + ' ' + config['MODEL_NAME'])

    preds_path = os.path.join(output_dir, 'results.tsv')
    test.to_csv(preds_path, header=True, sep='\t', index=False, encoding='utf-8')

    for corr in ('pearson', 'spearman'):
        print(corr)
        print(correlations[corr])
        print(np.mean(correlations[corr]))
        print(np.std(correlations[corr]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True)
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--features_pref', default=None, required=False)
    parser.add_argument('--test_size', default=0.1, type=float)
    args = parser.parse_args()
    config = load_config(args)
    train, test = read_data_files(args.train_path, args.test_path, features_pref=args.features_pref)
    train_cycle(train, test, config, args.output_dir, args.test_size)


if __name__ == '__main__':
    main()
