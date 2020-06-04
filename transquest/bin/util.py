import numpy as np
import os
import shutil

from copy import deepcopy

from scipy.stats import pearsonr, spearmanr

from collections import defaultdict

import torch

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from transquest.algo.transformers.evaluation import pearson_corr, spearman_corr
from transquest.algo.transformers.run_model import QuestModel
from transquest.util.draw import draw_scatterplot
from transquest.data.normalizer import un_fit


def train_model(train_set, config, n_fold=None, test_size=None, return_model=False):
    config['running_seed'] = config['SEED'] * n_fold if n_fold is not None else config['SEED']
    model = QuestModel(config['model_type'], config['model_name'], num_labels=1, use_cuda=torch.cuda.is_available(),
                       args=config)
    if test_size:
        train_n, eval_df_n = train_test_split(train_set, test_size=test_size, random_state=config['running_seed'])
    else:
        train_n = train_set
        eval_df_n = None
    if config['regression']:
        model.train_model(train_n, eval_df=eval_df_n, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                          mae=mean_absolute_error)
    else:
        model.train_model(train_n, eval_df=eval_df_n)
    if return_model:
        return model


def evaluate_model(test_sets, config, run, model=None):
    if model is None:
        model = QuestModel(config['model_type'], config['best_model_dir'], num_labels=1, use_cuda=torch.cuda.is_available(), args=config)
    metrics = {}
    if config['regression']:
        metrics = {'pearson_corr': pearson_corr, 'spearman_corr': spearman_corr, 'mae': mean_absolute_error}
    for lang_pair, test_set in test_sets.items():
        _, model_outputs = model.eval_model(test_set.tensor_dataset, **metrics)
        test_set.df[predictions_column(run)] = model_outputs
    return test_sets


def predictions_column(training_run):
    return 'predictions_{}'.format(training_run)


def train_cycle(train_set, test_sets, config, output_dir, test_size):
    run = 0
    if config['evaluate_during_training']:
        if config['n_fold'] > 1:
            config_copy = deepcopy(config)
            for i in range(config['n_fold']):
                print('Training with N folds. Now N is {}'.format(i))
                run = i
                if os.path.exists(config['output_dir']) and os.path.isdir(config['output_dir']):
                    shutil.rmtree(config['output_dir'])

                config_copy['best_model_dir'] = config['best_model_dir'] + '.{}'.format(run)
                train_model(train_set.tensor_dataset, config_copy, n_fold=i, test_size=test_size)
                test_sets = evaluate_model(test_sets, config_copy, run)
        else:
            train_model(train_set.tensor_dataset, config, test_size=test_size)
            test_sets = evaluate_model(test_sets, config, run)
    else:
        model = train_model(train_set.tensor_dataset, config, return_model=True)
        test_sets = evaluate_model(test_sets, config, run, model=model)

    runs = 1 if config['n_fold'] < 2 else config['n_fold']

    for lang_pair, test_set in test_sets.items():
        test_set.df = un_fit(test_set.df, 'labels')

        correlations = defaultdict(list)
        for r in range(runs):
            test_set.df = un_fit(test_set.df, predictions_column(r))
            correlations['pearson'].append(pearsonr(test_set.df[predictions_column(r)], test_set.df['labels'])[0])
            correlations['spearman'].append(spearmanr(test_set.df[predictions_column(r)], test_set.df['labels'])[0])
            plot_path = os.path.join(output_dir, 'results.{}.{}.png'.format(lang_pair, r))
            draw_scatterplot(test_set.df, 'labels', predictions_column(r), plot_path, config['model_type'] + ' ' + config['model_name'])

        preds_path = os.path.join(output_dir, 'results.{}.tsv'.format(lang_pair))
        test_set.df.to_csv(preds_path, header=True, sep='\t', index=False, encoding='utf-8')

        for corr in ('pearson', 'spearman'):
            print(lang_pair)
            print(corr)
            print(correlations[corr])
            print(np.mean(correlations[corr]))
            print(np.std(correlations[corr]))
