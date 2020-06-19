import numpy as np
import pandas as pd
import torch
import os
import shutil

from copy import deepcopy
from collections import defaultdict

from sklearn.model_selection import train_test_split

from transquest.algo.transformers.run_model import QuestModel
from transquest.util.draw import draw_scatterplot
from transquest.data.normalizer import un_fit


def count_parameters(model):
    n_param = 0
    for p in model.parameters():
        if p.requires_grad:
            n_param += p.numel()
    return n_param


def train_model(train_set, config, n_fold=None, test_size=None):
    config['running_seed'] = config['SEED'] * n_fold if n_fold is not None else config['SEED']
    model = QuestModel(config['model_type'], config['model_name'], use_cuda=torch.cuda.is_available(), args=config)
    print('Trainable parameters: {}'.format(count_parameters(model.model)))
    if test_size:
        train_n, eval_df_n = train_test_split(train_set, test_size=test_size, random_state=config['running_seed'])
    else:
        train_n = train_set
        eval_df_n = None
    model.train_model(train_n, eval_df=eval_df_n)
    return model


def evaluate_model(test_sets, config, run, model=None):
    if model is None:
        model = QuestModel(config['model_type'], config['best_model_dir'], use_cuda=torch.cuda.is_available(), args=config)
    for lang_pair, test_set in test_sets.items():
        _, model_outputs = model.eval_model(test_set.tensor_dataset)
        test_set.df[_preds_col(run)] = model_outputs
    return test_sets


def _preds_col(fold):
    return 'preds_{}'.format(fold)


def train_cycle(train_set, test_sets, config):
    test_size = config['test_size'] if config['evaluate_during_training'] else None
    config_copy = deepcopy(config)
    model = None
    for i in range(config['n_fold']):
        print('Training with N folds. Now N is {}'.format(i))
        fold = i
        if os.path.exists(config['output_dir']) and os.path.isdir(config['output_dir']):
            shutil.rmtree(config['output_dir'])
        config_copy['best_model_dir'] = config['best_model_dir'] + '.{}'.format(fold)
        model = train_model(train_set.tensor_dataset, config_copy, n_fold=i, test_size=test_size)
        test_sets = evaluate_model(test_sets, config_copy, fold)

    columns = list(sorted(model.metrics))
    index = list(sorted(test_sets))
    results_df = pd.DataFrame(columns=columns, index=index)

    for lang_pair in sorted(test_sets):
        test_set = test_sets[lang_pair]
        test_set.df = un_fit(test_set.df, 'labels')

        results = defaultdict(list)
        for r in range(config['n_fold']):
            test_set.df = un_fit(test_set.df, _preds_col(r))
            for metric in model.metrics:
                results[metric].append(model.metrics[metric](test_set.df[_preds_col(r)], test_set.df['labels']))
            if config['regression']:
                plot_path = os.path.join(config['output_dir'], 'results.{}.{}.png'.format(lang_pair, r))
                draw_scatterplot(test_set.df, 'labels', _preds_col(r), plot_path, config['model_type'] + ' ' + config['model_name'])

        preds_path = os.path.join(config['output_dir'], 'predictions.{}.tsv'.format(lang_pair))
        test_set.df.to_csv(preds_path, header=True, sep='\t', index=False, encoding='utf-8')

        for metric in sorted(results):
            results_df.at[lang_pair, metric] = '{:.3f}±{:.2f}'.format(np.mean(results[metric]), np.std(results[metric]))

    results_df.to_csv(os.path.join(config['output_dir'], 'results.tsv'), sep='\t')
