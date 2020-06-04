import os
import argparse
import torch

from transquest.algo.transformers.run_model import QuestModel
from transquest.algo.transformers.evaluation import pearson_corr, spearman_corr
from sklearn.metrics import mean_absolute_error

from transquest.data.normalizer import un_fit
from transquest.data.dataset import DatasetSentLevel
from transquest.util.draw import draw_scatterplot

from transquest.data.load_config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_file')
    parser.add_argument('-m', '--model_dir')
    parser.add_argument('-o', '--output_dir')
    parser.add_argument('-c', '--config')
    parser.add_argument('--features_path', default=None, required=False)
    args = parser.parse_args()

    config = load_config(args)
    model = QuestModel(
        config['model_type'], args.model_dir, num_labels=1, use_cuda=torch.cuda.is_available(), args=config
    )
    test_set = DatasetSentLevel(config, evaluate=True)
    test_set.make_dataset(args.test_file)
    result, model_outputs = model.eval_model(
        test_set.tensor_dataset, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
    test_set.df['predictions'] = model_outputs
    test_set.df = un_fit(test_set.df, 'labels')
    test_set.df = un_fit(test_set.df, 'predictions')

    out_preds = os.path.join(args.output_dir, 'predictions')
    out_preds = '{}.{}.{}'.format(out_preds, config['model_type'], config['model_name'])
    out_scatter = os.path.join(args.output_dir, 'scatter')
    out_scatter = '{}.{}.{}'.format(out_scatter, config['model_type'], config['model_name'])
    test_set.df.to_csv('{}.tsv'.format(out_preds), header=True, sep='\t', index=False, encoding='utf-8')
    draw_scatterplot(
        test_set.df, 'labels', 'predictions', '{}.png'.format(out_scatter), config['model_type'] + ' ' + config['model_name'])


if __name__ == '__main__':
    main()
