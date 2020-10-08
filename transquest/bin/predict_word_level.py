import argparse
import os
import sys

import torch

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from transquest.algo.transformers.evaluation import pearson_corr, spearman_corr
from transquest.algo.transformers.run_model import QuestModel

from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetWordLevel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--src_file', required=True)
    parser.add_argument('--tgt_file', required=True)
    parser.add_argument('--tags_file', required=True)
    parser.add_argument('--out_file', required=False, default=None)
    parser.add_argument('--output_dir', required=False, default=None)
    args = parser.parse_args()
    config = load_config(args)
    test_set = DatasetWordLevel(config, evaluate=True)
    test_data = test_set.make_dataset(
        os.path.join(args.src_file),
        os.path.join(args.tgt_file),
        os.path.join(args.tags_file),
        no_cache=True,
    )
    assert os.path.isdir(args.model_dir)
    model = QuestModel(config['model_type'], args.model_dir, use_cuda=torch.cuda.is_available(), args=config)
    _, preds = model.evaluate(test_data)
    out = open(args.out_file, 'w') if args.out_file is not None else sys.stdout
    for pred in preds:
        out.write('{}\n'.format(' '.join([str(p) for p in pred])))


if __name__ == '__main__':
    main()
