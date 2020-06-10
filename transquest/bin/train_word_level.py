import argparse
import os

from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetWordLevel
from transquest.bin.util import train_model, evaluate_model


def train_cycle(train, test, config, test_size):
    train_model(train, config, test_size=test_size)
    model_outputs = evaluate_model(test, config, run=0)
    return model_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--train_features_paths', nargs='+', default=None, required=False)
    parser.add_argument('--test_features_paths', nargs='+', default=None, required=False)
    parser.add_argument('--train_mt_path', default=None, required=False)
    parser.add_argument('--test_mt_path', default=None, required=False)
    parser.add_argument('--test_size', default=0.1, type=float)
    args = parser.parse_args()
    config = load_config(args)
    train_set = DatasetWordLevel(config, evaluate=False)
    test_set = DatasetWordLevel(config, evaluate=True)
    train_set.make_dataset(
        os.path.join(args.data_dir, 'train', 'train.src'),
        os.path.join(args.data_dir, 'train', 'train.mt'),
        os.path.join(args.data_dir, 'train', 'train.tags'),
        features_path=args.train_features_paths,
        mt_path=args.train_mt_path,
    )
    test_set.make_dataset(
        os.path.join(args.data_dir, 'dev', 'dev.src'),
        os.path.join(args.data_dir, 'dev', 'dev.mt'),
        os.path.join(args.data_dir, 'dev', 'dev.tags'),
        features_path=args.test_features_paths,
        mt_path=args.test_mt_path,
    )
    train_cycle(train_set.tensor_dataset, test_set.tensor_dataset, config, args.test_size)


if __name__ == '__main__':
    main()
