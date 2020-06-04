import argparse

from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetSentLevel
from transquest.bin.util import train_cycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True)
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--lang_pair', required=True)
    parser.add_argument('--train_features_path', required=False, default=None)
    parser.add_argument('--test_features_path', required=False, default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--test_size', default=0.1, type=float)
    args = parser.parse_args()
    config = load_config(args)
    train_set = DatasetSentLevel(config, evaluate=False)
    train_set.make_dataset(args.train_path, features_path=args.train_features_path)

    test_sets = dict()
    test_sets[args.lang_pair] = DatasetSentLevel(config, evaluate=True)
    test_sets[args.lang_pair].make_dataset(args.test_path, features_path=args.test_features_path)

    train_cycle(train_set, test_sets, config, args.output_dir, args.test_size)


if __name__ == '__main__':
    main()
