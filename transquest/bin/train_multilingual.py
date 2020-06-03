import argparse
import os

from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetSentLevel
from transquest.bin.util import train_cycle


def build_paths(formattable_main, lang_pair, formattable_features=None):
    path = formattable_main.format(lang_pair)
    assert os.path.exists(path)
    features_path = None
    if formattable_features is not None:
        features_path = formattable_features.format(lang_pair)
    return path, features_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True)
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--train_features_path', required=False, default=None)
    parser.add_argument('--test_features_path', required=False, default=None)
    parser.add_argument('--lang_pairs', nargs='+', required=False, default=None)
    parser.add_argument('--test_lang_pair', required=False, default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--test_size', default=0.1, type=float)
    args = parser.parse_args()
    config = load_config(args)
    train_set = DatasetSentLevel(config, evaluate=False)
    examples = []
    for lang_pair in args.lang_pairs:
        train_path, train_features_path = build_paths(args.train_path, lang_pair, args.train_features_path)
        train_tsv = train_set.read(train_path, features_path=train_features_path)
        examples.extend(train_set.load_examples(train_tsv))
    train_data = train_set.make_tensors(examples)

    test_set = DatasetSentLevel(config, evaluate=True)
    test_path, test_features_path = build_paths(args.test_path, args.test_lang_pair, args.test_features_path)
    test_data = test_set.make_dataset(test_path, features_path=test_features_path)
    test_tsv = test_set.read(test_path, features_path=test_features_path)

    train_cycle(train_data, test_data, test_tsv, config, args.output_dir, args.test_size)


if __name__ == '__main__':
    main()
