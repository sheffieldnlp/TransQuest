import argparse
import os


import torch

from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetWordLevel
from transquest.bin.util import train_model
from transquest.algo.transformers.run_model import QuestModel


def train_cycle(train, test, config, test_size):
    train_model(train, config, test_size=test_size)
    model = QuestModel(config["model_type"], config["best_model_dir"], use_cuda=torch.cuda.is_available(), args=config)
    model.eval_model(test)


def main(args):
    print(args)
    config = load_config(args)
    train_set = DatasetWordLevel(config, evaluate=False)
    test_set = DatasetWordLevel(config, evaluate=True)
    train_set.make_dataset(
        os.path.join(args.data_dir, "train", "train.src"),
        os.path.join(args.data_dir, "train", "train.src_tags"),
        os.path.join(args.data_dir, "train", "train.mt"),
        os.path.join(args.data_dir, "train", "train.tags"),
        features_path=args.train_features_paths,
        raw_mt_path=args.train_mt_path,
        wmt_format=True,
    )
    test_set.make_dataset(
        os.path.join(args.data_dir, "dev", "dev.src"),
        os.path.join(args.data_dir, "dev", "dev.src_tags"),
        os.path.join(args.data_dir, "dev", "dev.mt"),
        os.path.join(args.data_dir, "dev", "dev.tags"),
        features_path=args.test_features_paths,
        raw_mt_path=args.test_mt_path,
        wmt_format=True,
    )
    train_cycle(train_set.tensor_dataset, test_set.tensor_dataset, config, args.test_size)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_features_paths", nargs="+", default=None, required=False)
    parser.add_argument("--test_features_paths", nargs="+", default=None, required=False)
    parser.add_argument("--train_raw_mt_path", default=None, required=False)
    parser.add_argument("--test_raw_mt_path", default=None, required=False)
    parser.add_argument("--test_size", default=0.1, type=float)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
