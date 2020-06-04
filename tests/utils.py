import os

from transquest.data.load_config import load_config

test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, '../data')


class Args:

    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir


class DataSent:

    train_tsv = os.path.join(data_dir, 'toy', 'toy.sien.tsv')
    test_tsv = os.path.join(data_dir, 'toy', 'toy.sien.tsv')
    config_path = os.path.join(data_dir, 'toy', 'toy.json')
    features_pref = os.path.join(data_dir, 'toy', 'features')
    out_dir = os.path.join(data_dir, 'toy', 'output')
    args = Args(config_path, out_dir)


class DataWord:

    tags_txt = os.path.join(data_dir, 'toy-word-level', 'toy.tags')
    src_txt = os.path.join(data_dir, 'toy-word-level', 'toy.src')
    tgt_txt = os.path.join(data_dir, 'toy-word-level', 'toy.mt')
    config_path = os.path.join(data_dir, 'toy-word-level', 'toy.json')
    out_dir = os.path.join(data_dir, 'toy-word-level', 'output')
    args = Args(config_path, out_dir)
