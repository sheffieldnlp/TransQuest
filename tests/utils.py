import os

from transquest.data.load_config import load_config

test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, '../data')


class Args:

    def __init__(self, config, output_dir=None, model_dir=None):
        self.config = config
        self.output_dir = output_dir
        self.model_dir = model_dir


class DataSent:

    train_classif_tsv = os.path.join(data_dir, 'toy', 'toy.classif.tsv')
    test_classif_tsv = os.path.join(data_dir, 'toy', 'toy.classif.tsv')
    train_tsv = os.path.join(data_dir, 'toy', 'toy.sien.tsv')
    test_tsv = os.path.join(data_dir, 'toy', 'toy.sien.tsv')
    config_path = os.path.join(data_dir, 'toy', 'toy.json')
    features_pref = os.path.join(data_dir, 'toy', 'features')
    out_dir = os.path.join(data_dir, 'toy', 'output')
    args = Args(config_path, output_dir=out_dir)


class DataWord:

    tags_txt = os.path.join(data_dir, 'toy-word-level', 'toy.tags')
    src_txt = os.path.join(data_dir, 'toy-word-level', 'toy.src')
    tgt_txt = os.path.join(data_dir, 'toy-word-level', 'toy.mt')
    config_path = os.path.join(data_dir, 'toy-word-level', 'toy.json')
    features_path = os.path.join(data_dir, 'toy-word-level', 'toy.word_probas')
    mt_path = os.path.join(data_dir, 'toy-word-level', 'toy.mt_out')
    out_dir = os.path.join(data_dir, 'toy-word-level', 'output')
    args = Args(config_path, output_dir=out_dir)


class DataWordTest(DataWord):

    args = Args(DataWord.config_path, model_dir=os.path.join(data_dir, 'toy-word-level', 'model'))
