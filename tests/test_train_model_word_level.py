import os
import unittest

from transquest.bin.train import train_model  # TODO: this method should be in a different place
from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetWordLevel

from tests.utils import DataWord as w


test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, '../data')


class TestTrain(unittest.TestCase):

    config = load_config(w.args)

    def test_trains_model_word_level(self):
        config = load_config(w.args)
        config['model_type'] = 'xlmrobertatoken'
        dataset = DatasetWordLevel(self.config)
        dataset.make_dataset(w.src_txt, w.tgt_txt, w.tags_txt, wmt_format=True)
        train_model(dataset.tensor_dataset, config, test_size=0.5)

    def test_trains_model_word_level_with_features(self):
        config = load_config(w.args)
        config['model_type'] = 'xlmrobertatokeninject'
        dataset = DatasetWordLevel(self.config)
        dataset.make_dataset(w.src_txt, w.tgt_txt, w.tags_txt, features_path=[w.features_path], mt_path=w.mt_path, wmt_format=True)
        train_model(dataset.tensor_dataset, config, test_size=0.5)