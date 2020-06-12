import os
import unittest

from argparse import Namespace

from tests.utils import data_dir

from transquest.bin.train_multilingual import main as main_multilingual
from transquest.bin.train_word_level import main as main_word_level


class TestTrainMulti(unittest.TestCase):

    args = Namespace(
        train_path=os.path.join(data_dir, 'toy', 'toy.{}.tsv'),
        test_path=os.path.join(data_dir, 'toy', 'toy.{}.tsv'),
        train_features_path=None,
        test_features_path=None,
        lang_pairs=['sien', 'neen'],
        output_dir=os.path.join(data_dir, 'toy', 'output'),
        config=os.path.join(data_dir, 'toy', 'toy.json'),
        test_size=0.1,
    )

    def test_trains_model_with_multiple_lang_pairs(self):
        main_multilingual(self.args)


class TestTrainWord(unittest.TestCase):

    args = Namespace(
        data_dir=os.path.join(data_dir, 'toy-word-level'),
        output_dir=os.path.join(data_dir, 'toy-word-level', 'output'),
        config=os.path.join(data_dir, 'toy-word-level', 'toy.json'),
        train_features_paths=None,
        test_features_paths=None,
        train_mt_path=None,
        test_mt_path=None,
        test_size=0.5,
    )

    def test_trains_model_for_word_level(self):
        main_word_level(self.args)

    def test_trains_model_for_word_level_with_features(self):
        self.args.config = os.path.join(data_dir, 'toy-word-level', 'toy.wfeatures.json')
        self.args.train_features_paths = [
            os.path.join(data_dir, 'toy-word-level', 'toy.feature1'),
            os.path.join(data_dir, 'toy-word-level', 'toy.feature2')
        ]
        self.args.test_features_paths = [
            os.path.join(data_dir, 'toy-word-level', 'toy.feature1'),
            os.path.join(data_dir, 'toy-word-level', 'toy.feature2')
        ]
        self.args.train_mt_path = os.path.join(data_dir, 'toy-word-level', 'toy.mt_out')
        self.args.test_mt_path = os.path.join(data_dir, 'toy-word-level', 'toy.mt_out')
        main_word_level(self.args)


if __name__ == '__main__':
    unittest.main()
