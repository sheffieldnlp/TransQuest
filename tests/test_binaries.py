import os
import unittest

from argparse import Namespace

from tests.utils import data_dir

from transquest.bin.train_multilingual import main


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
        main(self.args)


if __name__ == '__main__':
    unittest.main()
