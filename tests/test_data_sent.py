import os

import unittest

from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetSentLevel
from transquest.algo.transformers.run_model import QuestModel

from tests.utils import DataSent as d

test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, '../data')


class TestDataSent(unittest.TestCase):

    config = load_config(d.args)

    def test_reads_data(self):
        dataset = DatasetSentLevel(self.config)
        train_df = dataset.read(d.train_tsv)
        test_df = dataset.read(d.test_tsv)
        assert len(train_df) == 9
        assert len(test_df) == 9

    def test_reads_data_with_injected_features(self):
        dataset = DatasetSentLevel(self.config)
        train_df = dataset.read(d.train_tsv, features_path='{}.train.tsv'.format(d.features_pref))
        test_df = dataset.read(d.test_tsv, features_path='{}.test.tsv'.format(d.features_pref))
        assert train_df.shape == (9, 5)
        assert test_df.shape == (9, 5)

    def test_loads_examples(self):
        dataset = DatasetSentLevel(self.config)
        train_df = dataset.read(d.train_tsv)
        examples = dataset.load_examples(train_df)
        assert len(examples) == 9

    def test_loads_examples_with_features(self):
        dataset = DatasetSentLevel(self.config)
        test_df = dataset.read(d.test_tsv, features_path='{}.test.tsv'.format(d.features_pref))
        examples = dataset.load_examples(test_df)
        assert len(examples) == 9
        for ex in examples:
            assert ex.features_inject['feature1'] == 0.2
            assert ex.features_inject['feature2'] == 0.5

    def test_loads_and_caches_examples_with_features(self):
        dataset = DatasetSentLevel(self.config)
        dataset = dataset.make_dataset(
            d.train_tsv,
            features_path='{}.train.tsv'.format(d.features_pref),
        )
        assert len(dataset.tensors) == 5
        assert dataset.tensors[4].shape == (9, 2)


if __name__ == '__main__':
    unittest.main()