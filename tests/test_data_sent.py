import os

import unittest

from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetSentLevel

from tests.utils import DataSent as d

test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, '../data')


class TestDataSent(unittest.TestCase):

    config = load_config(d.args)

    def test_reads_data(self):
        train_set = DatasetSentLevel(self.config)
        train_set.read(d.train_tsv)
        assert len(train_set.df) == 9

    def test_reads_data_with_injected_features(self):
        dataset = DatasetSentLevel(self.config)
        dataset.read(d.train_tsv, features_path='{}.train.tsv'.format(d.features_pref))
        assert dataset.df.shape == (9, 5)

    def test_loads_examples(self):
        dataset = DatasetSentLevel(self.config)
        dataset.read(d.train_tsv)
        dataset.load_examples()
        assert len(dataset.examples) == 9

    def test_loads_examples_with_features(self):
        dataset = DatasetSentLevel(self.config)
        dataset.read(d.test_tsv, features_path='{}.test.tsv'.format(d.features_pref))
        dataset.load_examples()
        assert len(dataset.examples) == 9
        for ex in dataset.examples:
            assert ex.features_inject['feature1'] == 0.2
            assert ex.features_inject['feature2'] == 0.5

    def test_loads_and_caches_examples_with_features(self):
        dataset = DatasetSentLevel(self.config)
        dataset.make_dataset(d.train_tsv, features_path='{}.train.tsv'.format(d.features_pref))
        assert len(dataset.tensor_dataset.tensors) == 5
        assert dataset.tensor_dataset.tensors[4].shape == (9, 2)


if __name__ == '__main__':
    unittest.main()
