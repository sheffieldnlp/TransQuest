import os
import unittest

from transquest.bin.train import train_model  # TODO: this method should be in a different place
from transquest.bin.train import train_cycle  # TODO: this method should be in a different place
from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetSentLevel

from tests.utils import DataSent as d


test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, '../data')


class TestTrain(unittest.TestCase):

    def test_trains_model_sent_level_classification(self):
        config = load_config(d.args)
        config['model_type'] = 'xlmroberta'
        config['regression'] = False
        config['num_labels'] = 3
        dataset = DatasetSentLevel(config, evaluate=False)
        dataset.make_dataset(d.train_classif_tsv)
        train_model(dataset.tensor_dataset, config, test_size=0.5)

    def test_trains_model_sent_level_regression(self):
        config = load_config(d.args)
        config['model_type'] = 'xlmroberta'
        dataset = DatasetSentLevel(config, evaluate=False)
        dataset.make_dataset(d.train_tsv)
        train_model(dataset.tensor_dataset, config, test_size=0.5)

    def test_trains_model_with_injected_features(self):
        config = load_config(d.args)
        config['model_type'] = 'xlmrobertainject'
        config['feature_combination'] = 'concat'
        dataset = DatasetSentLevel(config, evaluate=False)
        dataset.make_dataset(d.train_tsv, features_path='{}.train.tsv'.format(d.features_pref))
        train_model(dataset.tensor_dataset, config, test_size=0.5)

    def test_trains_model_with_injected_features_with_reduce(self):
        config = load_config(d.args)
        config['model_type'] = 'xlmrobertainject'
        config['feature_combination'] = 'reduce'
        dataset = DatasetSentLevel(config, evaluate=False)
        dataset.make_dataset(d.train_tsv, features_path='{}.train.tsv'.format(d.features_pref))
        train_model(dataset.tensor_dataset, config, test_size=0.5)

    def test_trains_model_with_injected_features_with_conv(self):
        config = load_config(d.args)
        config['model_type'] = 'xlmrobertainject'
        config['feature_combination'] = 'conv'
        dataset = DatasetSentLevel(config, evaluate=False)
        dataset.make_dataset(d.train_tsv, features_path='{}.train.tsv'.format(d.features_pref))
        train_model(dataset.tensor_dataset, config, test_size=0.5)

    def test_runs_training_cycle(self):
        config = load_config(d.args)
        config['model_type'] = 'xlmrobertainject'
        config['n_fold'] = 2
        config['test_size'] = 0.5

        train_set = DatasetSentLevel(config, evaluate=False)
        train_set.make_dataset(d.train_tsv, features_path='{}.train.tsv'.format(d.features_pref))

        test_sets = dict()
        test_sets['sien'] = DatasetSentLevel(config, evaluate=True)
        test_sets['sien'].make_dataset(d.test_tsv, features_path='{}.test.tsv'.format(d.features_pref))

        train_cycle(train_set, test_sets, config)
