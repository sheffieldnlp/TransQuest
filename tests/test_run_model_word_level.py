import os
import unittest

from transquest.bin.train import train_model  # TODO: this method should be in a different place
from transquest.bin.train import train_cycle  # TODO: this method should be in a different place
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
        train = dataset.make_dataset(w.src_txt, w.tgt_txt, w.tags_txt)
        train_model(train, config, test_size=0.5)
