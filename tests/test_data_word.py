import os
import unittest

from transquest.data.mapping_tokens_bpe import map_tokens_bpe
from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetWordLevel

from transquest.algo.transformers.run_model import QuestModel

from tests.utils import DataWord as d

test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, '../data')


class TestDataSent(unittest.TestCase):

    config = load_config(d.args)

    def test_reads_data(self):
        dataset = DatasetWordLevel(self.config)
        src, tgt, labels = dataset.read(d.src_txt, d.tgt_txt, d.tags_txt)
        assert len(src) == len(tgt) == len(labels)
        for src_i, tgt_i, labels_i in zip(src, tgt, labels):
            assert len(tgt_i.split()) == len(labels_i)

    def test_loads_examples(self):
        dataset = DatasetWordLevel(self.config)
        src, tgt, labels = dataset.read(d.src_txt, d.tgt_txt, d.tags_txt)
        examples = dataset.load_examples(src, tgt, labels)
        assert len(examples) == len(src)

    def test_makes_dataset(self):
        dataset = DatasetWordLevel(self.config)
        train = dataset.make_dataset(d.src_txt, d.tgt_txt, d.tags_txt)
        print(len(train))

    def test_maps_labels_to_bpe(self):
        tokens = '1934 besuchte José Ortega y Gasset Husserl in Freiburg .'.split()
        labels = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        pieces = ['▁1934', '▁besucht', 'e', '▁José', '▁Ort', 'ega', '▁y', '▁G', 'asset', '▁Hus', 'ser', 'l', '▁in', '▁Freiburg', '▁', '.']
        labels = map_tokens_bpe(tokens, pieces, labels)
        assert len(labels) == len(pieces)
