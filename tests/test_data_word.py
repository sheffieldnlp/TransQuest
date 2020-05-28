import os
import unittest

from transquest.data.read_plain import read_data
from transquest.data.mapping_tokens_bpe import map_tokens_bpe
from transquest.data.read_plain import load_examples
from transquest.data.make_dataset import make_dataset
from transquest.data.load_config import load_config

from transquest.algo.transformers.run_model import QuestModel

from tests.utils import Args

test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, '../data')


class TestDataSent(unittest.TestCase):

    tags_txt = os.path.join(data_dir, 'toy-word-level', 'toy.tags')
    src_txt = os.path.join(data_dir, 'toy-word-level', 'toy.src')
    tgt_txt = os.path.join(data_dir, 'toy-word-level', 'toy.mt')
    config_path = os.path.join(data_dir, 'toy-word-level', 'toy.json')
    out_dir = os.path.join(data_dir, 'toy-word-level', 'output')
    args = Args(config_path, out_dir)

    def test_reads_data(self):
        src, tgt, labels = read_data(self.src_txt, self.tgt_txt, self.tags_txt)
        assert len(src) == len(tgt) == len(labels)
        for src_i, tgt_i, labels_i in zip(src, tgt, labels):
            assert len(tgt_i.split()) == len(labels_i)

    def test_loads_examples(self):
        src, tgt, labels = read_data(self.src_txt, self.tgt_txt, self.tags_txt)
        examples = load_examples(src, tgt, labels)
        assert len(examples) == len(src)

    def test_makes_dataset(self):
        src, tgt, labels = read_data(self.src_txt, self.tgt_txt, self.tags_txt)
        examples = load_examples(src, tgt, labels)
        config = load_config(self.args)
        model = QuestModel(config['MODEL_TYPE'], config['MODEL_NAME'], args=config, use_cuda=False)
        make_dataset(examples, model.tokenizer, model.args)

    def test_maps_labels_to_bpe(self):
        tokens = '1934 besuchte José Ortega y Gasset Husserl in Freiburg .'.split()
        labels = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        pieces = ['▁1934', '▁besucht', 'e', '▁José', '▁Ort', 'ega', '▁y', '▁G', 'asset', '▁Hus', 'ser', 'l', '▁in', '▁Freiburg', '▁', '.']
        labels = map_tokens_bpe(tokens, pieces, labels)
        assert len(labels) == len(pieces)
