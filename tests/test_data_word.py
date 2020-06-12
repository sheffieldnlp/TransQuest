import os
import unittest

from transquest.data.mapping_tokens_bpe import map_pieces
from transquest.data.load_config import load_config
from transquest.data.dataset import DatasetWordLevel
from transquest.algo.model_classes import model_classes

from transquest.algo.model_classes import XLMRobertaTokenizer

from tests.utils import DataWord as d

test_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(test_dir, '../data')


class TestDataSent(unittest.TestCase):

    config = load_config(d.args)

    def test_reads_data(self):
        dataset = DatasetWordLevel(self.config)
        src, tgt, labels, features, mt_out = dataset.read(d.src_txt, d.tgt_txt, d.tags_txt)
        assert len(src) == len(tgt) == len(labels)
        for src_i, tgt_i, labels_i in zip(src, tgt, labels):
            assert len(tgt_i.split()) == len(labels_i)

    def test_loads_examples(self):
        dataset = DatasetWordLevel(self.config)
        src, tgt, labels, features, mt_out = dataset.read(d.src_txt, d.tgt_txt, d.tags_txt)
        dataset.load_examples(src, tgt, labels)
        assert len(dataset.examples) == len(src)

    def test_makes_dataset(self):
        dataset = DatasetWordLevel(self.config)
        dataset.make_dataset(d.src_txt, d.tgt_txt, d.tags_txt)
        print(len(dataset.tensor_dataset.tensors))

    def test_makes_dataset_with_features(self):
        dataset = DatasetWordLevel(self.config)
        dataset.make_dataset(d.src_txt, d.tgt_txt, d.tags_txt, [d.features_path], d.mt_path)
        assert dataset.tensor_dataset.tensors[4].shape == (5, 1, 128)

    def test_maps_labels_to_bpe(self):
        tokens = '1934 besuchte José Ortega y Gasset Husserl in Freiburg .'.split()
        labels = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        pieces = ['▁1934', '▁besucht', 'e', '▁José', '▁Ort', 'ega', '▁y', '▁G', 'asset', '▁Hus', 'ser', 'l', '▁in', '▁Freiburg', '▁', '.']
        labels = map_pieces(tokens, pieces, labels, 'first')
        assert len(labels) == len(pieces)

    def test_maps_labels_to_bpe_chinese(self):
        tokens = "2018 年 7 月 ， 法拉奇为 2018 年宾夕法尼亚州美国参议院选举的共和党候选人卢 · 巴莱塔 （ Lou Barletta ） 担任了筹款人。"
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=False)
        pretrained_pieces = tokenizer.tokenize(tokens)
        tokens = tokens.split()
        labels = [1] * len(tokens)
        result = map_pieces(tokens, pretrained_pieces, labels, 'first')
        assert len(result) == len(pretrained_pieces)

    def test_maps_probas_to_bpe_chinese(self):
        mt = "印度教 和 佛教 的 许多 最 广泛 的 咒语 源自 于 神灵 的 召唤 ， 例如 ： @ @"
        raw_mt_tokens = "印度@@ 教@@ 和@@ 佛@@ 教@@ 的@@ 许多@@ 最@@ 广泛@@ 的@@ 咒@@ 语@@ 源@@ 自@@ 于@@ 神@@ 灵@@ 的@@ 召@@ 唤@@ ，@@ 例如@@ ：@@"
        _, _, tokenizer_class = model_classes['xlmroberta']
        tokenizer = tokenizer_class.from_pretrained('xlm-roberta-base', do_lower_case=False)
        pretrained_pieces = tokenizer.tokenize(mt)
        raw_mt_tokens = raw_mt_tokens.split()
        probas = [0.1] * len(raw_mt_tokens)
        result = map_pieces(raw_mt_tokens, pretrained_pieces, probas, 'average')
        assert len(result) == len(pretrained_pieces)

    def test_maps_probas_to_bpe(self):
        mt_pieces = '1934 besuchte José Ort@@ ega y G@@ asset Hus@@ ser@@ l in Freiburg .'.split()
        pretrained_pieces = ['▁1934', '▁besucht', 'e', '▁José', '▁Ort', 'ega', '▁y', '▁G', 'asset', '▁Hus', 'ser', 'l', '▁in', '▁Freiburg', '▁', '.']
        probas = [-0.4458, -0.2745, -0.0720, -0.0023, -0.0059, -0.1458, -0.0750, -0.0124, -0.0269, -0.0364, -0.0530, -0.1499, -0.0124, -0.1145, -0.1100]
        result = map_pieces(mt_pieces, pretrained_pieces, probas[:-1], 'average', from_sep='@@')  # ignore <eos>
        assert len(result) == len(pretrained_pieces)
