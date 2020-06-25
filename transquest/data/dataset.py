import os
import numpy as np
import torch
import pandas as pd

from collections import OrderedDict

from multiprocessing import cpu_count
from multiprocessing import Pool
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset

from transquest.data.normalizer import fit
from transquest.data.containers import InputExampleSent
from transquest.data.containers import InputExampleWord
from transquest.data.containers import InputFeatures
from transquest.data.mapping_tokens_bpe import map_pieces

from transquest.algo.model_classes import model_classes

DEFAULT_FEATURE_NAME = 'feature'


class Dataset:

    def __init__(self, config, evaluate=False):
        self.config = config
        self.max_seq_length = config['max_seq_length']
        self.output_mode = 'regression' if config['regression'] else 'classification'
        self.cls_token_at_end = bool(config['model_type'] in ['xlnet'])
        self.pad_on_left = bool(config['model_type'] in ['xlnet'])
        self.cls_token_segment_id = 2 if config['model_type'] in ['xlnet'] else 0
        self.pad_token_segment_id = 4 if config['model_type'] in ['xlnet'] else 0
        self.process_count = cpu_count() - 2
        self.silent = config['silent']
        self.use_multiprocessing = config['use_multiprocessing']
        self.mask_padding_with_zero = True
        self.flatten = False
        self.stride = None
        self.evaluate = evaluate

        _, _, tokenizer_class = model_classes[config['model_type']]
        self.tokenizer = tokenizer_class.from_pretrained(config['model_name'], do_lower_case=self.config['do_lower_case'])
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.sep_token_extra = False

        self.df = None
        self.examples = None
        self.tensor_dataset = None

    def read(self, **kwargs):
        pass

    def load_examples(self, **kwargs):
        pass

    def make_tensors(self, no_cache=False, verbose=True):
        if not no_cache:
            no_cache = self.config['no_cache']

        os.makedirs(self.config['cache_dir'], exist_ok=True)

        mode = 'dev' if self.evaluate else 'train'
        cached_features_file = os.path.join(
            self.config['cache_dir'],
            'cached_{}_{}_{}_{}_{}'.format(
                mode, self.config['model_type'], self.max_seq_length, self.config['num_labels'], len(self.examples),
            ),
        )

        if os.path.exists(cached_features_file) and (
                (not self.config['reprocess_input_data'] and not no_cache) or (
                mode == 'dev' and self.config['use_cached_eval_features'] and not no_cache)
        ):
            features = torch.load(cached_features_file)
            if verbose:
                print(f"Features loaded from cache at {cached_features_file}")
        else:
            if verbose:
                print(f"Converting to features started. Cache is not used.")
            features = self._convert_examples_to_features(self.examples)
            if not no_cache:
                torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_features = None
        if features[0].features_inject:
            all_features = self._injected_features_to_tensor(features=features, max_len=self.max_seq_length)

        label_torch_type = torch.long if self.output_mode == 'classification' else torch.float
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=label_torch_type)

        tensors = [all_input_ids, all_input_mask, all_segment_ids, all_label_ids]
        if all_features is not None:
            tensors.append(all_features)
        self.tensor_dataset = TensorDataset(*tensors)

    def _convert_examples_to_features(self, examples):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        if self.use_multiprocessing:
            with Pool(self.process_count) as p:
                features = list(tqdm(p.imap(self._convert_example_to_feature, examples, chunksize=500), total=len(examples),
                                     disable=self.silent))
        else:
            features = [self._convert_example_to_feature(example) for example in tqdm(examples, disable=self.silent)]
        return features

    def _convert_example_to_feature(self, example, pad_token=0, sequence_a_segment_id=0, sequence_b_segment_id=1,):
        tokens_a = self.tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if self.sep_token_extra else 3
            self._truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if self.sep_token_extra else 2
            if len(tokens_a) > self.max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(self.max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [self.sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [self.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if self.cls_token_at_end:
            tokens = tokens + [self.cls_token]
            segment_ids = segment_ids + [self.cls_token_segment_id]
        else:
            tokens = [self.cls_token] + tokens
            segment_ids = [self.cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)

        input_ids = self._pad(input_ids, [pad_token] * padding_length)
        input_mask = self._pad(input_mask, [0 if self.mask_padding_with_zero else 1] * padding_length)
        segment_ids = self._pad(segment_ids, [self.pad_token_segment_id] * padding_length)

        label = self._map_labels(example=example, bpe_tokens=tokens_a, padding=([pad_token] * padding_length))
        features_inject = self._map_features(
            example=example, bpe_tokens=tokens_a, padding=([pad_token] * padding_length))

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        return InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label,
            features_inject=features_inject,
        )

    def _pad(self, seq, padding):
        if self.pad_on_left:
            return padding + seq
        else:
            return seq + padding

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _map_labels(self, **kwargs):
        return

    def _map_features(self, **kwargs):
        return

    def _injected_features_to_tensor(self, **kwargs):
        return


class DatasetWordLevel(Dataset):

    def __init__(self, config, evaluate=False):
        super().__init__(config, evaluate=evaluate)
        self.examples = None
        self.tensors = None

    def make_dataset(self, src_path, tgt_path, labels_path, features_path=None, mt_path=None, no_cache=False, verbose=True):
        src, tgt, labels, features, mt_out = self.read(src_path, tgt_path, labels_path, features_path=features_path, mt_path=mt_path)
        self.load_examples(src, tgt, labels, features, mt_out)
        self.make_tensors(no_cache=no_cache, verbose=verbose)

    def load_examples(self, src, tgt, labels, features=None, mt_out=None):
        self.examples = [
            InputExampleWord(guid=i, text_a=text_b, label=label)
            for i, (text_a, text_b, label) in enumerate(
                zip(src, tgt, labels)
            )
        ]
        if features is not None:
            for feature_name in features:
                for i, ex in enumerate(self.examples):
                    ex.features_inject[feature_name] = features[feature_name][i]
                    ex.mt_tokens = mt_out[i]

    def read(self, src_path, tgt_path, labels_path, features_path=None, mt_path=None):
        labels = self._read_labels(labels_path)
        src = [l.strip() for l in open(src_path)]
        tgt = [l.strip() for l in open(tgt_path)]
        mt_out = None
        if mt_path is not None:
            mt_out = [l.strip().split() for l in open(mt_path)]
        features = dict()
        if features_path is not None:
            for i, path in enumerate(features_path, start=1):
                features['{}{}'.format(DEFAULT_FEATURE_NAME, i)] = [[float(s) for s in l.split()[:-1]] for l in open(path)]
        return src, tgt, labels, features, mt_out

    @staticmethod
    def _read_labels(path):
        labels = []
        for line in open(path):
            tags = line.strip().split()
            tags = [1 if t == 'OK' else 0 for t in tags]
            labels_i = [t for i, t in enumerate(tags) if i % 2 != 0]
            assert len(labels_i) * 2 + 1 == len(tags)
            labels.append(labels_i)
        return labels

    def _map_labels(self, example, bpe_tokens, padding):
        labelled_tokens = example.text_a.split()
        labels = example.label
        labels = map_pieces(labelled_tokens, bpe_tokens, labels, method='first')
        labels = labels + [0]  # sep token
        return self._pad(labels + [0] if self.cls_token_at_end else [0] + labels, padding)

    def _map_features(self, example, bpe_tokens, padding):
        mapped = OrderedDict()
        for feature in example.features_inject:
            mapped_f = map_pieces(
                example.mt_tokens, bpe_tokens, example.features_inject[feature], method='average', from_sep='@@')
            mapped_f = mapped_f + [0]
            mapped[feature] = self._pad(mapped_f + [0] if self.cls_token_at_end else [0] + mapped_f, padding)
        return mapped

    def _injected_features_to_tensor(self, features, max_len, **kwargs):
        num_features = len(features[0].features_inject)
        features_arr = np.zeros((len(features), num_features, max_len))
        for i, f in enumerate(features):
            for j, feature_name in enumerate(f.features_inject.keys()):
                features_arr[i][j] = f.features_inject[feature_name]
        all_features = torch.tensor(features_arr, dtype=torch.float)
        return all_features


class DatasetSentLevel(Dataset):

    def __init__(self, config, evaluate=False, serving_mode=False):
        super().__init__(config, evaluate=evaluate)
        self.df = None
        self.examples = None
        self.tensors = None
        if serving_mode:
            self.make_dataset = self.make_dataset_text
        else:
            self.make_dataset = self.make_dataset_serving

    def make_dataset_text(self, data_path, features_path=None, no_cache=False, verbose=True):
        self.read(data_path, features_path=features_path)
        self.load_examples()
        self.make_tensors(no_cache=no_cache, verbose=verbose)

    def make_dataset_serving(self, input_list):
        self.process_request(input_list)
        self.load_examples()
        self.make_tensors(no_cache=True, verbose=False)

    def process_request(self, input_list):
        self.df = pd.DataFrame(input_list)
        self.df['labels'] = np.zeros(len(self.df))

    def read(self, data_path, features_path=None):
        select_columns = ['original', 'translation', 'z_mean']
        data = pd.read_csv(data_path, sep='\t', quoting=3)
        data = data[select_columns]
        data = data.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'})
        if self.output_mode == 'regression':
            data = fit(data, 'labels')
        if features_path is not None:
            features = pd.read_csv(features_path, sep='\t', header=None)
            num_features = len(features.columns)
            features.columns = ['{}{}'.format(DEFAULT_FEATURE_NAME, i) for i in range(1, num_features + 1)]
            assert len(features) == len(data)
            for column in features.columns:
                data[column] = features[column]
        self.df = data

    def load_examples(self):
        assert 'text_a' in self.df.columns and 'text_b' in self.df.columns
        examples = [
            InputExampleSent(i, text_a, text_b, label)
            for i, (text_a, text_b, label) in enumerate(
                zip(self.df["text_a"], self.df["text_b"], self.df["labels"])
            )
        ]
        if "{}1".format(DEFAULT_FEATURE_NAME) in self.df.columns:
            for col in self.df.columns:
                if col.startswith(DEFAULT_FEATURE_NAME):
                    values = self.df[col].to_list()
                    for i, ex in enumerate(examples):
                        ex.features_inject[col] = values[i]
        self.examples = examples

    def _map_labels(self, example, **kwargs):
        return example.label

    def _map_features(self, example, **kwargs):
        return example.features_inject

    def _injected_features_to_tensor(self, features, **kwargs):
        num_features = len(features[0].features_inject)
        features_arr = np.zeros((len(features), num_features))
        for i, f in enumerate(features):
            for j, feature_name in enumerate(f.features_inject.keys()):
                features_arr[i][j] = f.features_inject[feature_name]
        all_features = torch.tensor(features_arr, dtype=torch.float)
        return all_features
