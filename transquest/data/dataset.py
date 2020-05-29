import os
import numpy as np
import torch
import pandas as pd

from multiprocessing import cpu_count
from multiprocessing import Pool
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset

from transquest.data.normalizer import fit
from transquest.data.containers import InputExampleSent
from transquest.data.containers import InputExampleWord
from transquest.data.containers import InputFeatures
from transquest.data.mapping_tokens_bpe import map_tokens_bpe

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

    def read(self, **kwargs):
        pass

    def load_examples(self, **kwargs):
        pass

    def make_tensors(self, examples, no_cache=False, verbose=True):
        if not no_cache:
            no_cache = self.config['no_cache']

        os.makedirs(self.config['cache_dir'], exist_ok=True)

        mode = 'dev' if self.evaluate else 'train'
        cached_features_file = os.path.join(
            self.config['cache_dir'],
            'cached_{}_{}_{}_{}_{}'.format(
                mode, self.config['model_type'], self.max_seq_length, self.config['num_labels'], len(examples),
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
            features = self._convert_examples_to_features(examples)
            if not no_cache:
                torch.save(features, cached_features_file)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_features = None
        if features[0].features_inject:
            num_features = len(features[0].features_inject)
            features_arr = np.zeros((len(features), num_features))
            for i, f in enumerate(features):
                for j, feature_name in enumerate(f.features_inject.keys()):
                    features_arr[i][j] = f.features_inject[feature_name]
            all_features = torch.tensor(features_arr, dtype=torch.float)

        label_torch_type = torch.long if self.output_mode == 'classification' else torch.float
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=label_torch_type)

        tensors = [all_input_ids, all_input_mask, all_segment_ids, all_label_ids]
        if all_features is not None:
            tensors.append(all_features)
        dataset = TensorDataset(*tensors)
        return dataset

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
        if type(example) is InputExampleWord:
            label = self._pad(self._get_labels(example, tokens_a), ([pad_token] * padding_length))
        else:
            label = example.label

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        return InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label,
            features_inject=example.features_inject,
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

    def _get_labels(self, example, pieces):
        tokens = example.text_a.split()
        labels = example.label
        labels = map_tokens_bpe(tokens, pieces, labels)
        labels = labels + [0]  # sep token
        return labels + [0] if self.cls_token_at_end else [0] + labels


class DatasetWordLevel(Dataset):

    def __init__(self, config, evaluate=False):
        super().__init__(config, evaluate=evaluate)

    def make_dataset(self, src_path, tgt_path, labels_path, no_cache=False, verbose=True):
        src, tgt, labels = self.read(src_path, tgt_path, labels_path)
        examples = self.load_examples(src, tgt, labels)
        tensors = self.make_tensors(examples, no_cache=no_cache, verbose=verbose)
        return tensors

    @staticmethod
    def load_examples(src, tgt, labels):
        examples = [
            InputExampleWord(guid=i, text_a=text_b, label=label)
            for i, (text_a, text_b, label) in enumerate(
                zip(src, tgt, labels)
            )
        ]
        return examples

    def read(self, src_path, tgt_path, labels_path):
        labels = self._read_labels(labels_path)
        src = [l.strip() for l in open(src_path)]
        tgt = [l.strip() for l in open(tgt_path)]
        return src, tgt, labels

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


class DatasetSentLevel(Dataset):

    def __init__(self, config, evaluate=False):
        super().__init__(config, evaluate=evaluate)

    def make_dataset(self, data_path, features_path=None, no_cache=False, verbose=True):
        data = self.read(data_path, features_path=features_path)
        examples = self.load_examples(data)
        tensors = self.make_tensors(examples, no_cache=no_cache, verbose=verbose)
        return tensors

    def read(self, data_path, features_path=None):
        select_columns = ['original', 'translation', 'z_mean']
        data = pd.read_csv(data_path, sep='\t', quoting=3)
        data = data[select_columns]
        data = data.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'})
        data = fit(data, 'labels')
        if features_path is not None:
            features = pd.read_csv(features_path, sep='\t', header=None)
            num_features = len(features.columns)
            features.columns = ['{}{}'.format(DEFAULT_FEATURE_NAME, i) for i in range(1, num_features + 1)]
            assert len(features) == len(data)
            for column in features.columns:
                data[column] = features[column]
        return data

    def load_examples(self, df):
        assert 'text_a' in df.columns and 'text_b' in df.columns
        examples = [
            InputExampleSent(i, text_a, text_b, label)
            for i, (text_a, text_b, label) in enumerate(
                zip(df["text_a"], df["text_b"], df["labels"])
            )
        ]
        if "{}1".format(DEFAULT_FEATURE_NAME) in df.columns:
            for col in df.columns:
                if col.startswith(DEFAULT_FEATURE_NAME):
                    values = df[col].to_list()
                    for i, ex in enumerate(examples):
                        ex.features_inject[col] = values[i]
        return examples