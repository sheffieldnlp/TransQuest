from collections import OrderedDict


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.features_inject = OrderedDict()  # TODO: add method for setting features


class InputExampleSent(InputExample):  # TODO: use this for sentence-level

    def __init__(self, guid, text_a, text_b=None, label=None):
        super(InputExampleSent, self).__init__(guid, text_a, text_b=text_b, label=label)
        assert type(self.label) is float or type(self.label) is int


class InputExampleWord(InputExample):
    def __init__(self, guid, text_a, text_b=None, label=None, mt_tokens=None):
        super(InputExampleWord, self).__init__(guid, text_a, text_b=text_b, label=label)
        assert type(self.label) is list
        self.mt_tokens = mt_tokens
        self.text_b = None


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, features_inject=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.features_inject = features_inject
