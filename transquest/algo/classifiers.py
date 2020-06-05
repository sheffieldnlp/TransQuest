import torch
import torch.nn as nn

from transformers.modeling_roberta import RobertaClassificationHead
from transquest.algo.transformers.models.feature_injector import FeatureInjector


class RobertaClassificationHeadSequenceInjection(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHeadSequenceInjection, self).__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.feature_injector = FeatureInjector(config)

    def forward(self, pretrained, features_inject=None, **kwargs):
        batch_dim, max_len, hidd_dim = pretrained.shape
        x = pretrained[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)  # shape: (B, H)
        assert x.shape == (batch_dim, hidd_dim)
        x = self.feature_injector(x, features_inject)
        return x


class RobertaClassificationHeadTokenInjection(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHeadTokenInjection, self).__init__(config)
        self.dense = nn.Linear(config.hidden_size + config.num_features, config.num_labels)

    def forward(self, pretrained, features_inject=None, **kwargs):
        features_inject = torch.transpose(features_inject, 1, 2)
        x = torch.cat((pretrained, features_inject), dim=2)
        x = self.dense(x)
        return x


class RobertaClassificationHeadToken(RobertaClassificationHead):

    def __init__(self, config):
        super(RobertaClassificationHeadToken, self).__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pretrained, features_inject=None, **kwargs):
        x = self.dense(pretrained)
        return x
