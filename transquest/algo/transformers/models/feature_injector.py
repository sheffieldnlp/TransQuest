import numpy as np
import torch

from torch import nn


class Combinator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_features = config.num_features
        self.num_labels = config.num_labels
        self.hidden_dim = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    @staticmethod
    def prepare_features_inject(features_inject):
        if len(features_inject.shape) < 2:
            features_inject = features_inject.unsqueeze(1)  # shape: (B, H)
        return features_inject


class Outer(Combinator):

    def __init__(self, config):
        super(Outer, self).__init__(config)
        self.hidden_dim = config.hidden_size
        self.out_proj = nn.Linear(self.hidden_dim * self.num_features, self.num_labels)

    def forward(self, x, features_inject):
        features_inject = self.prepare_features_inject(features_inject)
        x = torch.einsum('bi,bj->bij', (x, features_inject))
        x = torch.flatten(x, start_dim=1)
        x = self.out_proj(x)
        return x


class OuterReducer(Combinator):

    def __init__(self, config):
        super(OuterReducer, self).__init__(config)
        self.hidden_dim = config.hidden_size
        self.reducer = nn.Sequential(
                nn.Linear(
                    self.hidden_dim * self.num_features,
                    int((self.hidden_dim * self.num_features) / 2)
                    ),
                self.dropout,
                nn.ReLU(inplace=True),
                nn.Linear(
                    int((self.hidden_dim * self.num_features) / 2),
                    int((self.hidden_dim * self.num_features) / 4)
                    ),
                self.dropout,
                nn.ReLU(inplace=True),
                )
        self.out_proj = nn.Linear(int((self.hidden_dim * self.num_features) / 4), self.num_labels)

    def forward(self, x, features_inject):
        features_inject = self.prepare_features_inject(features_inject)
        x = torch.einsum('bi,bj->bij', (x, features_inject))
        x = torch.flatten(x, start_dim=1)
        x = self.reducer(x)
        x = self.out_proj(x)
        return x


class Reduce(Combinator):

    def __init__(self, config):
        super(Reduce, self).__init__(config)
        self.reducer = nn.Linear(self.hidden_dim, self.num_features)
        self.dense = nn.Linear(self.num_features * 2, self.num_features * 2)
        self.out_proj = nn.Linear(self.num_features * 2, self.num_labels)

    def forward(self, x, features_inject):
        batch_dim, hidd_dim = x.shape
        features_inject = self.prepare_features_inject(features_inject)
        assert features_inject.shape[1] == self.num_features
        x = self.reducer(x)
        assert x.shape == (batch_dim, self.num_features)
        x = torch.cat((x, features_inject), dim=1)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class Concat(Combinator):

    def __init__(self, config):
        super(Concat, self).__init__(config)
        self.dense = nn.Linear(self.hidden_dim + self.num_features, self.hidden_dim + self.num_features)
        self.out_proj = nn.Linear(self.hidden_dim + self.num_features, self.num_labels)

    def forward(self, x, features_inject):
        features_inject = self.prepare_features_inject(features_inject)
        assert features_inject.shape[1] == self.num_features
        x = torch.cat((x, features_inject), dim=1)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class Convolution(Combinator):

    def __init__(self, config):
        super(Convolution, self).__init__(config)

        self.kernel_size = 8
        self.stride = 3
        self.out_conv_size = self.compute_conv_output_size(self.hidden_dim)
        self.out_pool_size = self.compute_conv_output_size(self.out_conv_size)

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=8, stride=3)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=3)
        self.dense = nn.Linear(self.out_pool_size + self.num_features, self.out_pool_size + self.num_features)
        self.out_proj = nn.Linear(self.out_pool_size + self.num_features, self.num_labels)

    def forward(self, x, features_inject):
        features_inject = self.prepare_features_inject(features_inject)
        assert features_inject.shape[1] == self.num_features

        x = torch.unsqueeze(x, dim=1)
        x = self.conv(x)
        x = self.pool(x)
        x = torch.tanh(x)
        x = torch.squeeze(x, dim=1)

        x = torch.cat((x, features_inject), dim=1)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

    def compute_conv_output_size(self, inp_dim):
        return int(np.floor((inp_dim - (self.kernel_size - 1) - 1)/self.stride + 1))


class FeatureInjector(nn.Module):

    methods = {
        'reduce': Reduce,
        'concat': Concat,
        'conv': Convolution,
        'outer': Outer,
        'outer_reducer': OuterReducer,
    }

    def __init__(self, config):
        super().__init__()
        self.combinator = self.methods[config.feature_combination](config)

    def forward(self, x, features_inject):
        return self.combinator(x, features_inject)
