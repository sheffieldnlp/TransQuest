from transformers.configuration_xlm_roberta import XLMRobertaConfig
from transformers.modeling_xlm_roberta import XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

from transquest.algo.transformers.models.xlm_roberta_model import XLMRobertaForSequenceClassification
from transquest.algo.classifiers import RobertaClassificationHeadInjection


class XLMRobertaInjectConfig(XLMRobertaConfig):

    def __init__(self, num_features=None, feature_combination="concat", **kwargs):
        super(XLMRobertaInjectConfig, self).__init__(**kwargs)
        self.num_features = num_features
        self.feature_combination = feature_combination


class XLMRobertaForSequenceClassificationInject(XLMRobertaForSequenceClassification):
    config_class = XLMRobertaInjectConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

    def __init__(self, config):
        super(XLMRobertaForSequenceClassificationInject, self).__init__(config, weight=None)
        self.classifier = RobertaClassificationHeadInjection(config)
