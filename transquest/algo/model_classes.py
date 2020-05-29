from transformers import (
    BertConfig,
    BertTokenizer,
    XLNetConfig,
    XLNetTokenizer,
    XLMConfig,
    XLMTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    AlbertConfig,
    AlbertTokenizer,
    CamembertConfig,
    CamembertTokenizer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    FlaubertConfig,
    FlaubertTokenizer,
)

from transquest.algo.transformers.models.albert_model import AlbertForSequenceClassification
from transquest.algo.transformers.models.bert_model import BertForSequenceClassification
from transquest.algo.transformers.models.camembert_model import CamembertForSequenceClassification
from transquest.algo.transformers.models.distilbert_model import DistilBertForSequenceClassification
from transquest.algo.transformers.models.roberta_model import RobertaForSequenceClassification
from transquest.algo.transformers.models.xlm_model import XLMForSequenceClassification
from transquest.algo.transformers.models.xlm_roberta_model import XLMRobertaForSequenceClassification
from transquest.algo.transformers.models.xlm_roberta_model import XLMRobertaForTokenClassification
from transquest.algo.transformers.models.xlm_roberta_model_inject import XLMRobertaForSequenceClassificationInject
from transquest.algo.transformers.models.xlm_roberta_model_inject import XLMRobertaInjectConfig
from transquest.algo.transformers.models.xlnet_model import XLNetForSequenceClassification

from transformers import FlaubertForSequenceClassification


model_classes = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "camembert": (CamembertConfig, CamembertForSequenceClassification, CamembertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "xlmrobertatoken": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
    "xlmrobertainject": (XLMRobertaInjectConfig, XLMRobertaForSequenceClassificationInject, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}