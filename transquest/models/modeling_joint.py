import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss


from transformers.configuration_roberta import RobertaConfig
from transformers.configuration_xlm_roberta import XLMRobertaConfig
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaClassificationHead,
    ROBERTA_INPUTS_DOCSTRING,
    _TOKENIZER_FOR_DOC,
    _CONFIG_FOR_DOC,
)

from transformers.modeling_outputs import TokenClassifierOutput


class RobertaForJointQualityEstimation(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier_sentlevel = RobertaClassificationHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_wordlevel = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # for sentence-level
        logits_sentlevel = self.classifier_sentlevel(sequence_output)
        # for word-level
        sequence_output = self.dropout(sequence_output)
        logits_wordlevel = self.classifier_wordlevel(sequence_output)

        loss = None
        if labels is not None:
            # for sentence-level (regression)
            loss_fct = MSELoss()
            loss_sentlevel = loss_fct(logits_sentlevel.view(-1), labels[:0].view(-1))

            # for word-level
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits_wordlevel.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels[:, 1:].view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss_wordlevel = loss_fct(active_logits, active_labels)
            else:
                loss_wordlevel = loss_fct(logits_wordlevel.view(-1, self.num_labels), labels[:, 1:].view(-1))

            # joint
            loss = loss_sentlevel + loss_wordlevel

        if not return_dict:
            output = (logits_sentlevel, logits_wordlevel,) + outputs[2:]
            return ((logits_sentlevel, logits_wordlevel,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits_wordlevel, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        )


class XLMRobertaForJointQualityEstimation(RobertaForJointQualityEstimation):
    config_class = XLMRobertaConfig
