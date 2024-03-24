import torch
import torch.nn as nn
from transformers.models.resnet.modeling_resnet import \
    ResNetPreTrainedModel, ResNetModel, ImageClassifierOutputWithNoAttention
from typing import Optional
from entmax import sparsemax_loss, entmax15_loss, entmax15, sparsemax
from torch.nn.functional import cross_entropy, softmax


str_to_loss = {'sparsemax': sparsemax_loss, 'entmax15': entmax15_loss, 'softmax': cross_entropy}
str_to_softmax = {'sparsemax': sparsemax, 'entmax15': entmax15, 'softmax': softmax}


class ResNetForImageClassification(ResNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.resnet = ResNetModel(config)
        # classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            loss_fct = str_to_loss[self.config.softmax_function]
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if self.config.softmax_function in ['entmax15', 'sparsemax']:
                loss = loss.mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
