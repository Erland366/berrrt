import torch.nn as nn
from transformers import BertForSequenceClassification


class BERTModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        freeze_base: bool,
        num_classes: int,
    ):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_classes,
        )

        if freeze_base:
            for param in self.model.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs
