import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BERRRTFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class BERRRTModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        layer_start,
        layer_end,
        freeze_base,
        num_classes,
        dropout=0.1,
        aggregation: str = "add",
    ):
        super().__init__()
        self.config = BertConfig.from_pretrained(
            pretrained_model_name_or_path, output_hidden_states=True
        )
        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path, config=self.config
        )

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.aggregation = aggregation
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

        if aggregation == "pool":
            self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        if aggregation == "concat":
            self.reduce_linear = nn.Linear(
                self.config.hidden_size * (layer_end - layer_start + 1),
                self.config.hidden_size,
            )
        elif aggregation == "weighted_sum":
            self.layer_weights = nn.Parameter(torch.ones(layer_end - layer_start + 1))
        elif aggregation == "attention":
            self.attention_weights = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        all_hidden_states = outputs.hidden_states[self.layer_start : self.layer_end + 1]

        if self.aggregation == "add":
            cumulative_output = torch.sum(torch.stack(all_hidden_states), dim=0)
        elif self.aggregation == "average":
            cumulative_output = torch.mean(torch.stack(all_hidden_states), dim=0)
        elif self.aggregation == "pool":
            stacked_outputs = torch.stack(
                all_hidden_states
            )  # [num_layers, batch_size, seq_len, hidden_size]
            stacked_outputs = stacked_outputs.permute(
                1, 3, 0, 2
            )  # [batch_size, hidden_size, num_layers, seq_len]
            pooled_output = self.pooling_layer(stacked_outputs).squeeze(
                -1
            )  # [batch_size, hidden_size, seq_len]
            cumulative_output = pooled_output.mean(
                dim=2
            )  # Final pooling across seq_len
        elif self.aggregation == "concat":
            concatenated_output = torch.cat(all_hidden_states, dim=-1)
            cumulative_output = self.reduce_linear(concatenated_output)
        elif self.aggregation == "weighted_sum":
            weighted_outputs = torch.stack(
                [
                    self.layer_weights[i] * output
                    for i, output in enumerate(all_hidden_states)
                ]
            )
            cumulative_output = torch.sum(weighted_outputs, dim=0)
        elif self.aggregation == "attention":
            stacked_outputs = torch.stack(all_hidden_states, dim=0)
            attention_scores = F.softmax(
                self.attention_weights(stacked_outputs).squeeze(-1), dim=0
            )
            cumulative_output = torch.sum(
                attention_scores.unsqueeze(-1).unsqueeze(-1) * stacked_outputs, dim=0
            )
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.aggregation}")

        pooled_output = cumulative_output[1]
        pooled_output = self.dropout(cumulative_output)
        logits = self.classifier(pooled_output)[:, 0, :]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
