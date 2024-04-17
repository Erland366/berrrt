import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from berrrt.aliases import SequenceOrTensor  # type: ignore


class BERRRTFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # For simplicity, we just use one attention head
        # Reference too adding attention heads:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L289
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, query: SequenceOrTensor, key: SequenceOrTensor, value: SequenceOrTensor
    ) -> SequenceOrTensor:
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        with torch.backends.cuda.sdp_kernel():
            attention_value = F.scaled_dot_product_attention(Q, K, V)

        return attention_value


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

        if aggregation == "concat":
            self.reduce_linear = nn.Linear(
                self.config.hidden_size * (layer_end - layer_start + 1),
                self.config.hidden_size,
            )
        elif aggregation == "weighted_sum":
            self.layer_weights = nn.Parameter(torch.ones(layer_end - layer_start + 1))
        elif aggregation == "attention":
            self.reduce_linear = nn.Linear(
                self.config.hidden_size * (layer_end - layer_start + 1),
                self.config.hidden_size,
            )
            self.attention_layer = Attention(self.config.hidden_size)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        all_hidden_states = outputs.hidden_states[
            self.layer_start : self.layer_end + 1
        ]  # tuple(12)
        # each of themn is [batch_size, seq_len, hidden_size]

        if self.aggregation == "add":
            stacked_output = torch.stack(
                all_hidden_states
            )  # [num_layers, batch_size, seq_len, hidden_size]
            cumulative_output = torch.sum(stacked_output, dim=0)
        elif self.aggregation == "average":
            stacked_output = torch.stack(
                all_hidden_states
            )  # [num_layers, batch_size, seq_len, hidden_size]
            cumulative_output = torch.mean(
                stacked_output, dim=0
            )  # [batch_size, seq_len, hidden_size]
        elif self.aggregation == "concat":
            concatenated_output = torch.cat(
                all_hidden_states, dim=-1
            )  # [batch_size, seq_len, hidden_size * num_layers]
            cumulative_output = self.reduce_linear(
                concatenated_output
            )  # [batch_size, seq_len, hidden_size]
        elif self.aggregation == "weighted_sum":
            weighted_outputs = torch.stack(
                [
                    self.layer_weights[i] * output
                    for i, output in enumerate(all_hidden_states)
                ]
            )
            cumulative_output = torch.sum(weighted_outputs, dim=0)
        elif self.aggregation == "attention":
            concatenated_output = torch.cat(
                all_hidden_states, dim=-1
            )  # [batch_size, seq_len, hidden_size * num_layers]
            reduce_to_hidden_dim = self.reduce_linear(
                concatenated_output
            )  # [batch_size, seq_len, hidden_size]

            cumulative_output = self.attention_layer(
                reduce_to_hidden_dim, reduce_to_hidden_dim, outputs.last_hidden_state
            )
        else:
            raise ValueError(f"Unsupported aggregation strategy: {self.aggregation}")

        pooled_output = self.dropout(
            cumulative_output
        )  # [batch_size, seq_len, hidden_size]

        # Take cls token and pass it through the classifier
        logits = self.classifier(pooled_output[:, 0, :])

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
