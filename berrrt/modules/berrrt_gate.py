import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from berrrt.aliases import SequenceOrTensor


class Gate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gate_weight = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.gate_weight(x)


class AttentionGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: SequenceOrTensor) -> SequenceOrTensor:
        # Multiple hidden_states turned into one

        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        with torch.backends.cuda.sdp_kernel():
            attention_value = F.scaled_dot_product_attention(Q, K, V)

        return attention_value


class BERRRTFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class BERRRTGateModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        layer_start,
        layer_end,
        freeze_base,
        num_classes,
        dropout: float = 0.1,
        gate: str = "softmax",
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

        self.layer_start = layer_start
        self.layer_end = layer_end
        self.num_classes = num_classes
        self.gate_type = gate
        self.num_layers = layer_end - layer_start + 1
        self.gate = Gate(
            self.bert.config.hidden_size,
            self.num_layers
            if gate in ["softmax", "sigmoid"]
            else self.bert.config.hidden_size,
        )
        self.berrrt_ffn = BERRRTFFN(self.bert.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states[self.layer_start : self.layer_end + 1]
        batch_size, seq_length, hidden_dim = outputs.last_hidden_state.shape

        cumulative_output = torch.zeros(
            batch_size, seq_length, hidden_dim, device=outputs.last_hidden_state.device
        )

        stacked_hidden_states = torch.stack(all_hidden_states, dim=-1)
        if self.gate_type in ["softmax", "sigmoid"]:
            if self.gate_type == "softmax":
                gate_values = F.softmax(
                    stacked_hidden_states, dim=0
                )  # Apply softmax across the first dimension (layers)
            else:  # Sigmoid
                gate_values = torch.sigmoid(stacked_hidden_states)
            for i, hidden_state in enumerate(all_hidden_states):
                berrrt_output = self.berrrt_ffn(hidden_state)
                layer_gate_values = gate_values[..., -1]
                print(f"{gate_values.shape = }")
                print(f"{layer_gate_values.shape = }")
                print(f"{berrrt_output.shape = }")
                weighted_output = (
                    layer_gate_values * berrrt_output
                )  # Element-wise multiplication
                cumulative_output += weighted_output
        elif self.gate_type == "attention":
            attention_output = self.gate(torch.stack(all_hidden_states, dim=0))
            for i, hidden_state in enumerate(all_hidden_states):
                attention_weighted_state = attention_output[i]
                berrrt_output = self.berrrt_ffn(attention_weighted_state)
                cumulative_output += berrrt_output

        pooled_output = cumulative_output[0]
        pooled_output = self.dropout(cumulative_output)
        logits = self.classifier(pooled_output)[:, 0, :]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return (
            {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
        )
