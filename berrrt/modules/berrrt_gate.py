import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class Gate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.gate_weight = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.gate_weight(x)


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
        output_dim = self.num_layers if gate == "softmax" else 1
        self.gate = Gate(self.bert.config.hidden_size, output_dim)
        self.berrrt_ffn = BERRRTFFN(self.bert.config)
        self.output_layer = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states[self.layer_start : self.layer_end + 1]
        batch_size, seq_length, hidden_dim = outputs.last_hidden_state.shape

        cumulative_output = torch.zeros(
            batch_size, seq_length, hidden_dim, device=outputs.last_hidden_state.device
        )

        if self.gate_type == "softmax":
            gate_values = [
                self.gate(hidden_state) for hidden_state in all_hidden_states
            ]
            gate_values = torch.stack(gate_values, dim=2)

            softmax_gate_values = F.softmax(gate_values, dim=2)

            for i, hidden_state in enumerate(all_hidden_states):
                berrrt_output = self.berrrt_ffn(hidden_state)

                layer_gate_values = softmax_gate_values[:, :, i, :].unsqueeze(-1)

                weighted_output = layer_gate_values * berrrt_output.unsqueeze(2)
                cumulative_output += weighted_output.sum(dim=2)

        elif self.gate_type == "sigmoid":
            for hidden_state in all_hidden_states:
                gate_values = self.gate(hidden_state)
                sigmoid_gate_values = torch.sigmoid(gate_values)
                berrrt_output = self.berrrt_ffn(hidden_state)

                weighted_output = sigmoid_gate_values * berrrt_output
                cumulative_output += weighted_output

        cumulative_output = self.dropout(cumulative_output)
        logits = self.output_layer(cumulative_output.mean(dim=1))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return (
            {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
        )
