import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel


class Gate(nn.Module):
    def __init__(self, input_dim, activation):
        super().__init__()
        self.activation = activation
        if activation == "sigmoid":
            output_dim = 1
        elif activation == "softmax":
            output_dim = 2
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.gate_weight = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.gate_weight(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation == "softmax":
            return F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")


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
        self, bert_model_name, layer_start, layer_end, freeze_base, activation="softm"
    ):
        super().__init__()
        config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(bert_model_name, config=config)

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False

        self._layer_start = None
        self._layer_end = None
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.gate = Gate(self.bert.config.hidden_size)
        self.berrrt_ffn = BERRRTFFN(self.bert.config)
        self.output_layer = nn.Linear(
            self.bert.config.hidden_size, self.bert.config.hidden_size
        )

    @property
    def layer_start(self):
        return self._layer_start

    @layer_start.setter
    def layer_start(self, value):
        if value < 0:
            raise ValueError("layer_start must be non-negative")
        if self._layer_end is not None and value > self._layer_end:
            raise ValueError("layer_start must be less than or equal to layer_end")
        self._layer_start = value

    @property
    def layer_end(self):
        return self._layer_end

    @layer_end.setter
    def layer_end(self, value):
        if value < 0:
            raise ValueError("layer_end must be non-negative")
        if self._layer_start is not None and value < self._layer_start:
            raise ValueError("layer_end must be greater than or equal to layer_start")
        if value >= self.config.num_hidden_layers:
            raise ValueError(
                f"layer_end must be less than the number of layers in the model ({self.config.num_hidden_layers} layers)"
            )
        self._layer_end = value

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        all_hidden_states = outputs.hidden_states[self.layer_start : self.layer_end + 1]
        cumulative_output = torch.zeros_like(outputs.last_hidden_state)

        for hidden_state in all_hidden_states:
            gate_values = self.gate(hidden_state)
            berrrt_output = self.berrrt_ffn(hidden_state)
            cumulative_output += gate_values * berrrt_output

        final_output = self.output_layer(cumulative_output)

        logits = final_output[:, 0, :]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.output_layer.out_features), labels.view(-1)
            )

        return (
            {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
        )
