import torch
import torch.nn.functional as F
from jaxtyping import Float  # type : ignore
from torch import nn
from transformers import BertConfig, BertModel

from berrrt.aliases import SequenceOrTensor


class AttentionGate(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

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


class BERRRTFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class BERRRTEarlyExitModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        layer_start: int,
        layer_end: int,
        freeze_base: bool,
        num_classes: int,
        dropout: float = 0.1,
        gate: str = "softmax",
    ) -> None:
        super().__init__()
        self.config = BertConfig.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            output_hidden_states=True,
        )
        self.bert = BertModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=self.config,
        )

        if freeze_base:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.num_classes = num_classes
        self.gate_type = gate
        self.num_layers = layer_end - layer_start + 1
        self.berrrt_ffn = BERRRTFFN(self.bert.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.linear_hidden = nn.Linear(self.bert.config.hidden_size, 1)
        self.attention_gate = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size, num_heads=12
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels: None | SequenceOrTensor = None,
        return_early_exit: bool = True,
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states[self.layer_start : self.layer_end + 1]
        batch_size, _, hidden_dim = outputs.last_hidden_state.shape

        cumulative_output: Float[torch.Tensor, "batch_size hidden_dim"] = torch.zeros(
            batch_size, hidden_dim, device=outputs.last_hidden_state.device
        )

        stacked_hidden_states: Float[
            torch.Tensor, "batch_size n_vocab hidden_dim n_layers"
        ] = torch.stack(all_hidden_states, dim=-1)
        cls_hidden_states: Float[torch.Tensor, "batch_size hidden_dim n_layers"] = (
            stacked_hidden_states[:, 0, :, :]
        )
        cls_hidden_states: Float[torch.Tensor, "batch_size n_layers hidden_dim"] = (
            cls_hidden_states.permute(0, 2, 1)
        )

        loss = 0
        early_exit_logits = []

        if self.gate_type in ["softmax", "sigmoid"]:
            linear_hidden_states: Float[torch.Tensor, "batch_size n_layers 1"] = (
                self.linear_hidden(cls_hidden_states)
            )
            if self.gate_type == "softmax":
                gate_values: Float[torch.Tensor, "barch_size n_layers 1"] = F.softmax(
                    linear_hidden_states, dim=1
                )  # Apply softmax across the first dimension (layers)
            else:  # Sigmoid
                gate_values = F.sigmoid(linear_hidden_states).squeeze(-1)

            gate_values = gate_values.squeeze(-1)
            for i in range(cls_hidden_states.shape[1]):
                berrrt_output = self.berrrt_ffn(
                    cls_hidden_states[:, i, ...]
                )  # [b, n_layers, 768]
                layer_gate_values = gate_values[..., i].unsqueeze(-1)
                weighted_output = (
                    layer_gate_values * berrrt_output
                )  # Element-wise multiplication
                cumulative_output += weighted_output
                pooled_output = self.dropout(cls_hidden_states[:, i, ...])
                logits = self.classifier(pooled_output)
                if return_early_exit:
                    early_exit_logits.append(logits)
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss += loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        elif self.gate_type == "attention":
            for i in range(cls_hidden_states.shape[1]):
                attention_output, attention_weights = self.attention_gate(
                    cls_hidden_states[:, i, ...],  # [b, 768]
                    cls_hidden_states[:, i, ...],  # [b, 768]
                    outputs.last_hidden_state[:, 0, :],  # [b, 768],
                )
                berrrt_output = self.berrrt_ffn(
                    attention_output
                )  # FFN applied to each attention-weighted layer output
                cumulative_output *= berrrt_output

                # Now calculate classifier per output
                pooled_output = self.dropout(cls_hidden_states[:, i, ...])
                logits = self.classifier(pooled_output)
                if return_early_exit:
                    early_exit_logits.append(logits)
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss += loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        pooled_output = cumulative_output
        pooled_output = self.dropout(cumulative_output)
        logits = self.classifier(pooled_output)  # [b, num_classes]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss += loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        # From 3.7, dictionary will always be in sort so don't worry!
        return_dict = {"logits": logits}
        if loss is not None:
            return_dict["loss"] = loss
        if return_early_exit:
            return_dict["early_exit_logits"] = early_exit_logits
        return return_dict
