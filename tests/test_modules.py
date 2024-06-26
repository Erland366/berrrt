import pytest
import torch
from hydra import compose, initialize

from berrrt.aliases import Device
from berrrt.modules.base import ModulesFactory

from .utils import create_input  # type: ignore
from .utils import default_testing_config


@pytest.mark.parametrize(
    "device", ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]
)
@pytest.mark.parametrize("hidden_dim", [256])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("gate_type", ["softmax", "sigmoid", "attention"])
@pytest.mark.slow
def test_gate_modules(hidden_dim: int, batch_size: int, device: Device, gate_type: str):
    input_ids, attention_mask = create_input((batch_size, hidden_dim), device=device)

    with initialize(
        version_base=None,
        config_path="../berrrt/conf",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "modules=berrrt_gate",
                "modules_name=berrrt_gate",
            ],
        )
    cfg = default_testing_config(cfg)

    cfg.modules.gate = gate_type

    model = ModulesFactory(cfg.modules_name).create_model(**cfg.modules).to(device)
    result = model(input_ids, attention_mask)
    assert result["logits"].shape == (batch_size, cfg.modules.num_classes)


@pytest.mark.parametrize(
    "device", ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]
)
@pytest.mark.parametrize("hidden_dim", [256])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("gate_type", ["softmax", "sigmoid", "attention"])
@pytest.mark.slow
def test_ee_modules(hidden_dim: int, batch_size: int, device: Device, gate_type: str):
    input_ids, attention_mask = create_input((batch_size, hidden_dim), device=device)

    with initialize(
        version_base=None,
        config_path="../berrrt/conf",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "modules=berrrt_gate",
                "modules_name=berrrt_gate",
            ],
        )
    cfg = default_testing_config(cfg)

    cfg.modules.gate = gate_type

    model = ModulesFactory(cfg.modules_name).create_model(**cfg.modules).to(device)
    result = model(input_ids, attention_mask)
    assert result["logits"].shape == (batch_size, cfg.modules.num_classes)


@pytest.mark.parametrize(
    "device", ["cpu"] if not torch.cuda.is_available() else ["cpu", "cuda"]
)
@pytest.mark.parametrize("hidden_dim", [256])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "aggregation", ["concat", "weighted_sum", "attention", "add", "average"]
)
@pytest.mark.slow
def test_aggregation_modules(
    hidden_dim: int, batch_size: int, device: Device, aggregation: str
):
    input_ids, attention_mask = create_input((batch_size, hidden_dim), device=device)
    with initialize(
        version_base=None,
        config_path="../berrrt/conf",
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "modules=berrrt",
                "modules_name=berrrt",
            ],
        )
    cfg = default_testing_config(cfg)

    cfg.modules.aggregation = aggregation
    model = ModulesFactory(cfg.modules_name).create_model(**cfg.modules).to(device)
    result = model(input_ids, attention_mask)
    assert result["logits"].shape == (batch_size, cfg.modules.num_classes)
