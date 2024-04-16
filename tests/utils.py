import torch

from berrrt.aliases import Device, SequenceOrTensor


def create_input(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.int32,
    device: Device = "cpu",
    requires_grad: bool = False,
    seed: int | None = 3407,
) -> tuple[SequenceOrTensor, SequenceOrTensor]:
    if seed is not None:
        torch.manual_seed(seed)
    input_ids = torch.randint(
        low=0,
        high=8000,
        size=shape,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def default_testing_config(cfg: dict[str, str]) -> dict[str, str]:
    cfg.train.num_train_epochs = 1
    cfg.train.per_device_train_batch_size = 1
    cfg.train.per_device_eval_batch_size = 1
    cfg.train.use_cpu = True
    cfg.train.report_to = "none"

    del cfg.modules.additional_prefix

    return cfg
