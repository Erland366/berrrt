from os import PathLike
from typing import Sequence

import torch

__all__ = ["PathOrStr", "SequenceOrTensor"]

PathOrStr = str | PathLike
SequenceOrTensor = Sequence | torch.Tensor
Device = torch.device | str | None
