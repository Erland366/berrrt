from os import PathLike
from typing import Sequence, Union

from torch import Tensor

__all__ = ["PathOrStr", "SequenceOrTensor"]

PathOrStr = Union[str, PathLike[str]]
SequenceOrTensor = Union[Sequence, Tensor]
