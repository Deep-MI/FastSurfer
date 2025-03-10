from pathlib import Path
from typing import Literal, TypedDict

from numpy import ndarray
from yacs.config import CfgNode

from FastSurferCNN.utils import Plane


class ViewOperationDefinition(TypedDict):
    cfg: CfgNode
    ckpt: Path

ViewOperations = dict[Plane, ViewOperationDefinition | None]
ModalityMode = Literal["t1", "t2", "t1t2"]
ModalityDict = dict[Literal["t1", "t2"], ndarray]
RegistrationMode = Literal["robust", "coreg", "none"]
