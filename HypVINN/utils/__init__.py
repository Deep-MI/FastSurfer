from typing import Any, Literal, Optional

from numpy import ndarray

from FastSurferCNN.utils import Plane

ViewOperations = dict[Plane, dict[Literal["cfg", "ckpt"], Any] | None]
ModalityMode = Literal["t1", "t2", "t1t2"]
ModalityDict = dict[Literal["t1", "t2"], ndarray]
RegistrationMode = Literal["robust", "coreg", "none"]
