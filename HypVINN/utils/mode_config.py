# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Optional

from FastSurferCNN.utils import logging
from HypVINN.utils import ModalityMode

LOGGER = logging.get_logger(__name__)


def get_hypinn_mode(
        t1_path: Optional[Path],
        t2_path: Optional[Path],
) -> ModalityMode:

    LOGGER.info("Setting up input mode...")
    if t1_path is not None and t2_path is not None:
        if t1_path.is_file() and t2_path.is_file():
            return "t1t2"
        msg = []
        if not t1_path.is_file():
            msg.append(f"the t1 file does not exist ({t1_path})")
        if not t2_path.is_file():
            msg.append(f"the t2 file does not exist ({t2_path})")
        raise RuntimeError(
            f"ERROR: Both the t1 and the t2 flags were passed, but "
            f"{' and '.join(msg)}."
        )

    elif t1_path:
        if t1_path.is_file():
            return "t1"
        raise RuntimeError(
            f"ERROR: The t1 flag was passed, but the t1 file does not exist "
            f"({t1_path})."
        )
    elif t2_path:
        if t2_path.is_file():
            LOGGER.info(
                "Warning: T2 mode selected. The quality of segmentations based "
                "on only a T2 image is significantly worse than when T1 images "
                "are included."
            )
            return "t2"
        raise RuntimeError(
            f"ERROR: The t2 flag was passed, but the t1 file does not exist "
            f"({t1_path})."
        )
    else:
        raise RuntimeError(
            "No t1 or t2 flags were passed, invalid configuration."
        )
