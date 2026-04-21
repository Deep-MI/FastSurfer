# Copyright 2025 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

from typing import TypedDict

import numpy as np
from numpy import typing as npt

from FastSurferCNN.utils import AffineMatrix3x3, Vector3d


class MGHHeaderDict(TypedDict):
    """A dictionary with the four required fields of a MGH Header"""
    dims: Vector3d
    delta: Vector3d
    Mdc: AffineMatrix3x3
    Pxyz_c: Vector3d


def convert_numpy_to_json_serializable(obj: object) -> object:
    """Convert numpy types to JSON serializable types.

    Parameters
    ----------
    obj : dict, list, array, number, serializable
        Object to convert to JSON serializable type.

    Returns
    -------
    object
        JSON serializable version of the input object.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        # Handle numpy scalar types
        return obj.item()
    else:
        return obj
