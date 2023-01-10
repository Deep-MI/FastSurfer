# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

import argparse
from typing import Union, Literal, Optional

import nibabel as nib
import numpy as np

VoxSizeOption = Union[float, Literal["min"]]


def vox_size(a: str) -> VoxSizeOption:
    """Helper function to convert the vox_size argument to 'min' or a valid voxel size."""
    if a.lower() in ["auto", "min"]:
        return "min"
    a_float = float(a)
    if 0. < a_float <= 1.0:
        return a_float
    else:
        raise argparse.ArgumentError(f"'{a}' is not 'min' or a float between 0 and 1 (vox_size).")


def conform_to_one_mm(a: str) -> Optional[float]:
    """Helper function to convert conform_to_1mm thresholds to numbers."""
    if a is None or a.lower() in ["none", "infinity"]:
        return None
    a_float = float(a)
    if 0. < a_float <= 1.0:
        return a_float
    else:
        raise argparse.ArgumentError(f"'{a}' is not between 0 and 1.")


def target_dtype(a: str) -> str:
    """Helper function to check for valid dtypes."""
    dtypes = nib.freesurfer.mghformat.data_type_codes.value_set('label')
    dtypes.add("any")
    _a = a.lower()
    if _a in dtypes:
        return _a
    msg = "The following dtypes are verified: " + ", ".join(dtypes)
    if np.dtype(_a).name == _a:
        # numpy recognizes the dtype, but nibabel probably does not.
        print(f"WARNING: While numpy recognizes the dtype {a}, nibabel might not and this might lead to compatibility "
              f"issues. {msg}")
        return _a
    else:
        raise argparse.ArgumentError(f"Invalid dtype {a}. {msg}")
