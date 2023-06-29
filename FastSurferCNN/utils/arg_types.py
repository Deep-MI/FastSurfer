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
    """Helper function to convert the vox_size argument to 'min' or a valid voxel size.

    Parameters
    ----------
    a : str
        vox size type. Can be auto, bin or a number between 1 an 0

    Returns
    -------
    [MISSING]
        

    Raises
    ------
    argparse.ArgumentTypeError
        An error from creating or using an argument. Additionally, vox_sizes may be 'min'.

    
    """

    if a.lower() in ["auto", "min"]:
        return "min"
    try:
        return float_gt_zero_and_le_one(a)
    except argparse.ArgumentError as e:
        raise argparse.ArgumentTypeError(
            e.args[0] + " Additionally, vox_sizes may be 'min'."
        ) from None


def float_gt_zero_and_le_one(a: str) -> Optional[float]:
    """Helper function to check whether a parameters is a float between 0 and one.

    Parameters
    ----------
    a : str
        String of a number or none, infinity

    Returns
    -------
    [MISSING]
        

    
    """

    if a is None or a.lower() in ["none", "infinity"]:
        return None
    a_float = float(a)
    if 0.0 < a_float <= 1.0:
        return a_float
    else:
        raise argparse.ArgumentTypeError(f"'{a}' is not between 0 and 1.")


def target_dtype(a: str) -> str:
    """Helper function to check for valid dtypes.

    Parameters
    ----------
    a : str
        datatype

    Returns
    -------
    [MISSING]
        

    Raises
    ------
    argparse.ArgumentTypeError
        Invalid dtype

    
    """

    dtypes = nib.freesurfer.mghformat.data_type_codes.value_set("label")
    dtypes.add("any")
    _a = a.lower()
    if _a in dtypes:
        return _a
    msg = "The following dtypes are verified: " + ", ".join(dtypes)
    if np.dtype(_a).name == _a:
        # numpy recognizes the dtype, but nibabel probably does not.
        print(
            f"WARNING: While numpy recognizes the dtype {a}, nibabel might not and this might lead to compatibility "
            f"issues. {msg}"
        )
        return _a
    else:
        raise argparse.ArgumentTypeError(f"Invalid dtype {a}. {msg}")


def int_gt_zero(value: Union[str, int]) -> int:
    """Conversion to positive integers.

    Parameters
    ----------
    value : Union[str, int]
        integer to convert

    Returns
    -------
    val : int
        converted integer

    Raises
    ------
    argparse
        ArgumentTypeError: Invalid value, must not be negative.

    
    """

    val = int(value)
    if val <= 0:
        raise argparse.ArgumentTypeError("Invalid value, must not be negative.")
    return val


def int_ge_zero(value) -> int:
    """Conversion to integers greater 0.

    Parameters
    ----------
    value :
        integer to convert

    Returns
    -------
    val : int
        given value if bigger or equal to zero

    Raises
    ------
    argparse
        ArgumentTypeError: Invalid value, must be greater than 0.

    """

    val = int(value)
    if val < 0:
        raise argparse.ArgumentTypeError("Invalid value, must be greater than 0.")
    return val


def unquote_str(value) -> str:
    """Unquotes a (single quoted) string.

    Parameters
    ----------
    value :
        String to be unquoted

    Returns
    -------
    val : str
        A string of the value without quoting with '''

    
    """

    val = str(value)
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1]
    return val
