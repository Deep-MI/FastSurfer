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
from typing import Literal, Optional, Union

import nibabel as nib
import numpy as np

VoxSizeOption = Union[float, Literal["min"]]


def vox_size(a: str) -> VoxSizeOption:
    """
    Convert the vox_size argument to 'min' or a valid voxel size.

    Parameters
    ----------
    a : str
        Vox size type. Can be auto, bin or a number between 1 an 0.

    Returns
    -------
    str or float
        If 'auto' or 'min' is provided, it returns a string('auto' or 'min').
        If a valid voxel size (between 0 and 1) is provided, it returns a float.

    Raises
    ------
    argparse.ArgumentTypeError
        If the arguemnt is not "min", "auto" or convertible to a float between 0 and 1.
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
    """
    Check whether a parameters are a float between 0 and one.

    Parameters
    ----------
    a : str
        String of a number or none, infinity.

    Returns
    -------
    float or None
        If `a` is a valid float between 0 and 1, return the float value.
        If `a` is 'none' or 'infinity', return None.

    Raises
    ------
    argparse.ArgumentTypeError
        If `a` is neither a float between 0 and 1.
    """
    if a is None or a.lower() in ["none", "infinity"]:
        return None
    a_float = float(a)
    if 0.0 < a_float <= 1.0:
        return a_float
    else:
        raise argparse.ArgumentTypeError(f"'{a}' is not between 0 and 1.")


def target_dtype(a: str) -> str:
    """
    Check for valid dtypes.

    Parameters
    ----------
    a : str
        Datatype descriptor.

    Returns
    -------
    str
        The validated data type.

    Raises
    ------
    argparse.ArgumentTypeError
        Invalid dtype.

    See Also
    --------
    numpy.dtype
        For more information on numpy data types and their properties.
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
    """
    Convert to positive integers.

    Parameters
    ----------
    value : Union[str, int]
        Integer to convert.

    Returns
    -------
    val : int
        Converted integer.

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
    """
    Convert to integers greater 0.

    Parameters
    ----------
    value : int
        Integer to convert.

    Returns
    -------
    val : int
        Given value if bigger or equal to zero.

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
    """
    Unquote a (single quoted) string.

    Parameters
    ----------
    value : str
        String to be unquoted.

    Returns
    -------
    val : str
        A string of the value without quoting with '''.
    """
    val = str(value)
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1]
    return val
