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
from itertools import permutations, product
from typing import Literal, cast

import nibabel as nib
import numpy as np

VoxSizeOption = float | Literal["min"]
ImageSizeOption = int | Literal["fov", "auto"]

__axcode = ("rl", "ap", "si")
__orders = tuple(permutations(range(3)))
__flips = ((0, 1),) * 3
__axcodes = ["".join(__axcode[ii[i]][j] for i, j in enumerate(jj)) for ii, jj in product(__orders, product(*__flips))]
VALID_ORIENTATIONS = ["native", *map(lambda x: "soft " + x, __axcodes), *__axcodes]

OrientationType = str
# future better typing, requires Python 3.11 (Syntax Error before that)
# OrientationType = Literal[*VALID_ORIENTATIONS]



def orientation(a: str) -> OrientationType:
    """
    Convert the orientation argument to a valid orientation from 'native', 'soft[-_ ]<orientation/i>', and
    '<orientation/i>', where <orientation/i> is any valid orientation (case-insensitive).

    Parameters
    ----------
    a : str
        Target orientation type, handles cases.

    Returns
    -------
    str
        One of 'native', 'soft <orientation>', or '<orientation>'.

    Raises
    ------
    argparse.ArgumentTypeError
        If the argument is not a valid choice.
    """
    r = a.lower().replace("_", " ").replace("-", " ").strip()
    if r in VALID_ORIENTATIONS:
        return cast(OrientationType, r)
    valid_orientations = "'native', 'soft-<orientation>', or '<orientation>'"
    raise argparse.ArgumentTypeError(f"'{a}' is not a valid orientation from {valid_orientations}.") from None


def string_to_bool(a: str) -> bool:
    """
    Convert a string to a boolean value.

    Parameters
    ----------
    a : str
        String to convert.

    Returns
    -------
    bool
        If a is "on", "true", "yes", "y", "1" (case-insensitive).
    """
    return a.lower() in ("on", "true", "yes", "y", "1")

def vox_size(a: str) -> VoxSizeOption | None:
    """
    Convert the vox_size argument to 'min' or a valid voxel size.

    Parameters
    ----------
    a : str
        Vox size type. Can be auto, min or a number between 1 an 0.

    Returns
    -------
    str or float or None
        If 'auto' or 'min' is provided, it returns a string('auto' or 'min').
        If a valid voxel size (between 0 and 1) is provided, it returns a float.
        If 'any', it returns None.

    Raises
    ------
    argparse.ArgumentTypeError
        If the argument is not "min", "auto" or convertible to a float between 0 and 1.
    """
    if a is None or a.lower() == "any":
        return None
    if a.lower() in ["auto", "min"]:
        return "min"
    try:
        return float_gt_zero_and_le_one(a)
    except argparse.ArgumentError as e:
        raise argparse.ArgumentTypeError(e.args[0] + " Additionally, vox_sizes may be 'min'.") from None

def img_size(a: str) -> ImageSizeOption | None:
    """
    Convert the img_size argument to 'fov', 'auto' or int as a valid image size.

    Parameters
    ----------
    a : str
        Image size type. Can be auto, fov or an integer greater than 0.

    Returns
    -------
    str or int
        If 'auto' or 'fov' is provided, it returns a string('auto' or 'fov').
        If a valid image size (greater than 0) is provided, it returns an int.
        If 'any', it returns None.

    Raises
    ------
    argparse.ArgumentTypeError
        If the argument is not "fov", "auto" or convertible to an int greater than 0.
    """
    if a.lower() in ("auto", "fov"):
        return cast(ImageSizeOption, a.lower())
    if a.lower() == "any":
        return None
    try:
        return int_gt_zero(a)
    except argparse.ArgumentError as e:
        raise argparse.ArgumentTypeError(e.args[0] + " Additionally, img_sizes may be 'fov'.") from None


def float_gt_zero_and_le_one(a: str) -> float | None:
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
    dtypes = list(nib.freesurfer.mghformat.data_type_codes.value_set("label"))
    dtypes.append("any")
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


def int_gt_zero(value: str | int) -> int:
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


def int_ge_zero(value: str) -> int:
    """
    Convert to integers greater 0.

    Parameters
    ----------
    value : str
        String to convert to int.

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
    Unquote a (single quoted) string, i.e. remove one level of single-quotes.

    Parameters
    ----------
    value : str
        String to be unquoted.

    Returns
    -------
    val : str
        A string of the value without leading and trailing single-quotes.
    """
    val = str(value)
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1]
    return val
