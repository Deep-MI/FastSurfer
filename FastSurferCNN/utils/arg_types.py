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

from itertools import permutations, product
from typing import Literal, cast

import nibabel as nib
import numpy as np

VoxSizeOption = float | Literal["min"]
ImageSizeOption = int | Literal["fov", "auto"]

__axcode = ("rl", "ap", "si")
__orders = tuple(permutations(range(3)))
__flips = ((0, 1),) * 3
ORIENTATIONS = ["".join(__axcode[ii[i]][j] for i, j in enumerate(k)) for ii, k in product(__orders, product(*__flips))]
VALID_ORIENTATIONS = ["native", *map(lambda x: "soft-" + x, ORIENTATIONS), *ORIENTATIONS]

StrictOrientationType = str
OrientationType = str
# future better typing, requires Python 3.11 (Syntax Error before that)
# OrientationType = Literal[*VALID_ORIENTATIONS]
# StrictOrientationType = Literal[*ORIENTATIONS]



def orientation(a: str) -> OrientationType:
    """
    Convert the orientation argument to a valid orientation from 'native', 'soft[-_ ]<orientation/i>', and
    '<orientation/i>', where <orientation/i> is any valid orientation (case-insensitive).

    Parameters
    ----------
    a : str
        Target orientation type, case-insensitive.

    Returns
    -------
    str
        One of 'native', 'soft-<orientation>', or '<orientation>'.

    Raises
    ------
    ValueError
        If the argument is not a valid orientation.
    """
    r = a.lower().replace("_", "-").replace(" ", "-").strip()
    if r in VALID_ORIENTATIONS:
        return cast(OrientationType, r)
    valid_orientations_short = "'native', 'soft-<orientation>', or '<orientation>'"
    raise ValueError(f"'{a}' is not a valid orientation from {valid_orientations_short}.") from None


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
    if not isinstance(a, str):
        return bool(a)
    return a.lower() in ("on", "true", "yes", "y", "1")

def vox_size(a: str | float | None) -> VoxSizeOption | None:
    """
    Convert the vox_size argument to 'min' or a valid voxel size.

    Parameters
    ----------
    a : str, float, None
        Vox size type. Can be auto, min or a number between 1 an 0.

    Returns
    -------
    str or float or None
        If 'auto' or 'min' is provided, it returns a string('auto' or 'min').
        If a valid voxel size (between 0 and 1) is provided, it returns a float.
        If 'any', it returns None.

    Raises
    ------
    ValueError
        If the argument is not "min", "auto" or convertible to a float between 0 and 1.
    """
    if a is None or isinstance(a, str) and a.lower() == "any":
        return None
    if isinstance(a, str) and a.lower() in ["auto", "min"]:
        return "min"
    try:
        return float_gt_zero_and_le_one(a)
    except ValueError as e:
        raise ValueError(e.args[0] + " Additionally, vox_size may be 'min'.") from None

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
    ValueError
        If the argument is not "fov", "auto" or convertible to an int greater than 0.
    """
    if a.lower() in ("auto", "fov"):
        return cast(ImageSizeOption, a.lower())
    if a.lower() == "any":
        return None
    try:
        return int_gt_zero(a)
    except ValueError as e:
        raise ValueError(e.args[0] + " Additionally, img_size may be 'fov'.") from None


def float_gt_zero_and_le_one(a: str | float) -> float | None:
    """
    Check whether a parameters are a float between 0 and one.

    Parameters
    ----------
    a : str, float
        String of a number or none, infinity.

    Returns
    -------
    float or None
        If `a` is a valid float between 0 and 1, return the float value.
        If `a` is 'none' or 'infinity', return None.

    Raises
    ------
    ValueError
        If `a` is neither a float between 0 and 1.
    """
    if a is None or isinstance(a, str) and a.lower() in ["none", "infinity"]:
        return None
    a_float = float(a)
    if 0.0 < a_float <= 1.0:
        return a_float
    else:
        raise ValueError(f"'{a}' is not between 0 and 1.")


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
    ValueError
        If the dtype is invalid.

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
        from logging import getLogger

        warnmsg = f"While numpy recognizes the dtype {a}, nibabel might not leading to compatibility issues. "
        getLogger(__name__).warning(warnmsg + msg)
        return _a
    else:
        raise ValueError(f"Invalid dtype {a}. {msg}")


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
    ValueError
        Invalid value, must not be negative.
    """
    val = int(value)
    if val <= 0:
        raise ValueError("Invalid value, must not be negative.")
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
    ValueError
        Invalid value, must be greater than 0.
    """
    val = int(value)
    if val < 0:
        raise ValueError("Invalid value, must be greater than 0.")
    return val


def unquote_str(value: str) -> str:
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
