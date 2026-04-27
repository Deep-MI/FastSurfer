# Copyright 2025 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

__all__ = [
    "checkpoint",
    "check_literal_type",
    "common",
    "load_config",
    "logging",
    "lr_scheduler",
    "mapper",
    "meters",
    "metrics",
    "misc",
    "parser_defaults",
    "parallel",
    "Plane",
    "PlaneAxial",
    "PlaneCoronal",
    "PlaneSagittal",
    "PLANES",
    "ScalarType",
    "Shape1d",
    "Shape2d",
    "Shape3d",
    "Shape4d",
    "ShapeType",
]

from contextlib import contextmanager
from typing import Literal, TypeVar, Union, get_args, get_origin

# there are very few cases, when we do not need nibabel in any "full script" so always
# including nibabel does not overly drag down performance
try:
    from nibabel.analyze import SpatialHeader as nibabelHeader
    from nibabel.analyze import SpatialImage as nibabelImage
    HAS_NIBABEL = True
    __all__ += ["nibabelImage", "nibabelHeader"]
# Some scripts like the build script do not require the full FastSurfer environment. This makes sure, this typing
# module is still functional in such cases.
except (ImportError, ModuleNotFoundError):
    HAS_NIBABEL = False
try:
    from numpy import bool_, dtype, float64, ndarray, number
    HAS_NUMPY = True
# Some scripts like the build script do not require the full FastSurfer environment. This makes sure, this typing
# module is still functional in such cases.
except (ImportError, ModuleNotFoundError):
    HAS_NUMPY = False

PlaneAxial = Literal["axial"]
PlaneCoronal = Literal["coronal"]
PlaneSagittal = Literal["sagittal"]
Plane = PlaneAxial | PlaneCoronal | PlaneSagittal
PLANES: tuple[PlaneAxial, PlaneCoronal, PlaneSagittal] = ("axial", "coronal", "sagittal")
Shape1d = tuple[int]
Shape2d = tuple[int, int]
Shape3d = tuple[int, int, int]
Shape4d = tuple[int, int, int, int]
ShapeType = TypeVar("ShapeType", bound=tuple[int, ...])

if HAS_NUMPY:
    AffineMatrix3x3 = ndarray[tuple[Literal[3], Literal[3]], dtype[float64]]
    AffineMatrix4x4 = ndarray[tuple[Literal[4], Literal[4]], dtype[float64]]
    ScalarType = TypeVar("ScalarType", covariant=True, bound=number)
    Vector2d = ndarray[tuple[Literal[2]], dtype[float64]]
    Vector3d = ndarray[tuple[Literal[3]], dtype[float64]]
    Image2d = ndarray[Shape2d, dtype[ScalarType]]
    Image3d = ndarray[Shape3d, dtype[ScalarType]]
    Image4d = ndarray[Shape4d, dtype[ScalarType]]
    Mask2d = ndarray[Shape2d, dtype[bool_]]
    Mask3d = ndarray[Shape3d, dtype[bool_]]
    Mask4d = ndarray[Shape4d, dtype[bool_]]
    __all__ += [
        "AffineMatrix4x4",
        "Image2d",
        "Image3d",
        "Image4d",
        "Mask2d",
        "Mask3d",
        "Mask4d",
        "AffineMatrix3x3",
        "ScalarType",
        "Vector2d",
        "Vector3d",
    ]

LiteralType = TypeVar("LiteralType")

def check_literal_type(value, literal_type: type[LiteralType]) -> tuple[bool, LiteralType]:
    """
    A simple type checker and converter.

    Parameters
    ----------
    value : Any
        The value to check.
    literal_type : type of Literal
        The Literal to check against.

    Returns
    -------
    bool
        Whether the value is valid for the given Literal type.
    LiteralType
        Value cast to the Literal type.
    """
    from types import UnionType

    _origin_type = get_origin(literal_type)
    if _origin_type is Literal:
        return value in get_args(literal_type), value
    elif _origin_type is Union or _origin_type is UnionType:
        return any(check_literal_type(value, lt)[0] for lt in get_args(literal_type)), value
    else:
        raise TypeError("literal_type is not a Literal or Union of Literals.")
