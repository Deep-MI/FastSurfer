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
    "AffineMatrix4x4",
    "checkpoint",
    "common",
    "Image2d",
    "Image3d",
    "Image4d",
    "load_config",
    "logging",
    "lr_scheduler",
    "mapper",
    "Mask2d",
    "Mask3d",
    "Mask4d",
    "meters",
    "metrics",
    "misc",
    "nibabelImage",
    "nibabelHeader",
    "parser_defaults",
    "parallel",
    "Plane",
    "PlaneAxial",
    "PlaneCoronal",
    "PlaneSagittal",
    "PLANES",
    "RotationMatrix3x3",
    "ScalarType",
    "Shape2d",
    "Shape3d",
    "Shape4d",
    "ShapeType",
    "Vector2d",
    "Vector3d",
]

from typing import Literal, TypeVar

# there are very few cases, when we do not need nibabel in any "full script" so always
# including nibabel does not overly drag down performance
try:
    from nibabel.analyze import SpatialHeader as nibabelHeader
    from nibabel.analyze import SpatialImage as nibabelImage
# Some scripts like the build script do not require the full FastSurfer environment. This makes sure, this typing
# module is still functional in such cases.
except (ImportError, ModuleNotFoundError):
    nibabelImage = None
    nibabelHeader = None
try:
    from numpy import bool_, dtype, float64, ndarray, number
# Some scripts like the build script do not require the full FastSurfer environment. This makes sure, this typing
# module is still functional in such cases.
except (ImportError, ModuleNotFoundError):
    float64 = float
    bool_ = bool
    # by typing this with tuple, ndarray[...] and dtype [...] will still be valid syntax
    ndarray = tuple
    dtype = tuple
    from numbers import Number as number

AffineMatrix4x4 = ndarray[tuple[Literal[4], Literal[4]], dtype[float64]]
PlaneAxial = Literal["axial"]
PlaneCoronal = Literal["coronal"]
PlaneSagittal = Literal["sagittal"]
Plane = PlaneAxial | PlaneCoronal | PlaneSagittal
PLANES: tuple[PlaneAxial, PlaneCoronal, PlaneSagittal] = ("axial", "coronal", "sagittal")
ScalarType = TypeVar("ScalarType", bound=number)
Vector2d = ndarray[tuple[Literal[2]], dtype[float64]]
Vector3d = ndarray[tuple[Literal[3]], dtype[float64]]
Shape2d = tuple[int, int]
Shape3d = tuple[int, int, int]
Shape4d = tuple[int, int, int, int]
ShapeType = TypeVar("ShapeType", bound=tuple[int, ...])
Image2d = ndarray[Shape2d, dtype[ScalarType]]
Image3d = ndarray[Shape3d, dtype[ScalarType]]
Image4d = ndarray[Shape4d, dtype[ScalarType]]
Mask2d = ndarray[Shape2d, dtype[bool_]]
Mask3d = ndarray[Shape3d, dtype[bool_]]
Mask4d = ndarray[Shape4d, dtype[bool_]]
RotationMatrix3x3 = ndarray[tuple[Literal[3], Literal[3]], dtype[float64]]
