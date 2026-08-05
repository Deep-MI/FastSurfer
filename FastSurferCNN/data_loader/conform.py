# Copyright 2019
# AI in Medical Imaging, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

# IMPORTS
import argparse
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Literal, TypedDict, TypeVar, cast

import nibabel as nib
import numpy as np
import torch
from nibabel.orientations import (
    aff2axcodes,
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
    ornt_transform,
)
from numpy import typing as npt

from FastSurferCNN.utils import (
    AffineMatrix3x3,
    AffineMatrix4x4,
    ScalarType,
    Shape1d,
    Vector3d,
    deprecated,
    logging,
    nibabelHeader,
    nibabelImage,
)
from FastSurferCNN.utils.arg_types import ImageSizeOption, OrientationType, VoxSizeOption
from FastSurferCNN.utils.arg_types import float_gt_zero_and_le_one as __conform_to_one_mm
from FastSurferCNN.utils.arg_types import img_size as __img_size
from FastSurferCNN.utils.arg_types import orientation as __orientation
from FastSurferCNN.utils.arg_types import target_dtype as __target_dtype
from FastSurferCNN.utils.arg_types import vox_size as __vox_size
from FastSurferCNN.utils.common import array_flags

AXCODES = ("lr", "pa", "is")

HELPTEXT = """
Script to conform an MRI brain image to UCHAR, RAS orientation, 
and 1mm or minimal isotropic voxels

USAGE:
conform.py  -i <input> -o <output> <options>
OR
conform.py  -i <input> --check_only <options>
Dependencies:
    Python 3.10+
    Numpy
    https://www.numpy.org
    Nibabel to read and write FreeSurfer data
    https://nipy.org/nibabel/
Original Author: Martin Reuter
Modified by: David Kügler
Date: May-12-2025
"""

LOGGER = logging.getLogger(__name__)

OrntArrayType = np.ndarray[tuple[int, Literal[2]], np.dtype[ScalarType]]
IntVector3d = np.ndarray[tuple[Literal[3]], np.dtype[np.int64]]

if TYPE_CHECKING:
    from torch import Tensor

    _TA = TypeVar("_TA", bound=np.ndarray | Tensor)
    _TB = TypeVar("_TB", bound=np.ndarray | Tensor)
else:
    _TA = TypeVar("_TA", bound=np.ndarray)
    _TB = TypeVar("_TB", bound=np.ndarray)


def __rescale_type(a: str) -> float | int | None:
    """
    Convert a string to a rescale value.

    Parameters
    ----------
    a : str
        String to extract the limit from.

    Returns
    -------
    float, int, or None
        The value to rescale to.

    Raises
    ------
    argparse.ArgumentTypeError
        If a cannot be converted.
    """
    try:
        return int(a)
    except ValueError:
        pass
    try:
        return float(a)
    except ValueError:
        pass
    if a.lower().strip() == "none":
        return None
    raise argparse.ArgumentTypeError(f"'{a}' is not an int, float or 'none'.")


def make_parser() -> argparse.ArgumentParser:
    """
    Create an Argument parser for the conform script.

    Returns
    -------
    argparse.ArgumentParser
        The parser object.
    """
    parser = argparse.ArgumentParser(usage=HELPTEXT)
    parser.add_argument(
        "--version",
        action="version",
        version="$Id: conform.py,v 1.0 2025/05/12 15:30:12 mreuter, kueglerd Exp $",
    )
    parser.add_argument(
        "--input", "-i",
        dest="input",
        required=True,
        help="The path to input image.",
    )
    parser.add_argument(
        "--output", "-o",
        dest="output",
        help="The path to output image.",
    )
    parser.add_argument(
        "--order",
        dest="order",
        help="The order of interpolation to use to interpolate (0=nearest, 1=linear(default), 2=quadratic, 3=cubic).",
        choices=(0, 1, 2, 3),
        type=int,
        default=1,
    )
    parser.add_argument(
        "--check_only",
        dest="check_only",
        default=False,
        action="store_true",
        help="Specifies that to only check whether the input image is conformed, and do not write an output image.",
    )
    parser.add_argument(
        "--seg_input",
        dest="seg_input",
        action="store_true",
        help="Specifies that the input is a seg image: The *default values* for dtype and rescale are changed to "
             "'integer' and 'none', which only means the dtype must be an integer and no rescaling is performed.",
    )
    parser.add_argument(
        "--vox_size",
        dest="vox_size",
        metavar="<float>|min|any",
        default=1.0,
        type=__vox_size,
        help="Specifies the target voxel size to conform to (default: 1, conform to 1mm). Options: <float> between 0 "
             "and 1 (target voxel size, isotropic, similar to mri_convert's --conform_size <size>); 'min' (conform to "
             "the minimum voxel size); 'any' (ignore this criteria, accept any voxel size even non-isotropic).",
    )
    parser.add_argument(
        "--conform_min",
        dest="vox_size",
        action="store_const",
        const="min",
        help="(Legacy, prefer --vox_size min for same functionality) Specifies that the image should be conformed to "
             "the minimal voxel size (used for high-res processing) -- overwrites --vox_size.",
    )
    parser.add_argument(
        "--img_size",
        dest="img_size",
        default="auto",
        metavar="<int>|auto|fov|any",
        type=__img_size,
        help="Specifies the image size to conform to, cube: same value for all three directions. Options: <int> "
             "(cube, sets dimension of the target image), 'auto' (cube, infer dimensions of image from largest "
             "field-of-view dimension, min. 256), 'fov' (may not be cube, set all three dimensions of image to keep "
             "the field of view the same) or 'any' (ignore this criteria, in practice similar to fov).",
    )
    parser.add_argument(
        "--rescale",
        metavar="<number>|none",
        dest="rescale",
        default=255,
        type=__rescale_type,
        help="Specifies whether image intensities should be rescaled. Options: <number> (default: 255, will robustly "
             "rescale intensities to this value, e.g. 0-255), 'none' (no intensity rescaling, i.e. all intensities "
             "stay the same and values outside of the data type are clamped to the data type range).",
    )
    advanced = parser.add_argument_group("Advanced options")
    advanced.add_argument(
        "--conform_to_1mm_threshold",
        type=__conform_to_one_mm,
        metavar="<float>",
        help="Advanced option to change the threshold beyond which images are conformed to 1 (default: infinity, "
             "all images are conformed to their minimum voxel size).",
    )
    advanced.add_argument(
        "--dtype",
        dest="dtype",
        default="uint8",
        metavar="<dtype name, e.g. 'uint8'>|any",
        type=__target_dtype,
        help="Specifies the target data type of the target image or 'any' (default: 'uint8', as in FreeSurfer).",
    )
    advanced.add_argument(
        "--orientation",
        dest="orientation",
        default="lia",
        metavar="native|XXX|soft-XXX",
        type=__orientation,
        help="Specify the target (data) orientation. Options: 'native' (will not change the orientation at all, i.e. "
             "ignore the orientation), <orientation string>, e.g. 'LIA' or 'RAS' (force perfect alignment with the "
             "scanner directions, as required by FreeSurfer and similar to mri_convert's --out_orientation), or "
             "'soft-<orientation string>' like 'soft-LIA' (primary directions aligned, but no resampling required).",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="If verbose, more detailed messages are printed.",
    )
    parser.add_argument(
        "--log",
        dest="logfile",
        default="",
        action="store",
        help="If specified, path to a log file that is written to.",
    )
    return parser

def options_parse():
    """
    Command line option parser.

    Returns
    -------
    options
        Object holding options.
    """
    args = make_parser().parse_args()
    if args.input is None:
        raise RuntimeError("Please specify input image")
    if not args.check_only and args.output is None:
        raise RuntimeError("Please specify output image")
    if args.check_only and args.output is not None:
        raise RuntimeError("You passed in check_only. Please do not also specify output image")

    if args.seg_input:
        if args.dtype == "uint8":
            args.seg_input = "integer"
        if args.rescale == 255:
            args.rescale = "none"
    del args.seg_input

    return args

SelfReorientation = TypeVar("SelfReorientation", bound="Reorientation")

class Reorientation:
    """
    A class to organize data reorientation to canonical orientations.

    Strict conform-style reorientations default to the same image-center convention as FreeSurfer
    (``shape / 2``). Soft/native reorientations that only reorder or flip axes opt into the voxel-center
    convention (``(shape - 1) / 2``) explicitly so they can roundtrip native geometry without resampling.

    Attributes
    ----------
    source_affine : AffineMatrix4x4
        The vox2ras affine of the input image.
    vox2vox : AffineMatrix4x4
        The vox2vox transformation matrix.
    source_shape : Shape1d
        The shape of the input image.
    target_shape : Shape1d
        The shape of the output image.
    tol : float
        The threshold to check to determine identity or reordering.
    """

    def __init__(self, source_affine: AffineMatrix4x4, source_shape: IntVector3d, tol: float = 1e-6):
        """
        Creates a transformation from """
        self._source_affine: AffineMatrix4x4 = source_affine
        self._vox2vox: AffineMatrix4x4 = np.eye(4)
        self.source_shape = source_shape
        self.target_shape = source_shape
        self.tol = tol
        self.voxel_center = False

    @property
    def vox2vox(self) -> AffineMatrix4x4:
        """
        Returns a readonly view of the target2source vox2vox transformation matrix.
        """
        with array_flags(self._vox2vox, writeable=False) as readonly_view:
            return readonly_view

    @property
    def source_affine(self) -> AffineMatrix4x4:
        """
        Returns a readonly view of the source affine matrix.
        """
        with array_flags(self._source_affine, writeable=False) as readonly_view:
            return readonly_view

    @classmethod
    def from_target_orientation(
            cls: type[SelfReorientation],
            source_affine: AffineMatrix4x4,
            target_orientation: OrientationType,
            shape: npt.ArrayLike,
            target_vox_size: npt.ArrayLike | None = None,
            target_shape: npt.ArrayLike | None = None,
            tol: float = 1e-6,
    ) -> SelfReorientation:
        """
        Determine the affine matrix to reorder and flip/interpolate data from source_affine to orientation.

        The resulting transform is a vox2vox from source to target. Strict orientations (for example ``"LIA"``)
        use the FreeSurfer conform center convention by default, while ``"soft-..."`` and ``"native"``
        reorientations preserve the voxel-center convention explicitly so discrete reorder/flip transforms can
        be mapped back to native space exactly.

        Parameters
        ----------
        source_affine : AffineMatrix4x4
            The input image affine to detect the reorientation operations.
        target_orientation : OrientationType
            The target orientation to reorient to.
        shape : array_like of shape (3,)
            The source shape of the data to reorder. If a "wrong shape" is passed, the vox2vox offset will corrupt.
        target_vox_size : array_like of shape (3,), optional
            The target voxel size in native coordinates, defaults to source_affine.
        target_shape : array_like of shape (3,), optional
            The target shape in native coordinates, defaults to shape.
        tol : float, default=1e-6
            Tolerance to identify reordering.

        Returns
        -------
        Reorientation
            An object holding the source_affine and the vox2vox transform to reorient data from source_affine to
            target_orientation.
        """
        _target_orientation = target_orientation.lower()
        if _target_orientation == "native":
            _target_orientation = "soft " + "".join(aff2axcodes(source_affine, AXCODES))

        source_vox_size = np.linalg.norm(source_affine[:3, :3], axis=0)
        if target_vox_size is None:
            vox_size: Vector3d = source_vox_size
        else:
            vox_size = np.asarray(target_vox_size, dtype=np.float64)
            if vox_size.size == 1:
                vox_size = np.full((3,), fill_value=vox_size.item())

        is_soft = _target_orientation.startswith("soft")
        if is_soft:
            # use strict source affine to compute soft re-orientation if intended
            _target_orientation = _target_orientation[5:]
        if any(c not in "lrpais" for c in _target_orientation) or len(_target_orientation) != 3:
            raise ValueError(f"Invalid target_orientation: {target_orientation}.")

        # by setting only the 3x3 rotational part here, we force from_target_affine to determine the translation
        # (as center-conserving)
        _reorder_ornt = axcodes2ornt(_target_orientation, AXCODES)

        # strict version of the source vox2ras so can generate a soft transform
        _source_affine = ornt2vox2vox(io_orientation(source_affine), shape,source_vox_size)

        # first, target affine without voxelsize to determine the reordering of voxel sizes
        target_strict_affine: AffineMatrix3x3 = ornt2vox2vox(_reorder_ornt, (1,) * 3, )[:3, :3]
        matrix = np.pad(np.linalg.inv(_source_affine[:3, :3]) @ target_strict_affine, ((0, 1), (0, 1)))
        reorder = io_orientation(matrix)

        vox_size_in_target = vox_size[reorder.astype(np.int16)[:, 0]]

        # second run, now with correct ordering of output voxel sizes in vox2ras
        target_strict_affine: AffineMatrix3x3 = ornt2vox2vox(_reorder_ornt, (1,) * 3, vox_size_in_target)[:3, :3]
        if not is_soft:
            return cls.from_target_affine(
                source_affine,
                target_strict_affine,
                shape,
                target_shape,
                tol,
                voxel_center=False,
            )

        source_ornt = io_orientation(source_affine)
        target_ornt = axcodes2ornt(_target_orientation, AXCODES)
        discrete_vox2vox = inv_ornt_aff(ornt_transform(source_ornt, target_ornt), shape)
        soft_rot_mat = np.linalg.inv(_source_affine[:3, :3]) @ target_strict_affine

        if np.allclose(soft_rot_mat, np.round(soft_rot_mat), atol=tol):
            soft_rot_mat = np.round(soft_rot_mat)

        same_target_shape = target_shape is None or np.array_equal(np.asarray(target_shape), np.asarray(shape))
        if same_target_shape and not does_vox2vox_rot_require_interpolation(soft_rot_mat, vox_eps=tol, rot_eps=tol):
            return cls.from_vox2vox(source_affine, discrete_vox2vox, shape, target_shape, tol, voxel_center=True)

        return cls.from_vox2vox(source_affine, soft_rot_mat, shape, target_shape, tol, voxel_center=True)

    @classmethod
    def from_target_affine(
            cls: type[SelfReorientation],
            source_affine: AffineMatrix4x4,
            target_affine: AffineMatrix4x4 | AffineMatrix3x3,
            shape: npt.ArrayLike,
            target_shape: npt.ArrayLike | None = None,
            tol: float = 1e-6,
            *,
            voxel_center: bool = False,
    ) -> SelfReorientation:
        """
        Determine the affine matrix to reorder and flip/interpolate data from source_affine to orientation.

        The resulting transform is a vox2vox from source to target.

        Parameters
        ----------
        source_affine : AffineMatrix4x4
            The input image affine to detect the reorientation operations.
        target_affine : AffineMatrix4x4, AffineMatrix3x3
            The target affine to reorient to.
        shape : array_like of shape (3,)
            The source shape of the data to reorder. If a "wrong shape" is passed, the vox2vox offset will corrupt.
        target_shape : array_like of shape (3,), optional
            The target shape in native coordinates, defaults to shape.
        tol : float, default=1e-6
            Tolerance to identify reordering.

        Returns
        -------
        Reorientation
            An object holding the source_affine and the vox2vox transform to reorient data from source_affine to
            target_orientation.

        Other Parameters
        ----------------
        voxel_center : bool, default=False
            Whether to use the voxel-center convention ``(shape - 1) / 2``. The default ``False`` matches
            FreeSurfer-style strict conforming. Soft/native callers should pass ``True``.
        """
        if target_affine.shape == (4, 4):
            v2v = np.linalg.inv(source_affine) @ target_affine

        elif target_affine.shape == (3, 3):
            v2v = np.linalg.inv(source_affine[:3, :3]) @ target_affine
            if np.allclose(v2v, np.round(v2v), atol=tol):
                v2v = np.round(v2v)

        elif target_affine.shape == (3, 4):
            v2v = np.linalg.inv(source_affine) @ np.concatenate([target_affine, np.eye(4)[3:]], axis=0)

        else:
            raise ValueError(f"target_affine must be of shape (3, 3), (4, 4) or (3, 4), but was {target_affine.shape}.")

        return cls.from_vox2vox(
            source_affine,
            v2v,
            shape,
            target_shape,
            tol,
            voxel_center=voxel_center,
        )

    @classmethod
    def from_vox2vox(
            cls: type[SelfReorientation],
            source_affine: AffineMatrix4x4,
            vox2vox: AffineMatrix3x3 | AffineMatrix4x4,
            shape: npt.ArrayLike,
            target_shape: npt.ArrayLike | None = None,
            tol: float = 1e-6,
            *,
            voxel_center: bool = False,
    ) -> SelfReorientation:
        """
        Determine the affine matrix to reorder and flip/interpolate data from source_affine to orientation.

        The resulting transform is a vox2vox from source to target.

        Parameters
        ----------
        source_affine : AffineMatrix4x4
            The input image affine to detect the reorientation operations.
        vox2vox : AffineMatrix4x4, AffineMatrix3x3
            The out2in vox2vox matrix to use, for a 3x3 matrix compute translation by assuming a rotation around the
            center (this is consistent with vox2vox of `scipy.ndimage.affine_transform`, `apply_image` and
            `nibabel.orientations.aff2axcodes`).
        shape : array_like of shape (3,)
            The source shape of the data to reorder. If a "wrong shape" is passed, the vox2vox offset will corrupt.
        target_shape : array_like of shape (3,), optional
            The target shape in native coordinates, defaults to shape.
        tol : float, default=1e-6
            Tolerance to identify reordering.

        Returns
        -------
        Reorientation
            An object holding the source_affine and the vox2vox transform to reorient data from source_affine to
            target_orientation.

        Other Parameters
        ----------------
        voxel_center : bool, default=False
            Whether to use the voxel-center convention ``(shape - 1) / 2``. The default ``False`` matches
            FreeSurfer-style strict conforming. Soft/native callers should pass ``True``.

        See Also
        --------
        FastSurferCNN.data_loader.conform.apply_vox2vox : Apply a vox2vox matrix to a 3D image.
        scipy.ndimage.affine_transform : Apply an affine transform to data.
        nibabel.orientations.aff2axcodes : Generate Orientation Codes from an affine matrix.
        """
        _shape = np.asarray(shape, dtype=np.int64)
        obj = cls(source_affine, _shape, tol)

        _target_shape = _shape if target_shape is None else np.asarray(target_shape, dtype=np.int64)
        obj.voxel_center = voxel_center
        if vox2vox.shape == (3, 3):
            # make center stay consistent, so in_voxcenter and out_voxcenter should have same ras coordinates
            translation = _translation_to_fix_center(
                vox2vox,
                _shape,
                _target_shape,
                voxel_center=voxel_center,
            )

            # expand the rotation matrix to 4x4 by 0 and translation
            rot_cols = np.concatenate([vox2vox, np.zeros((1, 3))], axis=0)
            trans_col = np.append(translation, 1)
            obj._vox2vox = np.concatenate([rot_cols, trans_col[:, None]], axis=1)
        elif vox2vox.shape == (4, 4):
            obj._vox2vox = cast(AffineMatrix4x4, vox2vox)

        elif vox2vox.shape == (3, 4):
            obj._vox2vox = np.concatenate([vox2vox, np.eye(4)[3:]], axis=0)

        else:
            raise ValueError(f"vox2vox must be of shape (3, 3), (4, 4) or (3, 4), but was {vox2vox.shape}.")
        obj.target_shape = obj.reorder_axes(_target_shape)
        return obj

    def snap_translation_to_grid_(self: SelfReorientation) -> None:
        """Modifies the translation to snap to the grid, if no rotation or scaling is present."""
        if not does_vox2vox_rot_require_interpolation(self.vox2vox):
            # here we check whether the vox2vox is a pure rotation that requires interpolation, if so we try to find a
            # close-by reorientation that does not require interpolation
            # Note, that we are rounding on the in-grid here, but this does not matter, because this vox2vox does not
            # rescale ("does_vox2vox_rot_require_interpolation"). Downscaling by integer values requires interpolation!
            ivox2vox = np.linalg.inv(self.vox2vox)
            self._vox2vox = np.linalg.inv(np.concatenate([ivox2vox[:, :3], np.fix(ivox2vox[:, 3:])], axis=1))

    def is_identity(self) -> bool:
        """Whether the internal vox2vox is the identity."""
        return np.allclose(self.vox2vox, np.eye(4))

    def __call__(self, image_data: _TA, order: int = 1, vox_eps: float = 1e-4, rot_eps: float | None = None) -> _TA:
        """
        Reorder and flip image_data such that the data is according to the source_affine and vox2vox attributes.

        Parameters
        ----------
        image_data : np.ndarray, torch.Tensor
            The image data to reorder/flip.
        order : int, default=1
            Order of interpolation (0=nearest,1=linear,2=quadratic,3=cubic).
        vox_eps : float, default=1e-4
            The epsilon for the voxelsize check.
        rot_eps : float, optional
            The epsilon for the affine rotation check, defaults to the attribute tol.

        Returns
        -------
        np.ndarray, torch.Tensor
            The reordered/flipped image data.
        """
        # is already target_orientation and no cropping
        if self.is_identity() and (self.target_shape is None or np.allclose(image_data.shape, self.target_shape)):
            return image_data
        else:  # is not target_affine yet
            if self.target_shape is None:
                raise ValueError("target_shape must be initialized for __call__")
            out_shape = np.asarray(self.target_shape)
            return apply_vox2vox(image_data, self.vox2vox, out_shape, order, vox_eps, rot_eps or self.tol)

    @property
    def target_affine(self) -> AffineMatrix4x4:
        """The target affine after reorientation."""
        return self.source_affine @ self.vox2vox

    @property
    def inverse(self: SelfReorientation) -> SelfReorientation:
        """
        A Reorientation object that can be used to reverse the reorientation of this object.
        """
        if self.is_identity():
            return self
        else:
            return self.from_vox2vox(
                self.target_affine,
                np.linalg.inv(self.vox2vox),
                self.target_shape,
                self.reorder_axes(self.source_shape),
                self.tol,
                voxel_center=self.voxel_center,
            )

    def reorder_axes(self, vector: np.ndarray[tuple[Literal[3]], np.dtype[ScalarType]]) \
            -> np.ndarray[tuple[Literal[3]], np.dtype[ScalarType]]:
        """
        Reorder a vector according to the vox2vox of this Reorientation.

        Parameters
        ----------
        vector : np.ndarray of shape (3,)
            The vector to reorder.

        Returns
        -------
        ndarray of shape (3,)
            Reordered vector.
        """
        return vector[io_orientation(self.vox2vox)[:, 0].astype(np.int64)]


def apply_orientation(arr: _TB | npt.ArrayLike, ornt: OrntArrayType) -> _TB:
    """
    Apply transformations implied by `ornt` to the first n axes of the array `arr`.

    Parameters
    ----------
    arr : array-like or torch Tensor of data with ndim >= n
        The image/data to reorient.
    ornt : (n,2) orientation array
       Orientation transform. ``ornt[N,1]` is flip of axis N of the array implied by `shape`, where 1 means no flip and
       -1 means flip. For example, if ``N==0`` and ``ornt[0,1] == -1``, and there's an array ``arr`` of shape `shape`,
       the flip would correspond to the effect of ``np.flipud(arr)``. ``ornt[:,0]`` is the transpose that needs to be
       done to the implied array, as in ``arr.transpose(ornt[:,0])``.

    Returns
    -------
    t_arr : ndarray or Tensor
       The data array `arr` transformed according to `ornt`.

    See Also
    --------
    nibabel.orientations.apply_orientation
        This function is an extension to `nibabel.orientations.apply_orientation`.
    """
    from nibabel.orientations import OrientationError
    from nibabel.orientations import apply_orientation as _apply_orientation

    # only import torch, if it is likely we are dealing with a tensor
    if hasattr(arr, "device"):
        from torch import is_tensor as _is_tensor

        if _is_tensor(arr):  # arr torch.Tensor
            ornt = np.asarray(ornt)
            n = ornt.shape[0]
            if arr.ndim < n:
                raise OrientationError("Data array has fewer dimensions than orientation")
            # apply ornt transformations
            flip_dims = np.nonzero(ornt[:, 1] == -1)[0].tolist()
            if len(flip_dims) > 0:
                arr = arr.flip(flip_dims)
            full_transpose = np.arange(arr.ndim)
            # ornt indicates the transpose that has occurred - we reverse it
            full_transpose[:n] = np.argsort(ornt[:, 0])
            return cast(_TB, arr.permute(*full_transpose))

    return _apply_orientation(arr, ornt)


@deprecated("Use apply_vox2vox or Reorientation.__call__ instead of map_image.")
def map_image(
        img: nibabelImage,
        out_affine: AffineMatrix4x4,
        out_shape: np.ndarray[Shape1d, np.dtype[np.integer]] | Iterable[int],
        ras2ras: AffineMatrix4x4 | None = None,
        order: int = 1,
        dtype: np.dtype[ScalarType] | npt.DTypeLike | None = None,
        vox_eps: float = 1e-4,
        rot_eps: float = 1e-6,
) -> npt.NDArray[ScalarType]:
    """
    Map image to new voxel space (RAS orientation).

    Parameters
    ----------
    img : nibabel.spatialimages.SpatialImage
        The src 3D image with data and affine set.
    out_affine : AffineMatrix4x4
        Trg image affine.
    out_shape : tuple[int, ...], np.ndarray of int
        The target shape information.
    ras2ras : AffineMatrix4x4, optional
        An additional mapping that should be applied (default=id to just reslice).
    order : int, default=1
        Order of interpolation (0=nearest,1=linear,2=quadratic,3=cubic).
    dtype : Type, None, default=None
        Target dtype of the resulting image (especially relevant for reorientation, None=keep dtype of img).
    vox_eps : float, default=1e-4
        The epsilon for the voxelsize check.
    rot_eps : float, default=1e-6
        The epsilon for the affine rotation check.

    Returns
    -------
    np.ndarray
        Mapped image data array.
    """
    if ras2ras is None:
        ras2ras = np.eye(4)

    # compute vox2vox from src to trg
    vox2vox = np.linalg.inv(out_affine) @ ras2ras @ img.affine
    # here we apply the inverse vox2vox (to pull back the src info to the target image)
    image_data = np.asarray(img.dataobj, dtype=dtype)
    return apply_vox2vox(image_data, vox2vox, out_shape=out_shape, order=order, vox_eps=vox_eps, rot_eps=rot_eps)


def apply_vox2vox(
        image_data: _TA,
        vox2vox: AffineMatrix4x4,
        out_shape: np.ndarray[tuple[int], np.dtype[np.integer]] | Iterable[int],
        order: int = 1,
        vox_eps: float = 1e-4,
        rot_eps: float = 1e-6,
    ) -> _TA:
    """
    Map image to new voxel space (RAS orientation).

    Parameters
    ----------
    image_data : np.ndarray
        The 3D image data.
    vox2vox : np.ndarray
        To-apply out2in vox2vox (!) for consistentcy with `scipy.ndimage.affine_transform`.
    out_shape : tuple[int, ...], np.ndarray
        The target shape information.
    order : int, default=1
        Order of interpolation (0=nearest,1=linear,2=quadratic,3=cubic).
    vox_eps : float, default=1e-4
        The epsilon for the voxelsize check.
    rot_eps : float, default=1e-6
        The epsilon for the affine rotation check.

    Returns
    -------
    np.ndarray
        Mapped image data array.
    """
    # convert frames to single image

    out_shape = tuple(out_shape)
    # if input has frames
    if image_data.ndim > 3:
        # if the output has no frames
        if len(out_shape) == 3:
            if any(s != 1 for s in image_data.shape[3:]):
                raise ValueError(f"Multiple input frames {tuple(image_data.shape)} not supported!")
            if hasattr(image_data, "device"):
                from torch import Tensor
                if isinstance(image_data, Tensor):
                    image_data = image_data.squeeze(tuple(range(3, image_data.ndim)))  # ty:ignore[invalid-assignment]
                else:
                    raise TypeError("image_data has a device attribute but is not a torch.Tensor!")
            else:
                image_data = np.squeeze(  # ty:ignore[invalid-assignment]
                    image_data,
                    axis=tuple(range(3, image_data.ndim)),
                )
        # if the output has the same number of frames as the input
        elif image_data.shape[3:] == out_shape[3:]:
            # add a frame dimension to vox2vox
            _vox2vox = np.eye(5, dtype=vox2vox.dtype)
            _vox2vox[:3, :3] = vox2vox[:3, :3]
            _vox2vox[:3, 4:] = vox2vox[:3, 3:]
            vox2vox = np.linalg.inv(_vox2vox)
        else:
            raise ValueError(
                    f"Input image and requested output shape have different frames: {image_data.shape} vs. {out_shape}!"
                )

    delta = np.abs(vox2vox - np.eye(4))
    off_diag = np.ones((4, 4)) - np.eye(4)
    if np.all(np.less(delta, np.eye(4) * max(vox_eps, rot_eps) + off_diag * rot_eps)) and image_data.shape == out_shape:
        # no interpolation needed, just use image_data
        return image_data

    if not does_vox2vox_rot_require_interpolation(vox2vox, vox_eps=vox_eps, rot_eps=rot_eps):

        # second condition: translations are integers
        if np.allclose(vox2vox[:, 3], np.round(vox2vox[:, 3]), atol=1e-4):
            inv_vox2vox = np.linalg.inv(vox2vox)
            # reorder axes, ornt is normally orientation2ras, but here we already inverted the matrix (as part of the
            # apply_vox2vox interface convention), so vox2vox is out2in
            ornt = io_orientation(inv_vox2vox).astype(np.int16)
            reordered = apply_orientation(image_data, ornt)

            # offset is the delta between vox coord zero and cut-off.
            # vox0_out[d] = output-space coordinate of input voxel (0,0,0) in output dimension d.
            vox0_out = np.round(inv_vox2vox[:3, 3]).astype(np.int16)
            # io_orientation returns one row per INPUT axis: ornt[k] = [output_axis, direction] for input axis k.
            # We need per OUTPUT dimension d: which input axis c_d maps to it, and whether it is flipped.
            # inv_ornt_axes[d] = c_d = the input axis that maps to output dimension d.
            inv_ornt_axes = np.argsort(ornt[:, 0]).astype(np.int16)
            # apply_orientation flips input axis c_d (at reordered position d) when ornt[c_d, 1] == -1.
            flipped_per_out_dim = (ornt[inv_ornt_axes, 1] == -1).astype(np.int16)
            # At reordered index 0 in dim d:
            #   not flipped: input axis c_d = 0  → out_d = trans_d  → offset = -trans_d = -vox0_out[d]
            #   flipped:     input axis c_d = reordered.shape[d]-1 → out_d = -sign*(reordered.shape[d]-1)+trans_d
            #                                                       → offset = reordered.shape[d] - 1 - vox0_out[d]
            # (note reordered.shape[d] = data.shape[c_d])
            offsets = -vox0_out + flipped_per_out_dim * (np.asarray(reordered.shape[:3], np.int16) - 1)
            # pad=0 => pad with zeros
            return crop_transform(reordered, offsets=offsets, target_shape=out_shape, pad=0)

    # TODO: in contrast to the type annotation, the following is not compatible with non-cpu torch.Tensor
    from scipy.ndimage import affine_transform

    return affine_transform(image_data, vox2vox, output_shape=out_shape, order=order)


def getscale(
        data: np.ndarray,
        dst_min: float | int,
        dst_max: float | int,
        f_low: float = 0.0,
        f_high: float = 0.999,
) -> tuple[float, float]:
    """
    Get offset and scale of image intensities to robustly rescale to dst_min..dst_max.

    Similar to the intensity rescaling used during FreeSurfer-style conforming.

    Parameters
    ----------
    data : np.ndarray
        Image data (intensity values).
    dst_min : float, int
        Future minimal intensity value.
    dst_max : float, int
        Future maximal intensity value.
    f_low : float, default=0.0
        Robust cropping at low end (0.0=no cropping).
    f_high : float, default=0.999
        Robust cropping at higher end (0.999=crop one thousandth of highest intensity).

    Returns
    -------
    float src_min
        (adjusted) offset.
    float
        Scale factor.
    """

    if f_low < 0. or f_high > 1. or f_low > f_high:
        raise ValueError("Invalid values for f_low or f_high, must be within 0 and 1.")

    # get min and max from source
    data_min = np.min(data)
    data_max = np.max(data)

    if data_min < 0.0:
        LOGGER.warning("Input image has value(s) below 0.0 !")
    LOGGER.info(f"Input:    min: {data_min}  max: {data_max}")

    if f_low == 0.0 and f_high == 1.0:
        return data_min, 1.0

    # compute non-zeros and total vox num
    num_nonzero_voxels = (np.abs(data) >= 1e-15).sum()
    num_total_voxels = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram (number of samples)
    bins = 1000
    hist, bin_edges = np.histogram(data, bins=bins, range=(data_min, data_max))

    # compute cumulative histogram
    cum_hist = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit: f_low fraction of total voxels
    lower_cutoff = int(f_low * num_total_voxels)
    binindex_lt_low_cutoff = np.flatnonzero(cum_hist < lower_cutoff)

    lower_binedge_index = 0
    # if we find any voxels
    if len(binindex_lt_low_cutoff) > 0:
        lower_binedge_index = binindex_lt_low_cutoff[-1] + 1

    src_min: float = bin_edges[lower_binedge_index].item()

    # get upper limit (cutoff only based on non-zero voxels, i.e. how many
    # non-zero voxels to ignore)
    upper_cutoff = num_total_voxels - int((1.0 - f_high) * num_nonzero_voxels)
    binindex_ge_up_cutoff = np.flatnonzero(cum_hist >= upper_cutoff)

    if len(binindex_ge_up_cutoff) > 0:
        upper_binedge_index = binindex_ge_up_cutoff[0] - 2
    elif np.isclose(cum_hist[-1], 1.0, atol=1e-6) or num_nonzero_voxels < 10:
        # if we cannot find a cutoff, check, if we are running into numerical
        # issues such that cum_hist does not properly account for the full hist
        # index -1 should always yield the last element, which is data_max
        upper_binedge_index = -1
    else:
        # If no upper bound can be found, this is probably a bug somewhere
        raise RuntimeError(f"rescale upper bound not found: f_high={f_high}")

    src_max: float = bin_edges[upper_binedge_index].item()

    # scale
    if src_min == src_max:
        LOGGER.warning("Scaling between src_min and src_max. The input image is likely corrupted!")
        scale = 1.0
    else:
        scale = (dst_max - dst_min) / (src_max - src_min)
    # logger.info
    LOGGER.info(f"rescale:  min: {src_min:8.3f}  max: {src_max:8.3f}  scale: {scale:8.5f}")

    return src_min, scale


def scalecrop(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        src_min: float,
        scale: float,
) -> np.ndarray:
    """
    Crop the intensity ranges to specific min and max values.

    Parameters
    ----------
    data : np.ndarray
        Image data (intensity values).
    dst_min : float
        Future minimal intensity value.
    dst_max : float
        Future maximal intensity value.
    src_min : float
        Minimal value to consider from source (crops below).
    scale : float
        Scale value by which source will be shifted.

    Returns
    -------
    np.ndarray
        Scaled image data.
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    LOGGER.info("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))
    return data_new


def rescale(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        f_low: float = 0.0,
        f_high: float = 0.999
) -> np.ndarray:
    """
    Rescale image intensity values (0-255).

    Parameters
    ----------
    data : np.ndarray
        Image data (intensity values).
    dst_min : float
        Future minimal intensity value.
    dst_max : float
        Future maximal intensity value.
    f_low : float, default=0.0
        Robust cropping at low end (0.0=no cropping).
    f_high : float, default=0.999
        Robust cropping at higher end (0.999=crop one thousandth of highest intensity).

    Returns
    -------
    np.ndarray
        Scaled image data.
    """
    src_min, scale = getscale(data, dst_min, dst_max, f_low, f_high)
    data_new = scalecrop(data, dst_min, dst_max, src_min, scale)
    return data_new


def conform(
        img: nibabelImage,
        order: int = 1,
        vox_size: VoxSizeOption = 1.0,
        img_size: ImageSizeOption = 256,
        dtype: npt.DTypeLike | None = np.uint8,
        orientation: OrientationType | None = "lia",
        threshold_1mm: float | None = None,
        rescale: int | float | None = 255,
        vox_eps: float = 1e-4,
        rot_eps: float = 1e-6,
        file_type: type[nibabelImage] | None = None,
        **kwargs,
) -> nibabelImage:
    """Conform an image to the geometry and dtype conventions expected by FastSurfer.

    This follows the general behavior of ``mri_convert -c``: it turns image intensity values into UCHAR, reslices
    images to standard position, places them into a cubic target grid (typically 256x256x256 at 1 mm and larger cubes
    for higher-resolution isotropic inputs) and enforces 1mm or minimum isotropic voxel sizes.

    Parameters
    ----------
    img : nib.spatialimages.SpatialImage
        Loaded source image.
    order : int, default=1
        Interpolation order (0=nearest, 1=linear, 2=quadratic, 3=cubic).
    vox_size : float, "min", None, default=1.0
        Conform the image to this voxel size, a specific smaller voxel size (0-1, for high-res), or automatically
        determine the 'minimum voxel size' from the image (value 'min'). This assumes the smallest of the three voxel
        sizes. `None` disables this criterion.
    img_size : int, "fov", "auto", None, default=256
        Conform the image to this image size, e.g. a specific smaller size (for example for high-res), or automatically
        determine the image size from the field of view ('fov' or 'auto', the former may yield non-cube-images). `None`
        disables this criterion.
    dtype : type, None, default=np.uint8
        The dtype to enforce in the image (default: UCHAR, as mri_convert -c). `None` disregards this criterion.
    orientation : "soft-<orientationcode>", "<orientationcode>", "native", None, default="lia"
        Which orientation of the data/affine to force, <orientationcode> is [rlapsi]{3}, ie.e. any of lia, ras, etc.,
        None disables this criterion.
    threshold_1mm : float, optional
        The threshold above which the image is conformed to 1mm. Ignore, if `None` (default).
    rescale : int, float, None, default=255
        Whether intensity values should be rescaled, it will either be the upper limit or None to ignore rescaling.
    vox_eps : float, default=1e-4
        The epsilon for the voxelsize check.
    rot_eps : float, default=1e-6
        The epsilon for the affine rotation check.
    file_type : class, optional
        The class to use for the image object. If None, will use the class of `img`.

    Returns
    -------
    nibabel.spatialimages.SpatialImage
        Conformed image.

    Other Parameters
    ----------------
    conform_vox_size : float, optional
        Legacy parameter for vox_size, overwrites vox_size.
    conform_to_1mm_threshold : float, optional
        Legacy parameter for threshold_1mm, overwrites threshold_1mm.

    Notes
    -----
    This implementation is similar to ``mri_convert -c``, but not intended to reproduce it exactly. In particular, we
    first interpolate (float image) and then rescale to uchar, while ``mri_convert -c`` does this in the opposite
    order. We compute the scale factor from the input to keep the behavior similar overall.
    """
    if "conform_to_1mm_threshold" in kwargs:
        LOGGER.warning("conform_to_1mm_threshold is deprecated, replaced by threshold_1mm and will be removed.")
        threshold_1mm = kwargs["conform_to_1mm_threshold"]
    if "conform_vox_size" in kwargs:
        LOGGER.warning("conform_vox_size is deprecated, replaced by vox_size and will be removed.")
        vox_size = kwargs["conform_vox_size"]

    _vox_size, _img_size = conformed_vox_img_size(img, vox_size, img_size, threshold_1mm=threshold_1mm, vox_eps=vox_eps)

    __vox_size: Vector3d = np.asarray(img.header.get_zooms()[:3] if _vox_size is None else _vox_size, dtype=np.float64)
    __img_size: IntVector3d = np.asarray(img.shape[:3] if _img_size is None else _img_size, dtype=np.int64)
    _orientation: OrientationType = "native" if orientation is None else orientation

    TargetImageClass: type[nibabelImage] = type(img) if file_type is None else file_type
    target_header: nibabelHeader = TargetImageClass.header_class.from_header(img.header)

    reorient = Reorientation.from_target_orientation(img.affine, _orientation, img.shape, __vox_size, __img_size)
    target_vox_size = reorient.reorder_axes(__vox_size)

    target_header.set_zooms(np.concatenate([target_vox_size, img.header.get_zooms()[3:]], axis=0))
    target_header.set_data_shape(np.concatenate([reorient.target_shape, img.shape[3:]], axis=0))

    if LOGGER.getEffectiveLevel() <= logging.DEBUG:
        with np.printoptions(precision=2, suppress=True):
            from re import sub
            LOGGER.debug("affine: " + sub("\\s+", " ", str(reorient.target_affine[:3, :3])))

    # derive target datatype from input
    target_dtype: np.dtype = np.dtype(img.get_data_dtype() if dtype is None else dtype)
    target_header.set_data_dtype(target_dtype)
    limits: None | tuple[int | float, int | float] = None

    if rescale is None and np.issubdtype(target_dtype, np.integer):
        limits = np.iinfo(target_dtype).min, np.iinfo(target_dtype).max
    elif isinstance(rescale, int | float):
        limits = 0, rescale
    elif rescale is not None:
        raise ValueError(f"Invalid rescale value: {rescale}")

    # reorient the image to the "corrected" (target) affine, always use float here
    mapped_data = reorient(img.get_fdata(), order=order, vox_eps=vox_eps, rot_eps=rot_eps)

    # get scale for conversion on original input before mapping to be more similar to mri_convert
    if rescale is not None:
        src_min, scale = getscale(np.asanyarray(img.dataobj), 0, rescale)

        where_data_zero = np.isclose(mapped_data, 0)
        # apply rescale
        mapped_data = scalecrop(mapped_data, 0, rescale, src_min, scale)
        # map zero in input to zero in output (usually background)
        mapped_data[where_data_zero] = 0

    # clip data to limits
    if limits is not None:
        mapped_data = np.clip(mapped_data, *limits)

    # mapped data is still float here, clip to integers now
    if np.issubdtype(target_dtype, np.integer):
        mapped_data = np.rint(mapped_data)
    new_img = TargetImageClass(mapped_data.astype(target_dtype), reorient.target_affine, target_header)

    # # make sure we store uchar
    # from nibabel.freesurfer import mghformat
    # try:
    #     new_img.set_data_dtype(target_dtype)
    # except mghformat.MGHError as e:
    #     if "not recognized" not in e.args[0]:
    #         raise
    #     dtype_codes = mghformat.data_type_codes.code.keys()
    #     codes = set(k.name for k in dtype_codes if isinstance(k, np.dtype))
    #     logging.getLogger(__name__).error(
    #         f"The data type '{dtype}' is not recognized for MGH images, switching to '{new_img.get_data_dtype()}' "
    #         f"(supported: {tuple(codes)})."
    #     )

    return new_img


def _translation_to_fix_center(
        vox2vox_o2i: AffineMatrix4x4 | AffineMatrix3x3,
        shape: IntVector3d,
        target_shape: IntVector3d | None = None,
        *,
        voxel_center: bool = True,
) -> Vector3d:
    """
    Calculate the translation to keep the center of the image fixed after applying the vox2vox transformation.

    This low-level helper keeps the voxel-center convention by default because it is also used by
    discrete orientation helpers such as :func:`ornt2vox2vox`.

    Parameters
    ----------
    vox2vox_o2i: AffineMatrix4x4, AffineMatrix3x3
        The vox2vox matrix for in2out transformation to keep the center fixed for.
    shape: IntVector3d
        The shape of the input data.
    target_shape: IntVector3d, optional
        The shape of the output data (in input data order, not target order), defaults to the same as shape.
    voxel_center : bool, default=True
        Whether to conserve the voxel-center convention ``(shape - 1) / 2``. Set to ``False`` to use the
        FreeSurfer conform convention ``shape / 2`` instead.

    Returns
    -------
    ndarray of shape (3,)
        The translation to keep the center of the image fixed after applying the vox2vox transformation.
    """
    _target_shape = shape if target_shape is None else target_shape
    center_offset = 1 if voxel_center else 0
    in_voxcenter = (np.asarray(shape) - center_offset) / 2
    vox2vox4 = np.pad(vox2vox_o2i[:3, :3], ((0, 1), (0, 1)))
    out_voxcenter = (
        _target_shape[io_orientation(vox2vox4.T)[:, 0].astype(np.int64)] - center_offset
    ) / 2
    #    voxO2ras @ voxO2voxI @ voxI_coord = voxO2ras @ voxO_coord
    # => voxO_coord = voxI2voxO @ voxI_coord = voxI2voxO_rot @ voxI_coord + voxI2voxO_trans
    # => voxO2voxI_trans = voxO_coord - voxI2voxO_rot @ voxI_coord
    return in_voxcenter - vox2vox_o2i[:3, :3] @ out_voxcenter


def target_shape_from_shape_scale(shape: npt.ArrayLike, scale: npt.ArrayLike) -> IntVector3d:
    """
    Calculate a target shape, that would enclose input shape after rescaling by scale.

    Parameters
    ----------
    shape : array_like
        The shape of the input data.
    scale : array_like
        The scale factors of the input data (out_vox_size / in_vox_size).

    Returns
    -------
    int
        The shape resized by the scale and rounded.
    """
    return np.ceil(np.asarray(shape) / np.asarray(scale)).astype(np.int64)


def ornt2vox2vox(ornt: OrntArrayType, shape: npt.ArrayLike, scale: npt.ArrayLike | None = None) -> AffineMatrix4x4:
    """
    Calculate the mid-centered vox2vox matrix of the orientation transform `ornt` (operation, not target orientation).

    This helper keeps the voxel-center convention ``(shape - 1) / 2`` so pure axis reorder/flip transforms
    stay compatible with nibabel orientation utilities and can be inverted without introducing a half-voxel shift.

    Parameters
    ----------
    ornt : array_like
        The orientation to transform by. Importantly, if nibabel calls it axcode LIA, this is a LIA->RAS transform.
    shape : array_like
        The shape of the (input) data.
    scale : array_like, optional
        The scaling factor of the (input) data, defaults to 1. If scale is not one, the assumed target shape will be
        shape scaled by `scale` as computed by `target_shape_from_shape_scale` (so out_vox_size / in_vox_size).

    Returns
    -------
    AffineMatrix4x4
        The transformation affine, a homogeneous affine if shape is passed. Importantly, the convention is that the
        matrix is out2in! `nib.orientations.aff2axcodes(vox2vox)` yields the `ornt` that was passed in, and
        so that the transformation can be applied by
        `apply_vox2vox(image_data, vox2vox, out_shape=target_shape_from_shape_scale(shape, scale))` or
        `scipy.ndimage.affine_transform(...)`.

    See Also
    --------
    target_shape_from_shape_scale : Generate the target shape from input scale and scale factor.
    apply_vox2vox : Apply a vox2vox matrix to a 3D image.
    """
    _ornt = np.asarray(ornt, dtype=int)
    # read dim from ornt
    if _ornt.shape[1] != 2:
        raise ValueError("shape of ornt must be (dim, 2)")
    dim = _ornt.shape[0]
    if scale is None:
        _scale = np.ones((dim,))
    elif isinstance(scale, int | float):
        _scale = np.full((dim,), scale)
    else:
        _scale = np.asarray(scale).flatten()
        if not isinstance(_scale, Sequence | np.ndarray) or not np.issubdtype(_scale.dtype, np.number):
            raise ValueError("scale must be None, a scalar or an sequence/array of shape (ornt.shape[0])!")
        elif _scale.size == 1:
            _scale = np.full((dim,), _scale.item())
        elif _scale.size != dim:
            raise ValueError("scale must be of size ornt.shape[0] or a scalar!")
    _shape = np.asarray(shape)
    if _shape.size != dim:
        raise ValueError(f"The length of shape needs to be equal ornt.shape[0] ({dim})!")
    target_shape = target_shape_from_shape_scale(_shape, _scale)

    # reorder, then flip
    vox2vox = np.zeros((dim + 1,) * 2, dtype=float)
    vox2vox[_ornt[:, 0], np.arange(dim)] = _ornt[:, 1] * _scale
    vox2vox[:, dim] = np.append(_translation_to_fix_center(vox2vox, _shape, target_shape=target_shape), 1)
    return vox2vox


def does_vox2vox_rot_require_interpolation(
        vox2vox: AffineMatrix4x4 | AffineMatrix3x3,
        vox_eps: float = 1e-4,
        rot_eps: float = 1e-6,
) -> bool:
    """
    Check whether the affine requires resampling/interpolation or whether reordering is sufficient.

    Parameters
    ----------
    vox2vox : Affinematrix4x4, AffineMatrix3x3
        The affine matrix (direction does not matter for this check).
    vox_eps : float, default=1e-4
        The epsilon for the voxelsize check.
    rot_eps : float, default=1e-6
        The epsilon for the affine rotation check.

    Returns
    -------
    bool
        Whether the vox2vox matrix requires resampling, integer-value downsampling (e.g. solvable by strides) by
        definition also requires interpolation.
    """
    def isclose(x, y, eps):
        return np.isclose(x, y, atol=eps, rtol=0)

    _v2v_pos = np.abs(vox2vox[:3, :3])
    # all values -1, 1 or 0 ==> False (does not require interpolation)
    return not np.all(np.logical_or(isclose(_v2v_pos, 1, eps=vox_eps), isclose(_v2v_pos, 0, eps=rot_eps)))


def is_conform(
        img: nibabelImage,
        vox_size: VoxSizeOption = 1.0,
        img_size: ImageSizeOption = 256,
        dtype: npt.DTypeLike | None = np.uint8,
        orientation: OrientationType | None = "lia",
        verbose: bool = True,
        vox_eps: float = 1e-4,
        eps: float = 1e-6,
        threshold_1mm: float = 0.0,
        **kwargs,
) -> bool:
    """
    Check if an image is already conformed or not.

    Defaults: Dimensions: 256x256x256, Voxel size: 1x1x1, LIA orientation, and data type UCHAR.

    Parameters
    ----------
    img : nib.analyze.SpatialImage
        Loaded source image.
    vox_size : float, "min", None, default=1.0
        Which voxel size to conform to. Can either be a float between 0.0 and 1.0, 'min' (to check, whether the image is
        conformed to the minimal voxels size, i.e. conforming to smaller, but isotropic voxel sizes for high-res), or
        None to disable the criteria.
    img_size : int, "fov", "auto", None, default=256
        Conform the image to this image size, a specific smaller size (0-1, for high-res), or automatically determine
        the target size: "fov": derive from the fov per dimension; "auto": get the largest "fov" and use this 3 times.
    dtype : Type, None, default=numpy.uint8
        Specifies the intended target dtype, if None the dtype check is disabled.
    orientation : "soft-XXX", "XXX", "native", None, default="lia"
        Whether to force the conforming to a specific orientation specified by XXX, e.g. LIA.
    verbose : bool, default=True
        If True, details of which conformance conditions are violated (if any) are displayed.
    vox_eps : float, default=1e-4
        Allowed deviation from zero for voxel size check.
    eps : float, default=1e-6
        Allowed deviation from zero for the orientation check. Small inaccuracies can occur through the inversion
        operation. Already conformed images are thus sometimes not correctly recognized. The epsilon accounts for
        these small shifts.
    threshold_1mm : float, optional
        Above this threshold the image is conformed to 1mm (default: None = ignore).

    Returns
    -------
    bool:
        Whether the image is already conformed.

    Notes
    -----
    This function only needs the header (not the data).
    """
    if "conform_to_1mm_threshold" in kwargs:
        LOGGER.warning("conform_to_1mm_threshold is deprecated, replaced by threshold_1mm and will be removed.")
        threshold_1mm = kwargs["conform_to_1mm_threshold"]
    if "conform_vox_size" in kwargs:
        LOGGER.warning("conform_vox_size is deprecated, replaced by vox_size and will be removed.")
        vox_size = kwargs["conform_vox_size"]
    if "check_dtype" in kwargs:
        LOGGER.warning("check_dtype is deprecated, replaced by dtype=None and will be removed.")
        if kwargs["check_dtype"] is False:
            dtype: npt.DTypeLike | None = None

    _vox_size, _img_size = conformed_vox_img_size(img, vox_size, img_size, threshold_1mm=threshold_1mm, vox_eps=vox_eps)

    # check 3d
    if len(img.shape) > 3 and img.shape[3] != 1:
        raise ValueError(f"Multiple input frames ({img.shape[3]}) not supported!")

    checks: dict[str, tuple[bool | Literal["IGNORED"], str]] = {
        "Number of Dimensions 3": (img.ndim == 3, f"image ndim {img.ndim}")
    }

    # check voxel size, drop voxel sizes of dimension 4 if available
    izoom = np.array(img.header.get_zooms())
    vox_size_text = f"image {'x'.join(map(str, izoom))}"
    if _vox_size is None:
        checks[f"Voxel Size {vox_size}"] = "IGNORED", vox_size_text
    else:
        if not isinstance(_vox_size, np.ndarray):
            raise TypeError("_vox_size should be numpy.ndarray here")
        vox_size_criteria = f"Voxel Size {vox_size}={'x'.join(map(str, _vox_size))}"
        checks[vox_size_criteria] = np.allclose(izoom[:3], _vox_size, atol=vox_eps, rtol=0), vox_size_text

    # check dimensions
    img_size_text = f"image dimensions {img.shape}"
    if img_size in (None, "fov") or _img_size is None:
        img_size_criteria = f"Dimensions {img_size}"
        checks[img_size_criteria] = "IGNORED", img_size_text
    else:
        img_size_criteria = f"Dimensions {img_size}={'x'.join(map(str, _img_size[:3]))}"
        checks[img_size_criteria] = np.array_equal(np.asarray(img.shape[:3]), _img_size), img_size_text

    # check orientation LIA
    affcode = "".join(aff2axcodes(img.affine))
    with np.printoptions(precision=2, suppress=True):
        orientation_text = "affine=" + re.sub("\\s+", " ", str(img.affine[:3, :3])) + f" => {affcode}"
    if orientation is None or orientation == "native":
        checks[f"Orientation {orientation}"] = "IGNORED", orientation_text
    else:
        is_soft = not orientation.startswith("soft")
        is_correct_orientation = is_orientation(img.affine, orientation[-3:], is_soft, eps)
        checks[f"Orientation {orientation.upper()}"] = is_correct_orientation, orientation_text

    # check dtype uchar
    dtype_text = f"dtype {img.get_data_dtype().name}"
    if dtype is None:
        checks["Dtype None"] = "IGNORED", dtype_text
    else:
        _dtype: npt.DTypeLike = to_dtype(dtype)
        if isinstance(_dtype, str | np.dtype):
            _dtype_name = np.dtype(_dtype).name
        elif isinstance(_dtype, type):
            _dtype_name = _dtype.__name__
        else:
            _dtype_name = str(_dtype)
        checks[f"Dtype {_dtype_name}"] = np.issubdtype(img.get_data_dtype(), _dtype), dtype_text

    _is_conform = all(map(lambda x: x[0], checks.values()))

    logger = logging.getLogger(__name__)
    if not _is_conform:
        logger.log(logging.INFO, "The input image is not conformed.")

    if verbose:
        conform_str = ""
        if _vox_size is not None and not np.allclose(_vox_size, 1.0):
            if np.allclose(_vox_size[0], _vox_size, atol=1e-2):
                conform_str = f"{np.round(_vox_size[0], decimals=2):.2f}-"
            else:
                with np.printoptions(precision=2, suppress=True):
                    conform_str = str(_vox_size) + "-"
        logger.info(f"A {conform_str}conformed image must satisfy the following criteria:")
        for condition, (value, message) in checks.items():
            if isinstance(value, bool):
                value = "GOOD" if value else "BUT"
            logger.info(f" - {condition:<30}: {value} {message}")
    return _is_conform


def to_dtype(dtype: str | np.dtype | type | npt.DTypeLike) -> npt.DTypeLike:
    """
    Make sure to convert dtype to a numpy compatible dtype.

    Parameters
    ----------
    dtype : str, np.dtype
        Use this to determine the dtype.

    Returns
    -------
    numpy.typing.DTypeLike
        The dtype extracted.
    """
    if isinstance(dtype, str) and dtype.lower() == "uchar":
        dtype = "uint8"
    if isinstance(dtype, str):
        suptype = dtype.lower()[4:]
        if suptype in ("int", "signed"):
            return np.signedinteger
        elif suptype in ("uint", "unsigned"):
            return np.unsignedinteger
        elif hasattr(np, suptype):
            return getattr(np, suptype)
        return np.dtype(dtype)
    return dtype


def is_orientation(
        affine: AffineMatrix4x4,
        target_orientation: OrientationType = "lia",
        soft: bool = False,
        eps: float = 1e-6,
):
    """
    Checks whether the affine is LIA-oriented.

    Parameters
    ----------
    affine : AffineMatrix4x4
        The affine to check.
    target_orientation : OrientationType, default="lia"
        The target orientation for which to check the affine for.
    soft : bool, default=True
        Whether the orientation is required to be "exactly" (strict) LIA or just similar (soft) (i.e. it is roughly
        oriented as `target_orientation`).
    eps : float, default=1e-6
        The threshold in strict mode.

    Returns
    -------
    bool
        Whether the affine is LIA-oriented.
    """
    if "".join(aff2axcodes(affine, tol=eps)).lower() == target_orientation.lower():
        if soft:
            return True
    else:
        return False

    return does_vox2vox_rot_require_interpolation(affine / np.linalg.norm(affine, axis=0), rot_eps=eps, vox_eps=eps)


def conformed_vox_img_size(
        img: nibabelImage,
        vox_size: VoxSizeOption,
        img_size: ImageSizeOption,
        threshold_1mm: float | None = None,
        vox_eps: float = 1e-4,
        **kwargs,
) -> tuple[Vector3d | None, IntVector3d | None]:
    """
    Extract the voxel size and the image size.

    This function only needs the header (not the data).

    Parameters
    ----------
    img : nib.spatialimages.SpatialImage
        Loaded source image.
    vox_size : float, "min", None
        The voxel size parameter to use: either a voxel size as float, or the string "min" to automatically find a
        suitable voxel size (smallest per-dimension voxel size). None disregards the criterion (output also None).
    img_size : int, "fov", "auto", None
        The image size parameter: either an image size as int, the string "fov" to automatically derive a suitable
        image size (field of view), or "auto" like "fov" but largest size in every direction.
        `None` disregards the criterion, if vox_size is also `None`, else like "auto".
    threshold_1mm : float, optional
        The threshold for which image voxel size should be conformed to 1mm instead of conformed to the smallest voxel
        size (default or None: do not apply the threshold).
    vox_eps : float, default=1e-4
        The threshold to compare vox_sizes (differences below this are ignored).

    Returns
    -------
    np.ndarray of floats, None
        The determined voxel size to conform the image to (still in native orientation), shape: 3.
    np.ndarray of ints, None
        The size of the image adjusted to the conformed voxel size (still in native orientation), shape: 3.
    """
    if "conform_to_1mm_threshold" in kwargs:
        LOGGER.warning("conform_to_1mm_threshold is deprecated, replaced by threshold_1mm and will be removed.")
        threshold_1mm = kwargs["conform_to_1mm_threshold"]
    if "conform_vox_size" in kwargs:
        LOGGER.warning("conform_vox_size is deprecated, replaced by vox_size and will be removed.")
        vox_size = kwargs["conform_vox_size"]

    target_vox_size: Vector3d | None
    target_img_size: IntVector3d | None
    MAX_VOX_SIZE = 1.0
    MAX_DIMENSION = 256
    # number of decimals to round voxel sizes to, so that vox_eps-sized float noise does not affect results
    decimals = int(np.ceil(-np.log10(vox_eps)))
    # this is similar to mri_convert --conform_min, note, vox_size == 'auto' is extra, but not covered by VoxSizeOption
    if isinstance(vox_size, str) and (vox_size := cast(VoxSizeOption, vox_size.lower())) in ["min", "auto"]:
        # find minimal voxel side length
        min_vox_size = np.round(np.min(img.header.get_zooms()[:3]), decimals=decimals)
        # set to 1 mm if larger than that
        _conformed_vox_size = min(min_vox_size, MAX_VOX_SIZE)
        if threshold_1mm and _conformed_vox_size > threshold_1mm:
            _conformed_vox_size = MAX_VOX_SIZE
        target_vox_size = np.full((3,), _conformed_vox_size)
    # this is similar to mri_convert --conform_size <float>
    elif isinstance(vox_size, float | int) and 0.0 < vox_size <= MAX_VOX_SIZE:
        target_vox_size = np.full((3,), vox_size)
    elif vox_size is None:
        target_vox_size = None
    else:
        raise ValueError(f"Invalid value for vox_size passed: {vox_size}.")
    if img_size is None and target_vox_size is not None:
        # if we did specify a vox_size, no image size. use the field of view (which is essentially the old image size
        # scaled with the voxel size)
        img_size = "fov"
    if img_size is None:
        target_img_size = None
    elif isinstance(img_size, int) and img_size > 0:
        target_img_size = np.full((3,), img_size)
    elif isinstance(img_size, str) and (_img_size := img_size.lower()) in ["fov", "auto"]:
        thres = abs(1.0 - (threshold_1mm or 1.0))
        if target_vox_size is not None and np.allclose(target_vox_size, 1.0, atol=thres) and _img_size == "auto":
            target_img_size = np.full((3,), MAX_DIMENSION)
        # (other voxel sizes may use different sizes)
        else:
            target_img_size = np.array(img.shape[:3])
            if target_vox_size is not None:
                # correct sizes for changing voxel size (if voxel size is changing)
                # compute field of view dimensions in mm (in native orientation)
                fov = np.array(np.round(img.header.get_zooms()[:3], decimals=decimals)) * target_img_size
                # number of voxels needed to cover the field of view
                n_vox = fov / target_vox_size
                # n_vox is integer when the fov is a multiple of the voxel size, but floating-point
                # error makes it only approximately so. Storing the zoom as float32 leaves a relative
                # residual of ~1e-8 that scales with the voxel count, i.e. machine precision rather
                # than a voxel-size tolerance. Snap counts that are integer within that relative error
                # and round up only genuine partial voxels.
                rounded = np.rint(n_vox)
                target_img_size = np.where(
                    np.isclose(n_vox, rounded, rtol=1e-6, atol=0.0), rounded, np.ceil(n_vox)
                ).astype(int)
        # use cube (same size in all directions) with MAX_DIMENSION in each direction as minimum
        if _img_size == "auto":
            target_img_size = np.full_like(np.maximum(MAX_DIMENSION, target_img_size), np.amax(target_img_size))
    else:
        raise ValueError("Invalid value for img_size passed.")
    return target_vox_size, target_img_size


def check_affine_in_nifti(
        img: nib.Nifti1Image | nib.Nifti2Image,
        logger: logging.Logger | None = None,
) -> bool:
    """
    Check the affine in nifti Image.

    Sets affine with qform, if it exists and differs from sform.
    If qform does not exist, voxel sizes between header information and information
    in affine are compared.
    In case these do not match, the function returns False (otherwise True).

    Parameters
    ----------
    img : nib.Nifti1Image, nib.Nifti2Image
        Loaded nifti-image.
    logger : logging.Logger, optional
        Logger object or None (default) to log or print an info message to stdout (for None).

    Returns
    -------
    bool
        False, if voxel sizes in affine and header differ.
    """
    check = True
    message = ""

    if img.header["qform_code"] != 0 and not np.allclose(img.get_sform(), img.get_qform(), atol=0.001):
        message = (
            f"#############################################################\n"
            f"WARNING: qform and sform transform are not identical!\n"
            f" sform-transform:\n{img.header.get_sform()}\n"
            f" qform-transform:\n{img.header.get_qform()}\n"
            f"You might want to check your Nifti-header for inconsistencies!\n"
            f"!!! Affine from qform transform will now be used !!!\n"
            f"#############################################################"
        )
        # Set sform with qform affine and update the best affine in header
        img.set_sform(img.get_qform())
        img.update_header()

    else:
        # Check if affine correctly includes voxel information and print Warning/
        # Exit otherwise
        vox_size_header = img.header.get_zooms()

        # voxel size in xyz direction from the affine
        vox_size_affine = np.sqrt((img.affine[:3, :3] * img.affine[:3, :3]).sum(0))

        if not np.allclose(vox_size_affine, vox_size_header, atol=1e-3):
            message = (
                f"#############################################################\n"
                f"ERROR: Invalid Nifti-header! Affine matrix is inconsistent with "
                f"Voxel sizes. \nVoxel size (from header) vs. Voxel size in affine:\n"
                f"{tuple(vox_size_header[:3])}, {tuple(vox_size_affine)}\n"
                f"Input Affine----------------\n{img.affine}\n"
                f"#############################################################"
            )
            check = False

    if logger is not None:
        logger.info(message)

    else:
        LOGGER.info(message)

    return check

def print_options(options: dict):

    options = dict(options)
    for key in ("vox_size", "img_size", "dtype", "orientation"):
        if options.get(key, None) is None:
            options[key] = "any"

    msg = (
        "Image Conform Parameters:",
        "",
        "- verbosity: {verbose}",
        "- input volume: {input}",
        "- check only: {check_only}",
        "- dtype: {dtype}",
        "- voxel size: {vox_size}",
        "- round voxel size to 1mm if > threshold: {conform_to_1mm_threshold}",
        "- image size: {img_size}",
        "- affine orientation: {orientation}",
        "- log: stdout " + ("and '{logfile}'" if options["logfile"] else "only"),
    )
    if not options["check_only"]:
        msg += (
           "- output volume: {output}",
           "- order: {order}",
           "- rescale: {rescale}",
        )

    _logger = logging.getLogger(__name__ + ".print_options")
    for m in msg:
        if m is not None:
            _logger.info(m.format(**options))


def _crop_transform_make_indices(image_shape: Sequence[int], offsets: Sequence[int], target_shape: Sequence[int]) \
        -> tuple[list[tuple[int, int]] | None, tuple[slice, ...]]:
    """
    Create the indexing tuple and return padding tuples for the last N dimensions.

    Parameters
    ----------
    image_shape : Sequence[int]
        The shape of the image from which a region is to be cropped.
    offsets : Sequence[int]
        Exact location within the image from which the cropping should start, negative offsets pad.
    target_shape : Sequence[int]
        The desired shape of the cropped region.

    Returns
    -------
    paddings: list of 2-tuples of paddings or None
        A list of per-axis tuples of the padding to apply to the slice to get the target_shape.
    indices : tuple of indices
        A tuple of per-axis indices to index in the data to get the target_shape.
    """
    if len(offsets) != len(target_shape):
        raise ValueError(f"offsets {offsets} and target shape {target_shape} must be same length.")
    if len(offsets) > len(image_shape):
        raise ValueError("offsets too long for image")
    batch_dims = len(image_shape) - len(offsets)
    indices: list[slice] = [slice(None)] * batch_dims
    paddings: list[tuple[int, int]] = []
    any_pad = False
    for offset, t_shape, i_shape in zip(offsets, target_shape, image_shape[batch_dims:], strict=False):
        crop_end = min(offset + t_shape, i_shape)
        indices.append(slice(max(0, offset), crop_end))
        pads = (max(0, -offset), max(0, offset + t_shape - crop_end))
        paddings.append(pads)
        any_pad = any_pad or any(p != 0 for p in pads)

    return paddings if any_pad else None, tuple(indices)


def _crop_transform_pad_fn(image: _TA, pad_tuples: list[tuple[int, int]], pad: str | float) \
        -> Callable[[_TA], _TA] | None:
    """
    Generate a parameterized pad function.

    Parameters
    ----------
    image : np.ndarray, torch.Tensor
        Input image.
    pad_tuples : List[Tuple[int, int]]
        List of padding tuples for each axis.
    pad : str, float, torch.Tensor
        The padding strategy to use when padding is required, if int, pad with that value, if str, use that mode (e.g.
        "edge", "reflect", "symmetric" for numpy.ndarray and "reflect", "replicate", "circular", for torch.Tensor).

    Returns
    -------
    partial, None
        A partial function to pad the image or None.
    """
    if all(p1 == 0 and p2 == 0 for p1, p2 in pad_tuples):
        return None

    if isinstance(pad, str):
        # TorchPadModes are valid for torch, NumpyPadModes are valid for numpy (exhaustive lists ignoring "constant")
        TorchPadModes = ("reflect", "replicate", "circular")
        NumpyPadModes = ("edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect", "symmetric", "wrap")
        if isinstance(image, np.ndarray) and pad not in NumpyPadModes:
            raise ValueError("Invalid value for `pad` for numpy array!")
        elif isinstance(image, torch.Tensor) and pad not in TorchPadModes:
            raise ValueError("Invalid value for `pad` for torch tensor!")
        mode = pad
    else:
        mode = "constant"
    _func : Callable[[_TA], _TA]
    if isinstance(image, np.ndarray):
        _pad_width = [(0, 0)] * (image.ndim - len(pad_tuples)) + pad_tuples
        _func = partial(np.pad, mode=mode, pad_width=_pad_width)  # ty:ignore[invalid-assignment]
        if mode == "constant":
            _func = partial(_func, constant_values=pad)
    else:  # Tensor
        from itertools import chain

        from torch.nn.functional import pad as _pad

        _pad_value = list(chain.from_iterable(reversed(pad_tuples)))
        _func = partial(_pad, mode=mode, pad=_pad_value)  # ty:ignore[invalid-assignment]
        if mode == "constant":
            _func = partial(_func, value=pad)

    return _func


def crop_transform(
        image: _TA,
        offsets: Sequence[int] | None = None,
        target_shape: Sequence[int] | None = None,
        out: _TA | None = None,
        pad: int = 0,
) -> _TA:
    """
    Perform a crop transform of the last N dimensions on the image data.

    Cropping does not interpolate the image, but "just removes" border pixels/voxels. Negative offsets lead to padding.

    Parameters
    ----------
    image : np.ndarray, torch.Tensor
        Image of size [..., D_1, D_2, ..., D_N], where D_1, D_2, ..., D_N are the N image dimensions.
    offsets : Sequence[int], optional
        Offset of the cropped region for the last N dimensions (default: center crop, less crop/pad towards index 0).
        Negative offsets pad.
    target_shape : Sequence[int], optional
        If defined, target_shape specifies the target shape of the "cropped region", else the crop will be centered
        cropping offset[dim] voxels on each side (then the shape is derived by subtracting 2x the dimension-specific
        offset). target_shape should have the same number of elements as offsets. May be implicitly defined by out.
    out : np.ndarray, torch.Tensor, optional
        Array to store the cropped image in (optional), can be a view on image for memory-efficiency.
    pad :  int, str, default=0/zero-pad
        Padding strategy to use when padding is required, if int, pad with that value.

    Returns
    -------
    out : np.ndarray, torch.Tensor
        The image (stack) cropped in the last N dimensions by offsets to the shape target_shape, or if target_shape is
        not given image.shape[i+2] - 2*offset[i].

    Raises
    ------
    ValueError
        If neither offsets nor target_shape nor out are defined.
    ValueError
        If out is not target_shape.
    TypeError
        If the type of image is not an np.ndarray or a torch.Tensor.
    RuntimeError
        If the dimensionality of image, out, offset or target_shape is invalid or inconsistent.

    See Also
    --------
    numpy.pad
        For additional information refer to numpy.pad function.

    Notes
    -----
    Either offsets, target_shape or out must be defined.
    """
    if target_shape is None and out is not None:
        target_shape = out.shape

    # check the type of offsets
    if offsets is None:
        if target_shape is None:
            raise ValueError("Either target_shape or offsets must be defined!")
        _target_shape = image.shape[:-len(target_shape)] + tuple(target_shape)
        offsets = tuple(int((i - t) / 2) for t, i in zip(_target_shape, image.shape, strict=False))
        len_off = len(offsets)
    else:
        len_off = len(offsets)
        if target_shape is None:
            _target_shape = image.shape[:-len_off] + tuple(
                i - 2 * o for i, o in zip(image.shape[-len_off:], offsets, strict=False)
            )
        elif len_off == len(target_shape):
            _target_shape = tuple(
                i if t == -1 else t
                for i, t in zip(image.shape[-len_off:], target_shape, strict=False)
            )
            _target_shape = image.shape[:-len_off] + _target_shape
        else:
            raise ValueError("Incompatible offset and target_shape dimensionality (at least once).")

    if len_off > image.ndim:
        raise RuntimeError("shape of offsets is larger than dim of image allows.")

    pad_tuples, indices = _crop_transform_make_indices(
        image.shape, offsets, _target_shape
    )
    outval: _TA = image[indices]  # ty:ignore[invalid-argument-type, invalid-assignment]
    if pad_tuples is not None:
        pad_fn = _crop_transform_pad_fn(image, pad_tuples, pad)
        outval = outval if pad_fn is None else pad_fn(outval)

    if out is None:
        return outval
    else:
        out[:] = outval  # ty:ignore[invalid-assignment]
        # return out instead of outval to reduce memory
        return out


if __name__ == "__main__":
    # Command Line options are error checking done here
    try:
        options = options_parse()
    except RuntimeError as e:
        sys.exit("ERROR: " + str(e.args[0] if len(e.args) == 1 else e.args))

    logging.setup_logging(options.logfile) # logging to only the console

    if options.verbose:
        print_options(vars(options))

    LOGGER.info(f"Reading input: {options.input} ...")
    image = cast(nibabelImage, nib.load(options.input))

    if not isinstance(image, nib.analyze.SpatialImage):
        sys.exit(f"ERROR: Input image is not a spatial image: {type(image).__name__}")
    if len(image.shape) > 3 and image.shape[3] != 1:
        sys.exit(f"ERROR: Multiple input frames ({image.shape[3]}) not supported!")

    class _OptKwargs(TypedDict, total=False):
        threshold_1mm: float

    class OptKwargs(_OptKwargs):
        vox_size: VoxSizeOption
        img_size: ImageSizeOption
        dtype: npt.DTypeLike | None
        orientation: OrientationType | None
        verbose: bool

    opt_kwargs : OptKwargs = {
        "dtype": options.dtype if options.dtype != "any" else None,
        "vox_size":  options.vox_size,
        "img_size": options.img_size,
        "orientation": options.orientation,
        "verbose": options.verbose,
    }

    if hasattr(options, "conform_to_1mm_threshold"):
        opt_kwargs["threshold_1mm"] = options.conform_to_1mm_threshold

    try:
        image_is_conformed = is_conform(image, **opt_kwargs)
    except ValueError as e:
        sys.exit(e.args[0])

    if image_is_conformed:
        LOGGER.info(f"Input {options.input} is already conformed! Exiting.\n")
        sys.exit(0)
    else:
        # Note: if check_only, a non-conforming image leads to an error code, this
        # result is needed in recon_surf.sh
        if options.check_only:
            LOGGER.info("check_only flag provided. Exiting without conforming input image.\n")
            sys.exit(1)

    # If image is nifti image
    if options.input[-7:] == ".nii.gz" or options.input[-4:] == ".nii":
        if not check_affine_in_nifti(cast(nib.Nifti1Image | nib.Nifti2Image, image)):
            sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

    if options.output[-7:] == ".nii.gz" or options.output[-4:] == ".nii":
        file_type = nib.Nifti2Image
    elif options.output[-4:] == ".mgz":
        file_type = nib.MGHImage
    else:
        sys.exit("conform only supports mgz and nifti.")

    try:
        # new_image will be of class file_type
        new_image = conform(image, order=options.order, rescale=options.rescale, file_type=file_type, **opt_kwargs)
    except ValueError as e:
        sys.exit(e.args[0])
    LOGGER.info(f"Writing conformed image: {options.output}")

    nib.save(new_image, options.output)

    sys.exit(0)
