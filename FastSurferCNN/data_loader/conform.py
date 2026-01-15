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
from typing import TYPE_CHECKING, Literal, TypeVar, Union, cast

import nibabel
import nibabel as nib
import numpy as np
import numpy.typing as npt
from nibabel.freesurfer.mghformat import MGHHeader

if TYPE_CHECKING:
    import torch

from FastSurferCNN.utils import ScalarType, logging, nibabelImage
from FastSurferCNN.utils.arg_types import ImageSizeOption, OrientationType, StrictOrientationType, VoxSizeOption
from FastSurferCNN.utils.arg_types import float_gt_zero_and_le_one as __conform_to_one_mm
from FastSurferCNN.utils.arg_types import img_size as __img_size
from FastSurferCNN.utils.arg_types import orientation as __orientation
from FastSurferCNN.utils.arg_types import target_dtype as __target_dtype
from FastSurferCNN.utils.arg_types import vox_size as __vox_size

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

_TA = TypeVar("_TA", bound=Union[np.ndarray, "torch.Tensor"])
_TB = TypeVar("_TB", bound=Union[np.ndarray, "torch.Tensor"])


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


def to_target_orientation(
        image_data: _TA,
        source_affine: npt.NDArray[float],
        target_orientation: StrictOrientationType,
) -> tuple[_TA, Callable[[_TB], _TB]]:
    """
    Reorder and flip image_data such that the data is in orientation. This will always be without interpolation.

    Parameters
    ----------
    image_data : np.ndarray, torch.Tensor
        The image data to reorder/flip.
    source_affine : npt.NDArray[float]
        The affine to detect the reorientation operations.
    target_orientation : StrictOrientationType
        The target orientation to reorient to.

    Returns
    -------
    np.ndarray, torch.Tensor
        The data flipped and reordered so it is close to LIA (same type as image_data).
    Callable[[np.ndarray], np.ndarray], Callable[[torch.Tensor], torch.Tensor]
        A function that flips and reorders the data back (returns same type as output).
    """
    reorient_ornt, unorient_ornt = orientation_to_ornts(source_affine, target_orientation)

    if np.any([reorient_ornt[:, 1] != 1, reorient_ornt[:, 0] != np.arange(reorient_ornt.shape[0])]):  # is not lia yet
        def back_to_native(data: _TB) -> _TB:
            return apply_orientation(data, unorient_ornt)

        return apply_orientation(image_data, reorient_ornt), back_to_native
    else:  # data is already in lia
        def do_nothing(data: _TB) -> _TB:
            return data

        return image_data, do_nothing


def orientation_to_ornts(
        source_affine: npt.NDArray[float],
        target_orientation: StrictOrientationType,
) -> tuple[npt.NDArray[int], npt.NDArray[int]]:
    """
    Determine the nibabel `ornt` Array to reorder and flip data from source_affine such that the data is in orientation.

    Parameters
    ----------
    source_affine : npt.NDArray[float]
        The affine to detect the reorientation operations.
    target_orientation : StrictOrientationType
        The target orientation to reorient to.

    Returns
    -------
    npt.NDArray[int]
        The `ornt` transform from source_affine to target_orientation.
    npt.NDArray[int]
        The `ornt` transform back from target_orientation to source_affine.
    """
    from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

    source_ornt = io_orientation(source_affine)
    target_ornt = axcodes2ornt(target_orientation.upper())
    reorient_ornt = ornt_transform(source_ornt, target_ornt)
    unorient_ornt = ornt_transform(target_ornt, source_ornt)
    return reorient_ornt.astype(int), unorient_ornt.astype(int)


def apply_orientation(arr: _TB | npt.ArrayLike, ornt: npt.NDArray[int]) -> _TB:
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
    from torch import is_tensor as _is_tensor

    if _is_tensor(arr):
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
        t_arr = arr.permute(*full_transpose)
        return t_arr
    else:
        return _apply_orientation(arr, ornt)


def map_image(
        img: nibabelImage,
        out_affine: npt.NDArray[float],
        out_shape: tuple[int, ...] | npt.NDArray[int] | Iterable[int],
        ras2ras: npt.NDArray[np.number] | None = None,
        order: int = 1,
        dtype: np.dtype[ScalarType] | npt.DTypeLike | None = None,
        vox_eps: float = 1e-4,
        rot_eps: float = 1e-6,
) -> npt.NDArray[ScalarType]:
    """
    Map image to new voxel space (RAS orientation).

    Parameters
    ----------
    img : nibabelImage
        The src 3D image with data and affine set.
    out_affine : np.ndarray
        Trg image affine.
    out_shape : tuple[int, ...], np.ndarray
        The target shape information.
    ras2ras : np.ndarray, optional
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
    # convert frames to single image

    out_shape = tuple(out_shape)
    # if input has frames
    if image_data.ndim > 3:
        # if the output has no frames
        if len(out_shape) == 3:
            if any(s != 1 for s in image_data.shape[3:]):
                raise ValueError(f"Multiple input frames {tuple(image_data.shape)} not supported!")
            image_data = np.squeeze(image_data, axis=tuple(range(3, image_data.ndim)))
        # if the output has the same number of frames as the input
        elif image_data.shape[3:] == out_shape[3:]:
            # add a frame dimension to vox2vox
            _vox2vox = np.eye(5, dtype=vox2vox.dtype)
            _vox2vox[:3, :3] = vox2vox[:3, :3]
            _vox2vox[3:, 4:] = vox2vox[:3, 3:]
            vox2vox = _vox2vox
        else:
            raise ValueError(
                    f"Input image and requested output shape have different frames: {image_data.shape} vs. {out_shape}!"
                )

    delta = np.abs(vox2vox - np.eye(4))
    off_diag = np.ones((4, 4)) - np.eye(4)
    if np.all(np.less(delta, np.eye(4) * max(vox_eps, rot_eps) + off_diag * rot_eps)) and image_data.shape == out_shape:
        # no interpolation needed, just use image_data
        return image_data

    inv_vox2vox = np.linalg.inv(vox2vox)
    if not does_vox2vox_rot_require_interpolation(vox2vox, vox_eps=vox_eps, rot_eps=rot_eps):

        LOGGER.debug(f"vox2vox: {vox2vox}")
        # second condition: translations are integers
        if np.allclose(vox2vox[:, 3], np.round(vox2vox[:, 3]), atol=1e-4):
            # reorder axes
            ornt = nib.orientations.io_orientation(vox2vox)
            new_old_index = list(enumerate(map(int, ornt[:, 0])))
            # if the direction is flipped (ornt[j, 1] == -1), offset has to start at the other end
            offsets = [ornt[j, 1] * vox2vox[i, 3] + (ornt[j, 1] == -1) * img.shape[j] for i, j in new_old_index]
            offsets = list(map(lambda x: int(x.astype(int)), offsets))
            reordered = apply_orientation(image_data, ornt)
            # pad=0 => pad with zeros
            return crop_transform(reordered, offsets=offsets, target_shape=out_shape, pad=0)

    from scipy.ndimage import affine_transform

    return affine_transform(image_data, inv_vox2vox, output_shape=out_shape, order=order)


def getscale(
        data: np.ndarray,
        dst_min: float | int,
        dst_max: float | int,
        f_low: float = 0.0,
        f_high: float = 0.999,
) -> tuple[float, float]:
    """
    Get offset and scale of image intensities to robustly rescale to dst_min..dst_max.

    Equivalent to how mri_convert conforms images.

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
        vox_size: VoxSizeOption | None = 1.0,
        img_size: ImageSizeOption | None = 256,
        dtype: type | None = np.uint8,
        orientation: OrientationType | None = "lia",
        threshold_1mm: float | None = None,
        rescale: int | float | Literal["none"] = 255,
        vox_eps: float = 1e-4,
        rot_eps: float = 1e-6,
        **kwargs,
) -> nibabelImage:
    """Python version of mri_convert -c.

    mri_convert -c by default turns image intensity values into UCHAR, reslices images to standard position, fills up
    slices to standard 256x256x256 format and enforces 1mm or minimum isotropic voxel sizes.

    Parameters
    ----------
    img : nibabelImage
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

    Returns
    -------
    nib.MGHImage
        Conformed image.

    Other Parameters
    ----------------
    conform_vox_size : float, optional
        Legacy parameter for vox_size, overwrites vox_size.
    conform_to_1mm_threshold : float, optional
        Legacy parameter for threshold_1mm, overwrites threshold_1mm.

    Notes
    -----
    Unlike mri_convert -c, we first interpolate (float image), and then rescale to uchar. mri_convert is doing it the
    other way around. However, we compute the scale factor from the input to increase similarity.
    """
    if "conform_to_1mm_threshold" in kwargs:
        LOGGER.warning("conform_to_1mm_threshold is deprecated, replaced by threshold_1mm and will be removed.")
        threshold_1mm = kwargs["conform_to_1mm_threshold"]
    if "conform_vox_size" in kwargs:
        LOGGER.warning("conform_vox_size is deprecated, replaced by vox_size and will be removed.")
        vox_size = kwargs["conform_vox_size"]

    vox_img = conformed_vox_img_size(img, vox_size, img_size, threshold_1mm=threshold_1mm, vox_eps=vox_eps)
    _orientation: OrientationType = "native" if orientation is None else orientation
    h1 = prepare_mgh_header(img, *vox_img, _orientation, vox_eps=vox_eps, rot_eps=rot_eps)

    # affine is the computed target affine for the output image
    target_affine = h1.get_affine()
    if LOGGER.getEffectiveLevel() <= logging.DEBUG:
        with np.printoptions(precision=2, suppress=True):
            from re import sub
            LOGGER.debug("affine: " + sub("\\s+", " ", str(target_affine[:3, :3])))

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # derive target datatype from input
    target_dtype: np.dtype = np.dtype(img.get_data_dtype() if dtype is None else dtype)
    limits: None | tuple[int | float, int | float] = None

    if rescale is None and np.issubdtype(target_dtype, np.integer):
        limits = np.iinfo(target_dtype).min, np.iinfo(target_dtype).max
    elif isinstance(rescale, int | float):
        limits = 0, rescale
    elif rescale is not None:
        raise ValueError(f"Invalid rescale value: {rescale}")

    # reorient the image to the "corrected" (target) affine, always use float here
    mapped_data = map_image(
        img,
        target_affine,
        h1.get_data_shape(),
        order=order,
        dtype=float,
        vox_eps=vox_eps,
        rot_eps=rot_eps,
    )

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
    new_img = nibabel.MGHImage(mapped_data.astype(target_dtype), target_affine, h1)

    # make sure we store uchar
    from nibabel.freesurfer import mghformat
    try:
        new_img.set_data_dtype(target_dtype)
    except mghformat.MGHError as e:
        if "not recognized" not in e.args[0]:
            raise
        dtype_codes = mghformat.data_type_codes.code.keys()
        codes = set(k.name for k in dtype_codes if isinstance(k, np.dtype))
        logging.getLogger(__name__).error(
            f"The data type '{dtype}' is not recognized for MGH images, switching to '{new_img.get_data_dtype()}' "
            f"(supported: {tuple(codes)})."
        )

    return new_img


def prepare_mgh_header(
        img: nibabelImage,
        target_vox_size: npt.NDArray[float] | None = None,
        target_img_size: npt.NDArray[int] | None = None,
        orientation: OrientationType = "native",
        vox_eps: float = 1e-4,
        rot_eps: float = 1e-6,
) -> MGHHeader:
    """
    Prepare the header with affine by target voxel size, target image size and criteria - initialized from img.

    This implicitly prepares the affine, which can be computed by `return_value.get_affine()`.

    Parameters
    ----------
    img : nibabel.analyze.SpatialImage
        The image object to base the header on.
    target_vox_size : npt.NDArray[float], None, default=None
        The target voxel size, importantly still in native orientation (reordering after).
    target_img_size : npt.NDArray[int], None, default=None
        The target image size, importantly still in native orientation (reordering after).
    orientation : "native", "soft-<orientation>", "<orientation>", default="native"
        How the affine should look like.
    vox_eps : float, default=1e-4
        The epsilon for the voxelsize check.
    rot_eps : float, default=1e-6
        The epsilon for the affine rotation check.

    Returns
    -------
    nibabel.freesurfer.mghformat.MGHHeader
        The header object to the "conformed" image based on img and the other parameters.
    """
    # may copy some parameters if input was MGH format
    h1 = MGHHeader.from_header(img.header)
    # nibabel only copies header information, if the file type is the same (here, this would be only of mgh header)
    source_img_shape = img.header.get_data_shape()
    source_vox_size = img.header.get_zooms()

    source_mdc = img.affine[:3, :3] / np.linalg.norm(img.affine[:3, :3], axis=0, keepdims=True)
    # native
    if orientation == "native":
        re_order_axes = [0, 1, 2]
        mdc_affine = np.linalg.inv(source_mdc)
    else:
        _ornt_transform, _ = orientation_to_ornts(img.affine, orientation[-3:])
        re_order_axes = _ornt_transform[:, 0]
        if len(orientation) == 3:  # lia, ras, etc
            # this is a 3x3 matrix
            out_ornt = nib.orientations.axcodes2ornt(orientation[-3:].upper())
            mdc_affine = nib.orientations.inv_ornt_aff(out_ornt, source_img_shape)[:3, :3]
        else: # soft lia, ras, ....
            aff = _ornt_transform[:, 1][None] * source_mdc
            mdc_affine = np.stack([aff[:3, int(ax)] for ax in _ornt_transform[:, 0]], axis=-1)

    shape: list[int] = [(source_img_shape if target_img_size is None else target_img_size)[i] for i in re_order_axes]
    h1.set_data_shape(shape + [1])

    # --> h1['delta']
    h1.set_zooms([(target_vox_size if target_vox_size is not None else source_vox_size)[i] for i in re_order_axes])

    h1["Mdc"] = mdc_affine
    # fov should only be defined, if the image has same fov in all directions? fov == one number
    _fov = np.asarray([i * v for i, v in zip(h1.get_data_shape(), h1.get_zooms(), strict=False)])
    if _fov.min() == _fov.max():
        # fov is not needed for MGHHeader.get_affine()
        h1["fov"] = _fov[0]
    center = np.asarray(img.shape[:3], dtype=float) / 2.0
    h1["Pxyz_c"] = img.affine.dot(np.hstack((center, [1.0])))[:3]
    # There is a special case here, where an interpolation is triggered, but it is not necessary, if the position of
    # the center could "fix this" condition:
    vox2vox = np.linalg.inv(h1.get_affine()) @ img.affine
    if does_vox2vox_rot_require_interpolation(vox2vox, vox_eps=vox_eps, rot_eps=rot_eps):
        # 1. has rotation, or vox-size resampling => requires resampling
        pass
    else:
        # 2. img_size changes from odd to even and vice versa
        #    i.e. can changing the RAS center make an interpolation unnecessary?
        vec = np.linalg.inv(vox2vox)[:3, 3]
        tols = {"atol": 1.e-4, "rtol": 0.}
        # is it fixable?
        if not np.allclose(vec, np.round(vec), **tols) and np.allclose(vec * 2, np.round(vec * 2), **tols):
            new_center = (center + (1 - np.isclose(vec, np.round(vec), **tols)) / 2.0, [1.0])
            h1["Pxyz_c"] = img.affine.dot(np.hstack(new_center))[:3]

    # tr information is not copied when copying from non-mgh formats
    if len(img.header.get('pixdim', [])) :
        h1['tr'] = img.header['pixdim'][4] * 1000

    # The affine can be explicitly constructed by MGHHeader.get_affine() / h1.get_affine()
    # MdcD = np.asarray(h1["Mdc"]).T * h1["delta"]
    # vol_center = MdcD.dot(hdr["dims"][:3]) / 2
    # affine = from_matvec(MdcD, h1["Pxyz_c"] - vol_center)
    return h1


def does_vox2vox_rot_require_interpolation(
        vox2vox: npt.NDArray[float],
        vox_eps: float = 1e-4,
        rot_eps: float = 1e-6,
) -> bool:
    """
    Check whether the affine requires resampling/interpolation or whether reordering is sufficient.

    Parameters
    ----------
    vox2vox : np.ndarray
        The affine matrix.
    vox_eps : float, default=1e-4
        The epsilon for the voxelsize check.
    rot_eps : float, default=1e-6
        The epsilon for the affine rotation check.

    Returns
    -------
    bool
        Whether the vox2vox matrix requires resampling.
    """
    def isclose(x, y, eps):
        return np.isclose(x, y, atol=eps, rtol=0)

    _v2v_pos = np.abs(vox2vox[:3, :3])
    # all values -1, 1 or 0 ==> False (does not require interpolation)
    return not np.all(np.logical_or(isclose(_v2v_pos, 1, eps=vox_eps), isclose(_v2v_pos, 0, eps=rot_eps)))


def is_conform(
        img: nibabelImage,
        vox_size: VoxSizeOption | None = 1.0,
        img_size: ImageSizeOption | None = 256,
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
    img : nibabelImage
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
    affcode = "".join(nib.orientations.aff2axcodes(img.affine))
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
        _dtype_name = np.dtype(_dtype).name if isinstance(_dtype, (str, np.dtype)) else _dtype.__name__
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
        affine: npt.NDArray[float],
        target_orientation: OrientationType = "lia",
        soft: bool = False,
        eps: float = 1e-6,
):
    """
    Checks whether the affine is LIA-oriented.

    Parameters
    ----------
    affine : np.ndarray
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
    if "".join(nib.orientations.aff2axcodes(affine, tol=eps)).lower() == target_orientation.lower():
        if soft:
            return True
    else:
        return False

    return does_vox2vox_rot_require_interpolation(affine / np.linalg.norm(affine, axis=0), eps=eps)


def conformed_vox_img_size(
        img: nibabelImage,
        vox_size: VoxSizeOption | None,
        img_size: ImageSizeOption | None,
        threshold_1mm: float | None = None,
        vox_eps: float = 1e-4,
        **kwargs,
) -> tuple[npt.NDArray[float] | None, npt.NDArray[int] | None]:
    """
    Extract the voxel size and the image size.

    This function only needs the header (not the data).

    Parameters
    ----------
    img : nibabelImage
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
    numpy.typing.NDArray[float], None
        The determined voxel size to conform the image to (still in native orientation), shape: 3.
    numpy.typing.NDArray[int], None
        The size of the image adjusted to the conformed voxel size (still in native orientation), shape: 3.
    """
    if "conform_to_1mm_threshold" in kwargs:
        LOGGER.warning("conform_to_1mm_threshold is deprecated, replaced by threshold_1mm and will be removed.")
        threshold_1mm = kwargs["conform_to_1mm_threshold"]
    if "conform_vox_size" in kwargs:
        LOGGER.warning("conform_vox_size is deprecated, replaced by vox_size and will be removed.")
        vox_size = kwargs["conform_vox_size"]

    MAX_VOX_SIZE = 1.0
    MAX_DIMENSION = 256
    # this is similar to mri_convert --conform_min
    if isinstance(vox_size, str) and (vox_size := vox_size.lower()) in ["min", "auto"]:
        # find minimal voxel side length
        min_vox_size = np.round(np.min(img.header.get_zooms()[:3]), decimals=int(np.ceil(-np.log10(vox_eps))))
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
        raise ValueError("Invalid value for vox_size passed.")
    if img_size is None and target_vox_size is not None:
        # if we did specify a vox_size, no image size. use the field of view (which is essentially the old image size
        # scaled with the voxel size)
        img_size = "fov"
    if img_size is None:
        target_img_size = None
    elif isinstance(img_size, int) and img_size > 0:
        target_img_size = np.full((3,), img_size)
    elif isinstance(img_size, str) and (img_size := img_size.lower()) in ["fov", "auto"]:
        thres = abs(1.0 - (threshold_1mm or 1.0))
        if target_vox_size is not None and np.allclose(target_vox_size, 1.0, atol=thres) and img_size == "auto":
            target_img_size = np.full((3,), MAX_DIMENSION)
        # (other voxel sizes may use different sizes)
        else:
            target_img_size = np.array(img.shape[:3])
            if target_vox_size is not None:
                # correct sizes for changing voxel size (if voxel size is changing)
                # compute field of view dimensions in mm (in native orientation)
                fov = np.array(img.header.get_zooms()[:3]) * target_img_size
                # compute number of voxels needed to cover field of view
                target_img_size = np.ceil((fov / target_vox_size * 10000).astype(int).astype(float) / 10000).astype(int)
        # use cube (same size in all directions) with MAX_DIMENSION in each direction as minimum
        if img_size == "auto":
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

    header = cast(nib.Nifti1Header | nib.Nifti2Header, img.header)
    if header["qform_code"] != 0 and not np.allclose(img.get_sform(), img.get_qform(), atol=0.001):
        message = (
            f"#############################################################\n"
            f"WARNING: qform and sform transform are not identical!\n"
            f" sform-transform:\n{header.get_sform()}\n"
            f" qform-transform:\n{header.get_qform()}\n"
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
        vox_size_header = header.get_zooms()

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


def _crop_transform_make_indices(image_shape, offsets, target_shape):
    """
    Create the indexing tuple and return padding tuples for the last N dimensions.

    Parameters
    ----------
    image_shape : np.ndarray
        The shape of the image from which a region is to be cropped.
    offsets : Sequence[int]
        Exact location within the image from which the cropping should start.
    target_shape : Sequence[int], optional
        The desired shape of the cropped region.

    Returns
    -------
    paddings: list of 2-tuples of paddings or None
        A list of per-axis tuples of the padding to apply to the slice to get the target_shape.
    indices : tuple of indices
        A tuple of per-axis indices to index in the data to get the target_shape.
    """
    if len(offsets) != len(target_shape):
        raise ValueError(
            f"offsets {offsets} and target shape {target_shape} must be same length."
        )
    if len(offsets) > len(image_shape):
        raise ValueError("offsets too long for image")
    batch_dims = len(image_shape) - len(offsets)
    indices = [slice(None)] * batch_dims
    paddings = []
    any_pad = False
    for offset, t_shape, i_shape in zip(
        offsets, target_shape, image_shape[batch_dims:], strict=False
    ):
        crop_end = min(offset + t_shape, i_shape)
        indices.append(slice(max(0, offset), crop_end))
        pads = (max(0, -offset), max(0, offset + t_shape - crop_end))
        paddings.append(pads)
        any_pad = any_pad or any(p != 0 for p in pads)

    return paddings if any_pad else None, tuple(indices)


def _crop_transform_pad_fn(image, pad_tuples, pad):
    """
    Generate a parameterized pad function.

    Parameters
    ----------
    image : np.ndarray, torch.Tensor
        Input image.
    pad_tuples : List[Tuple[int, int]]
        List of padding tuples for each axis.

    Returns
    -------
    partial
        A partial function to pad the image.
    """
    if all(p1 == 0 and p2 == 0 for p1, p2 in pad_tuples):
        return None

    kwargs = {"mode": "constant"}
    if isinstance(pad, str):
        kwargs["mode"] = pad
    elif isinstance(image, np.ndarray):
        kwargs["constant_values"] = pad
    else:  # Tensor
        kwargs["value"] = pad

    from functools import partial

    if isinstance(image, np.ndarray):
        return partial(np.pad, pad_width=[(0, 0)] * (image.ndim - len(pad_tuples)) + pad_tuples, **kwargs)
    else:  # Tensor
        from itertools import chain

        from torch.nn.functional import pad as _pad

        return partial(_pad, pad=list(chain.from_iterable(reversed(pad_tuples))), **kwargs)


def crop_transform(
        image: _TA,
        offsets: Sequence[int] | None = None,
        target_shape: Sequence[int] | None = None,
        out: _TA | None = None,
        pad: int = 0,
) -> _TA:
    """
    Perform a crop transform of the last N dimensions on the image data.
    Cropping does not interpolate the image, but "just removes" border pixels/voxels.
    Negative offsets lead to padding.

    Parameters
    ----------
    image : np.ndarray, torch.Tensor
        Image of size [..., D_1, D_2, ..., D_N], where D_1, D_2, ..., D_N are the N
        image dimensions.
    offsets : Sequence[int], optional
        Offset of the cropped region for the last N dimensions (default: center crop
        with less crop/pad towards index 0).
    target_shape : Sequence[int], optional
        If defined, target_shape specifies the target shape of the "cropped region",
        else the crop will be centered cropping offset[dim] voxels on each side (then
        the shape is derived by subtracting 2x the dimension-specific offset).
        target_shape should have the same number of elements as offsets.
        May be implicitly defined by out.
    out : np.ndarray, torch.Tensor, optional
        Array to store the cropped image in (optional), can be a view on image for
        memory-efficiency.
    pad :  int, str, default=0/zero-pad
        Padding strategy to use when padding is required, if int, pad with that value.

    Returns
    -------
    out : np.ndarray, torch.Tensor
        The image (stack) cropped in the last N dimensions by offsets to the shape
        target_shape, or if target_shape is not given image.shape[i+2] - 2*offset[i].

    Raises
    ------
    ValueError
        If neither offsets nor target_shape nor out are defined.
    ValueError
        If out is not target_shape.
    TypeError
        If the type of image is not an np.ndarray or a torch.Tensor.
    RuntimeError
        If the dimensionality of image, out, offset or target_shape is invalid or
        inconsistent.

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
        _target_shape = image.shape[: -len(target_shape)] + tuple(target_shape)
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
    if out is None:
        if pad_tuples is None:
            return image[indices]
        else:
            pad_fn = _crop_transform_pad_fn(image, pad_tuples, pad)
            return pad_fn(image[indices])
    else:
        if pad_tuples is None:
            out[:] = image[indices]
        else:
            pad_fn = _crop_transform_pad_fn(image, pad_tuples, pad)
            out[:] = pad_fn(image[indices])

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
    image = nib.load(options.input)

    if not isinstance(image, nib.analyze.SpatialImage):
        sys.exit(f"ERROR: Input image is not a spatial image: {type(image).__name__}")
    if len(image.shape) > 3 and image.shape[3] != 1:
        sys.exit(f"ERROR: Multiple input frames ({image.shape[3]}) not supported!")

    opt_kwargs = {
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
        new_image = conform(image, order=options.order, rescale=options.rescale, file_type=file_type, **opt_kwargs)
    except ValueError as e:
        sys.exit(e.args[0])
    LOGGER.info(f"Writing conformed image: {options.output}")

    nib.save(new_image, options.output)

    sys.exit(0)
