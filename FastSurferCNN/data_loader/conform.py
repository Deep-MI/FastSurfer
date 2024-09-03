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
import logging
import sys
from collections.abc import Iterable
from enum import Enum
from typing import cast

import nibabel as nib
import numpy as np
import numpy.typing as npt

from FastSurferCNN.utils.arg_types import (
    VoxSizeOption,
)
from FastSurferCNN.utils.arg_types import (
    float_gt_zero_and_le_one as __conform_to_one_mm,
)
from FastSurferCNN.utils.arg_types import (
    target_dtype as __target_dtype,
)
from FastSurferCNN.utils.arg_types import (
    vox_size as __vox_size,
)

HELPTEXT = """
Script to conform an MRI brain image to UCHAR, RAS orientation, 
and 1mm or minimal isotropic voxels

USAGE:
conform.py  -i <input> -o <output> <options>
OR
conform.py  -i <input> --check_only <options>
Dependencies:
    Python 3.8+
    Numpy
    https://www.numpy.org
    Nibabel to read and write FreeSurfer data
    https://nipy.org/nibabel/
Original Author: Martin Reuter
Date: Jul-09-2019
"""

h_input = "path to input image"
h_output = "path to output image"
h_order = "order of interpolation (0=nearest, 1=linear(default), 2=quadratic, 3=cubic)"

LIA_AFFINE = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])


class Criteria(Enum):
    FORCE_LIA_STRICT = "lia strict"
    FORCE_LIA = "lia"
    FORCE_IMG_SIZE = "img size"
    FORCE_ISO_VOX = "iso vox"


DEFAULT_CRITERIA_DICT = {
    "lia": Criteria.FORCE_LIA,
    "strict_lia": Criteria.FORCE_LIA_STRICT,
    "iso_vox": Criteria.FORCE_ISO_VOX,
    "img_size": Criteria.FORCE_IMG_SIZE,
}
DEFAULT_CRITERIA = frozenset(DEFAULT_CRITERIA_DICT.values())


def options_parse():
    """
    Command line option parser.

    Returns
    -------
    options
        Object holding options.
    """
    parser = argparse.ArgumentParser(usage=HELPTEXT)
    parser.add_argument(
        "--version",
        action="version",
        version="$Id: conform.py,v 1.0 2019/07/19 10:52:08 mreuter Exp $",
    )
    parser.add_argument("--input", "-i", dest="input", help=h_input)
    parser.add_argument("--output", "-o", dest="output", help=h_output)
    parser.add_argument("--order", dest="order", help=h_order, type=int, default=1)
    parser.add_argument(
        "--check_only",
        dest="check_only",
        default=False,
        action="store_true",
        help="If True, only checks if the input image is conformed, and does not "
             "return an output.",
    )
    parser.add_argument(
        "--seg_input",
        dest="seg_input",
        default=False,
        action="store_true",
        help="Specifies whether the input is a seg image. If true, the check for "
             "conformance disregards the uint8 dtype criteria. Use --dtype any for "
             "equivalent results. --seg_input overwrites --dtype arguments.",
    )
    parser.add_argument(
        "--vox_size",
        dest="vox_size",
        default=1.0,
        type=__vox_size,
        help="Specifies the target voxel size to conform to. Also allows 'min' for "
             "conforming to the minimum voxel size, otherwise similar to mri_convert's "
             "--conform_size <size> (default: 1, conform to 1mm).",
    )
    parser.add_argument(
        "--conform_min",
        dest="conform_min",
        default=False,
        action="store_true",
        help="Specifies whether the input is or should be conformed to the "
        "minimal voxel size (used for high-res processing) - overwrites --vox_size.",
    )
    advanced = parser.add_argument_group("Advanced options")
    advanced.add_argument(
        "--conform_to_1mm_threshold",
        type=__conform_to_one_mm,
        help="Advanced option to change the threshold beyond which images are "
             "conformed to 1 (default: infinity, all images are conformed to their "
             "minimum voxel size).",
    )
    advanced.add_argument(
        "--dtype",
        dest="dtype",
        default="uint8",
        type=__target_dtype,
        help="Specifies the target data type of the target image or 'any' (default: "
             "'uint8', as in FreeSurfer)",
    )
    advanced.add_argument(
        "--no_strict_lia",
        dest="force_strict_lia",
        action="store_false",
        help="Ignore the forced LIA reorientation.",
    )
    advanced.add_argument(
        "--no_lia",
        dest="force_lia",
        action="store_false",
        help="Ignore the reordering of data into LIA (without interpolation). "
             "Superceeds --no_strict_lia",
    )
    advanced.add_argument(
        "--no_iso_vox",
        dest="force_iso_vox",
        action="store_false",
        help="Ignore the forced isometric voxel size (depends on --conform_min).",
    )
    advanced.add_argument(
        "--no_img_size",
        dest="force_img_size",
        action="store_false",
        help="Ignore the forced image dimensions (depends on --conform_min).",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="If verbose, more specific messages are printed",
    )
    args = parser.parse_args()
    if args.input is None:
        raise RuntimeError("ERROR: Please specify input image")
    if not args.check_only and args.output is None:
        raise RuntimeError("ERROR: Please specify output image")
    if args.check_only and args.output is not None:
        raise RuntimeError(
            "ERROR: You passed in check_only. Please do not also specify output image"
        )
    if args.seg_input and args.dtype not in ["uint8", "any"]:
        print("WARNING: --seg_input overwrites the --dtype arguments.")
    if not args.force_lia and args.force_strict_lia:
        print("INFO: --no_lia includes --no_strict_lia.")
        args.force_strict_lia = False
    return args


def map_image(
        img: nib.analyze.SpatialImage,
        out_affine: np.ndarray,
        out_shape: tuple[int, ...] | np.ndarray | Iterable[int],
        ras2ras: np.ndarray | None = None,
        order: int = 1,
        dtype: type | None = None
) -> np.ndarray:
    """
    Map image to new voxel space (RAS orientation).

    Parameters
    ----------
    img : nib.analyze.SpatialImage
        The src 3D image with data and affine set.
    out_affine : np.ndarray
        Trg image affine.
    out_shape : tuple[int, ...], np.ndarray
        The trg shape information.
    ras2ras : np.ndarray, optional
        An additional mapping that should be applied (default=id to just reslice).
    order : int, default=1
        Order of interpolation (0=nearest,1=linear,2=quadratic,3=cubic).
    dtype : Type, optional
        Target dtype of the resulting image (relevant for reorientation,
        default=keep dtype of img).

    Returns
    -------
    np.ndarray
        Mapped image data array.
    """
    from numpy.linalg import inv
    from scipy.ndimage import affine_transform

    if ras2ras is None:
        ras2ras = np.eye(4)

    # compute vox2vox from src to trg
    vox2vox = inv(out_affine) @ ras2ras @ img.affine

    # here we apply the inverse vox2vox (to pull back the src info to the target image)
    image_data = np.asanyarray(img.dataobj)
    # convert frames to single image

    out_shape = tuple(out_shape)
    # if input has frames
    if image_data.ndim > 3:
        # if the output has no frames
        if len(out_shape) == 3:
            if any(s != 1 for s in image_data.shape[3:]):
                raise ValueError(
                    f"Multiple input frames {tuple(image_data.shape)} not supported!"
                )
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
                    f"Input image and requested output shape have different frames:"
                    f"{image_data.shape} vs. {out_shape}!"
                )

    if dtype is not None:
        image_data = image_data.astype(dtype)

    if not is_resampling_vox2vox(vox2vox):
        # this is a shortcut to reordering resampling
        order = 0

    return affine_transform(
        image_data, inv(vox2vox), output_shape=out_shape, order=order,
    )


def getscale(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        f_low: float = 0.0,
        f_high: float = 0.999
) -> tuple[float, float]:
    """
    Get offset and scale of image intensities to robustly rescale to dst_min..dst_max.

    Equivalent to how mri_convert conforms images.

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
    float src_min
        (adjusted) offset.
    float
        Scale factor.
    """

    if f_low < 0. or f_high > 1. or f_low > f_high:
        raise ValueError(
            "Invalid values for f_low or f_high, must be within 0 and 1."
        )

    # get min and max from source
    data_min = np.min(data)
    data_max = np.max(data)

    if data_min < 0.0:
        # logger. warning
        print("WARNING: Input image has value(s) below 0.0 !")
    # logger.info
    print(f"Input:    min: {data_min}  max: {data_max}")

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
        raise RuntimeError(
            f"ERROR: rescale upper bound not found: f_high={f_high}"
        )

    src_max: float = bin_edges[upper_binedge_index].item()

    # scale
    if src_min == src_max:
        # logger.warning
        print("WARNING: Scaling between src_min and src_max. The input image "
              "is likely corrupted!")
        scale = 1.0
    else:
        scale = (dst_max - dst_min) / (src_max - src_min)
    # logger.info
    print(f"rescale:  min: {src_min}  max: {src_max}  scale: {scale}")

    return src_min, scale


def scalecrop(
        data: np.ndarray,
        dst_min: float,
        dst_max: float,
        src_min: float,
        scale: float
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
    print(
        "Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max())
    )

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


def find_min_size(img: nib.analyze.SpatialImage, max_size: float = 1) -> float:
    """
    Find minimal voxel size <= 1mm.

    Parameters
    ----------
    img : nib.analyze.SpatialImage
        Loaded source image.
    max_size : float
        Maximal voxel size in mm (default: 1.0).

    Returns
    -------
    float
        Rounded minimal voxel size.

    Notes
    -----
    This function only needs the header (not the data).
    """
    # find minimal voxel side length
    sizes = np.array(img.header.get_zooms()[:3])
    min_vox_size = np.round(np.min(sizes) * 10000) / 10000
    # set to max_size mm if larger than that (usually 1mm)
    return min(min_vox_size, max_size)


def find_img_size_by_fov(
        img: nib.analyze.SpatialImage,
        vox_size: float,
        min_dim: int = 256
) -> int:
    """
    Find the cube dimension (>= 256) to cover the field of view of img.

    If vox_size is one, the img_size MUST always be min_dim (the FreeSurfer standard).

    Parameters
    ----------
    img : nib.analyze.SpatialImage
        Loaded source image.
    vox_size : float
        The target voxel size in mm.
    min_dim : int
        Minimal image dimension in voxels (default 256).

    Returns
    -------
    int
        The number of voxels needed to cover field of view.

    Notes
    -----
    This function only needs the header (not the data).
    """
    if vox_size == 1.0:
        return min_dim
    # else (other voxel sizes may use different sizes)

    # compute field of view dimensions in mm
    sizes = np.array(img.header.get_zooms()[:3])
    max_fov = np.max(sizes * np.array(img.shape[:3]))
    # compute number of voxels needed to cover field of view
    conform_dim = int(np.ceil(int(max_fov / vox_size * 10000) / 10000))
    # use cube with min_dim (default 256) in each direction as minimum
    return max(min_dim, conform_dim)


def conform(
        img: nib.analyze.SpatialImage,
        order: int = 1,
        conform_vox_size: VoxSizeOption = 1.0,
        dtype: type | None = None,
        conform_to_1mm_threshold: float | None = None,
        criteria: set[Criteria] = DEFAULT_CRITERIA,
) -> nib.MGHImage:
    """Python version of mri_convert -c.

    mri_convert -c by default turns image intensity values
    into UCHAR, reslices images to standard position, fills up slices to standard
    256x256x256 format and enforces 1mm or minimum isotropic voxel sizes.

    Parameters
    ----------
    img : nib.analyze.SpatialImage
        Loaded source image.
    order : int
        Interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic).
    conform_vox_size : VoxSizeOption
        Conform image the image to voxel size 1. (default), a
        specific smaller voxel size (0-1, for high-res), or automatically
        determine the 'minimum voxel size' from the image (value 'min').
        This assumes the smallest of the three voxel sizes.
    dtype : Optional[Type]
        The dtype to enforce in the image (default: UCHAR, as mri_convert -c).
    conform_to_1mm_threshold : Optional[float]
        The threshold above which the image is conformed to 1mm
        (default: ignore).
    criteria : set[Criteria], default in DEFAULT_CRITERIA
        Whether to force the conforming to include a LIA data layout, an image size
        requirement and/or a voxel size requirement.

    Returns
    -------
    nib.MGHImage
        Conformed image.

    Notes
    -----
    Unlike mri_convert -c, we first interpolate (float image), and then rescale
    to uchar. mri_convert is doing it the other way around. However, we compute
    the scale factor from the input to increase similarity.
    """
    conformed_vox_size, conformed_img_size = get_conformed_vox_img_size(
        img, conform_vox_size, conform_to_1mm_threshold=conform_to_1mm_threshold,
    )
    from nibabel.freesurfer.mghformat import MGHHeader

    # may copy some parameters if input was MGH format
    h1 = MGHHeader.from_header(img.header)
    mdc_affine = h1["Mdc"]
    img_shape = img.header.get_data_shape()
    vox_size = img.header.get_zooms()
    do_interp = False
    affine = img.affine[:3, :3]
    if {Criteria.FORCE_LIA, Criteria.FORCE_LIA_STRICT} & criteria != {}:
        do_interp = bool(Criteria.FORCE_LIA_STRICT in criteria and is_lia(affine, True))
        re_order_axes = [np.abs(affine[:, j]).argmax() for j in (0, 2, 1)]
    else:
        re_order_axes = [0, 1, 2]

    if Criteria.FORCE_IMG_SIZE in criteria:
        h1.set_data_shape([conformed_img_size] * 3 + [1])
    else:
        h1.set_data_shape([img_shape[i] for i in re_order_axes] + [1])
    if Criteria.FORCE_ISO_VOX in criteria:
        h1.set_zooms([conformed_vox_size] * 3)  # --> h1['delta']
        do_interp |= not np.allclose(vox_size, conformed_vox_size)
    else:
        h1.set_zooms([vox_size[i] for i in re_order_axes])

    if Criteria.FORCE_LIA_STRICT in criteria:
        mdc_affine = LIA_AFFINE
    elif Criteria.FORCE_LIA in criteria:
        mdc_affine = affine[:3, re_order_axes]
        if mdc_affine[0, 0] > 0:  # make 0,0 negative
            mdc_affine[:, 0] = -mdc_affine[:, 0]
        if mdc_affine[1, 2] < 0:  # make 1,2 positive
            mdc_affine[:, 2] = -mdc_affine[:, 2]
        if mdc_affine[2, 1] > 0:  # make 2,1 negative
            mdc_affine[:, 1] = -mdc_affine[:, 1]
    else:
        mdc_affine = img.affine[:3, :3]

    mdc_affine = mdc_affine / np.linalg.norm(mdc_affine, axis=1)
    h1["Mdc"] = np.linalg.inv(mdc_affine)

    print(h1.get_zooms())
    h1["fov"] = max(i * v for i, v in zip(h1.get_data_shape(), h1.get_zooms(), strict=False))
    center = np.asarray(img.shape[:3], dtype=float) / 2.0
    h1["Pxyz_c"] = img.affine.dot(np.hstack((center, [1.0])))[:3]

    # There is a special case here, where an interpolation is triggered, but it is not
    # necessary, if the position of the center could "fix this"
    # condition: 1. no rotation, no vox-size resampling,
    if not is_resampling_vox2vox(np.linalg.inv(h1.get_affine()) @ img.affine):
        # 2. img_size changes from odd to even and vice versa
        ishape = np.asarray(img.shape)[re_order_axes]
        delta_shape = np.subtract(ishape, h1.get_data_shape()[:3])
        # 2. img_size changes from odd to even and vice versa
        if not np.allclose(np.remainder(delta_shape, 2), 0):
            # invert axis reordering
            delta_shape[re_order_axes] = delta_shape
            new_center = (center + delta_shape / 2.0, [1.0])
            h1["Pxyz_c"] = img.affine.dot(np.hstack(new_center))[:3]

    # Here, we are explicitly using MGHHeader.get_affine() to construct the affine as
    # MdcD = np.asarray(h1["Mdc"]).T * h1["delta"]
    # vol_center = MdcD.dot(hdr["dims"][:3]) / 2
    # affine = from_matvec(MdcD, h1["Pxyz_c"] - vol_center)
    affine = h1.get_affine()

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # target scalar type and dtype
    #sctype = np.uint8 if dtype is None else np.obj2sctype(dtype, default=np.uint8)
    sctype = np.uint8 if dtype is None else np.dtype(dtype).type
    target_dtype = np.dtype(sctype)

    src_min, scale = 0, 1.0
    # get scale for conversion on original input before mapping to be more similar to
    # mri_convert
    img_dtype = img.get_data_dtype()
    if any(img_dtype != dtyp for dtyp in (np.dtype(np.uint8), target_dtype)):
        src_min, scale = getscale(np.asanyarray(img.dataobj), 0, 255)

    kwargs = {}
    if sctype != np.uint:
        kwargs["dtype"] = "float"
    mapped_data = map_image(img, affine, h1.get_data_shape(), order=order, **kwargs)

    if img_dtype != np.dtype(np.uint8) or (img_dtype != target_dtype and scale != 1.0):
        scaled_data = scalecrop(mapped_data, 0, 255, src_min, scale)
        # map zero in input to zero in output (usually background)
        scaled_data[mapped_data == 0] = 0
        mapped_data = scaled_data

    if target_dtype == np.dtype(np.uint8):
        mapped_data = np.clip(np.rint(mapped_data), 0, 255)
    new_img = nib.MGHImage(sctype(mapped_data), affine, h1)

    # make sure we store uchar
    from nibabel.freesurfer import mghformat
    try:
        new_img.set_data_dtype(target_dtype)
    except mghformat.MGHError as e:
        if "not recognized" not in e.args[0]:
            raise
        dtype_codes = mghformat.data_type_codes.code.keys()
        codes = set(k.name for k in dtype_codes if isinstance(k, np.dtype))
        print(
            f"The data type '{options.dtype}' is not recognized for MGH images, "
            f"switching to '{new_img.get_data_dtype()}' (supported: {tuple(codes)})."
        )

    return new_img


def is_resampling_vox2vox(
        vox2vox: npt.NDArray[float],
        eps: float = 1e-6,
) -> bool:
    """
    Check whether the affine is resampling or just reordering.

    Parameters
    ----------
    vox2vox : np.ndarray
        The affine matrix.
    eps : float, default=1e-6
        The epsilon for the affine check.

    Returns
    -------
    bool
        The result.
    """
    _v2v = np.abs(vox2vox[:3, :3])
    # check 1: have exactly 3 times 1/-1 rest 0, check 2: all 1/-1 or 0
    return abs(_v2v.sum() - 3) > eps or np.any(np.maximum(_v2v, abs(_v2v - 1)) > eps)


def is_conform(
        img: nib.analyze.SpatialImage,
        conform_vox_size: VoxSizeOption = 1.0,
        eps: float = 1e-06,
        check_dtype: bool = True,
        dtype: type | None = None,
        verbose: bool = True,
        conform_to_1mm_threshold: float | None = None,
        criteria: set[Criteria] = DEFAULT_CRITERIA,
) -> bool:
    """
    Check if an image is already conformed or not.

    Dimensions: 256x256x256, Voxel size: 1x1x1, LIA orientation, and data type UCHAR.

    Parameters
    ----------
    img : nib.analyze.SpatialImage
        Loaded source image.
    conform_vox_size : VoxSizeOption, default=1.0
        Which voxel size to conform to. Can either be a float between 0.0 and
        1.0 or 'min' check, whether the image is conformed to the minimal voxels size, 
        i.e. conforming to smaller, but isotropic voxel sizes for high-res.
    eps : float, default=1e-06
        Allowed deviation from zero for LIA orientation check.
        Small inaccuracies can occur through the inversion operation. Already conformed
        images are thus sometimes not correctly recognized. The epsilon accounts for
        these small shifts.
    check_dtype : bool, default=True
        Specifies whether the UCHAR dtype condition is checked for;
        this is not done when the input is a segmentation.
    dtype : Type, optional
        Specifies the intended target dtype (default or None: uint8 = UCHAR).
    verbose : bool, default=True
        If True, details of which conformance conditions are violated (if any)
        are displayed.
    conform_to_1mm_threshold : float, optional
        Above this threshold the image is conformed to 1mm (default or None: ignore).
    criteria : set[Criteria], default in DEFAULT_CRITERIA
        An enum/set of criteria to check.

    Returns
    -------
    bool:
        Whether the image is already conformed.

    Notes
    -----
    This function only needs the header (not the data).
    """
    conformed_vox_size, conformed_img_size = get_conformed_vox_img_size(
        img, conform_vox_size, conform_to_1mm_threshold=conform_to_1mm_threshold
    )

    ishape = img.shape
    # check 3d
    if len(ishape) > 3 and ishape[3] != 1:
        raise ValueError(f"ERROR: Multiple input frames ({ishape[3]}) not supported!")

    checks = {
        "Number of Dimensions 3": (len(ishape) == 3, f"image ndim {img.ndim}")
    }
    # check dimensions
    if Criteria.FORCE_IMG_SIZE in criteria:
        img_size_criteria = f"Dimensions {'x'.join([str(conformed_img_size)] * 3)}"
        is_correct_img_size = all(s == conformed_img_size for s in ishape[:3])
        checks[img_size_criteria] = is_correct_img_size, f"image dimensions {ishape}"

    # check voxel size, drop voxel sizes of dimension 4 if available
    izoom = np.array(img.header.get_zooms())
    is_correct_vox_size = np.max(np.abs(izoom[:3] - conformed_vox_size)) < eps
    _vox_sizes = conformed_vox_size if is_correct_vox_size else izoom[:3]
    if Criteria.FORCE_ISO_VOX in criteria:
        vox_size_criteria = f"Voxel Size {'x'.join([str(conformed_vox_size)] * 3)}"
        image_vox_size = "image " + "x".join(map(str, izoom))
        checks[vox_size_criteria] = (is_correct_vox_size, image_vox_size)

    # check orientation LIA
    if {Criteria.FORCE_LIA, Criteria.FORCE_LIA_STRICT} & criteria != {}:
        is_strict = Criteria.FORCE_LIA_STRICT in criteria
        lia_text = "strict" if is_strict else "lia"
        if not (is_correct_lia := is_lia(img.affine, is_strict, eps)):
            import re
            print_options = np.get_printoptions()
            np.set_printoptions(precision=2)
            lia_text += ": " + re.sub("\\s+", " ", str(img.affine[:3, :3]))
            np.set_printoptions(**print_options)
        checks["Orientation LIA"] = (is_correct_lia, lia_text)

    # check dtype uchar
    if check_dtype:
        if dtype is None or (isinstance(dtype, str) and dtype.lower() == "uchar"):
            dtype = "uint8"
        else:  # assume obj
            #dtype = np.dtype(np.obj2sctype(dtype)).name
            dtype = np.dtype(dtype).type.__name__
        is_correct_dtype = img.get_data_dtype() == dtype
        checks[f"Dtype {dtype}"] = (is_correct_dtype, f"dtype {img.get_data_dtype()}")

    _is_conform = all(map(lambda x: x[0], checks.values()))

    if verbose:
        if not _is_conform:
            print("The input image is not conformed.")

        conform_str = (
            "conformed" if conform_vox_size == 1.0 else f"{conform_vox_size}-conformed"
        )
        print(f"A {conform_str} image must satisfy the following criteria:")
        for condition, (value, message) in checks.items():
            print(f" - {condition:<30}: {value if value else 'BUT ' + message}")
    return _is_conform


def is_lia(
        affine: npt.NDArray[float],
        strict: bool = True,
        eps: float = 1e-6,
):
    """
    Checks whether the affine is LIA-oriented.

    Parameters
    ----------
    affine : np.ndarray
        The affine to check.
    strict : bool, default=True
        Whether the orientation should be "exactly" LIA or just similar to LIA (i.e.
        it is more LIA than other directions).
    eps : float, default=1e-6
        The threshold in strict mode.

    Returns
    -------
    bool
        Whether the affine is LIA-oriented.
    """
    iaffine = affine[:3, :3]
    lia_nonzero = LIA_AFFINE != 0
    signs = np.all(np.sign(iaffine[lia_nonzero]) == LIA_AFFINE[lia_nonzero])
    if strict:
        directions = np.all(iaffine[np.logical_not(lia_nonzero)] <= eps)
    else:
        def get_primary_dirs(a): return np.argmax(abs(a), axis=0)

        directions = np.all(get_primary_dirs(iaffine) == get_primary_dirs(LIA_AFFINE))
    is_correct_lia = directions and signs
    return is_correct_lia


def get_conformed_vox_img_size(
        img: nib.analyze.SpatialImage,
        conform_vox_size: VoxSizeOption,
        conform_to_1mm_threshold: float | None = None
) -> tuple[float, int]:
    """
    Extract the voxel size and the image size.

    This function only needs the header (not the data).

    Parameters
    ----------
    img : nib.analyze.SpatialImage
        Loaded source image.
    conform_vox_size : float, "min"
        The voxel size parameter to use: either a voxel size as float, or the string
        "min" to automatically find a suitable voxel size (smallest per-dimension voxel
        size).
    conform_to_1mm_threshold : float, optional
        The threshold for which image voxel size should be conformed to 1mm instead of
        conformed to the smallest voxel size (default: None, never apply).

    Returns
    -------
    conformed_vox_size : float
        The determined voxel size to conform the image to.
    conformed_img_size : int
        The size of the image adjusted to the conformed voxel size.
    """
    # this is similar to mri_convert --conform_min
    auto_values = ["min", "auto"]
    if isinstance(conform_vox_size, str) and conform_vox_size.lower() in auto_values:
        conformed_vox_size = find_min_size(img)
        if conform_to_1mm_threshold and conformed_vox_size > conform_to_1mm_threshold:
            conformed_vox_size = 1.0
    # this is similar to mri_convert --conform_size <float>
    elif isinstance(conform_vox_size, float) and 0.0 < conform_vox_size <= 1.0:
        conformed_vox_size = conform_vox_size
    else:
        raise ValueError("Invalid value for conform_vox_size passed.")
    conformed_img_size = find_img_size_by_fov(img, conformed_vox_size)
    return conformed_vox_size, conformed_img_size


def check_affine_in_nifti(
        img: nib.Nifti1Image | nib.Nifti2Image,
        logger: logging.Logger | None = None
) -> bool:
    """
    Check the affine in nifti Image.

    Sets affine with qform, if it exists and differs from sform.
    If qform does not exist, voxel sizes between header information and information
    in affine are compared.
    In case these do not match, the function returns False (otherwise True).

    Parameters
    ----------
    img : Union[nib.Nifti1Image, nib.Nifti2Image]
        Loaded nifti-image.
    logger : Optional[logging.Logger]
        Logger object or None (default) to log or print an info message to
        stdout (for None).

    Returns
    -------
    bool
        False, if voxel sizes in affine and header differ.
    """
    check = True
    message = ""

    header = cast(nib.Nifti1Header | nib.Nifti2Header, img.header)
    if (
        header["qform_code"] != 0 and
        not np.allclose(img.get_sform(), img.get_qform(), atol=0.001)
    ):
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
        print(message)

    return check


if __name__ == "__main__":
    # Command Line options are error checking done here
    try:
        options = options_parse()
    except RuntimeError as e:
        sys.exit(*e.args)

    print(f"Reading input: {options.input} ...")
    image = nib.load(options.input)

    if not isinstance(image, nib.analyze.SpatialImage):
        sys.exit(f"ERROR: Input image is not a spatial image: {type(image).__name__}")
    if len(image.shape) > 3 and image.shape[3] != 1:
        sys.exit(f"ERROR: Multiple input frames ({image.shape[3]}) not supported!")

    _target_dtype = "uint8" if options.seg_input else options.dtype
    crit = DEFAULT_CRITERIA_DICT.items()
    opt_kwargs = {
        "criteria": set(c for n, c in crit if getattr(options, "force_" + n, True)),
    }
    if check_dtype := _target_dtype != "any":
        opt_kwargs["dtype"] = _target_dtype

    if hasattr(options, "conform_to_1mm_threshold"):
        opt_kwargs["conform_to_1mm_threshold"] = options.conform_to_1mm_threshold

    _vox_size = "min" if options.conform_min else options.vox_size
    try:
        image_is_conformed = is_conform(
            image,
            conform_vox_size=_vox_size,
            check_dtype=check_dtype,
            verbose=options.verbose,
            **opt_kwargs,
        )
    except ValueError as e:
        sys.exit(e.args[0])

    if image_is_conformed:
        print(f"Input {options.input} is already conformed! Exiting.\n")
        sys.exit(0)
    else:
        # Note: if check_only, a non-conforming image leads to an error code, this
        # result is needed in recon_surf.sh
        if options.check_only:
            print("check_only flag provided. Exiting without conforming input image.\n")
            sys.exit(1)

    # If image is nifti image
    if options.input[-7:] == ".nii.gz" or options.input[-4:] == ".nii":
        from nibabel import Nifti1Image, Nifti2Image
        if not check_affine_in_nifti(cast(Nifti1Image | Nifti2Image, image)):
            sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

    try:
        new_image = conform(
            image,
            order=options.order,
            conform_vox_size=_vox_size,
            **opt_kwargs,
        )
    except ValueError as e:
        sys.exit(e.args[0])
    print(f"Writing conformed image: {options.output}")

    nib.save(new_image, options.output)

    sys.exit(0)
