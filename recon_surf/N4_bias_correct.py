#!/usr/bin/env python3
# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
# Group 1: Python native modules
import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypeVar, cast

# Group 2: Internal modules
import image_io as iio

# Group 3: External modules
import numpy as np
import SimpleITK as sitk
from numpy import typing as npt

HELPTEXT = """

Script to call SITK N4 Bias Correction

USAGE:
N4_bias_correct  --in <img.nii> [--out <corrected.nii>] [--rescaled <rescaled.nii>] OPTIONS


Dependencies:
    Python 3.8+

    SimpleITK https://simpleitk.org/ (v2.1.1)


N4 Bias Correction (performed if either --out <file> or --rescaled <file> is passed):

The input image will be N4 bias corrected. Optimally you would pass a brainmask via
--mask. If --mask auto is given, all 0 values will be masked to speed-up correction 
and avoid influence of flat zero regions on the bias field. If no mask is given, no 
mask is applied. The mean image intensity of the --out file is adjusted to be equal
to the mean intensity of the input.


WM Normalization and UCHAR (done if --rescale <filename> is passed):

There are several options for additional normalization, depending on additional
parameters. The biasfield corrected image will always be converted and saved as
UCHAR.

After bias correction, the image will be rescaled with the goal to normalize the
target white matter intensity to values usually around 105-110.

The most reliable way to achieve this white matter normalization is to pass a brain 
segmentation (via --aseg). Then, the white matter regions from the aseg are used to 
rescale the intensity such that their average intensity will be at 105 (similar to 
FreeSurfer's nu.mgz).

If no brain segmentation (--aseg) is passed, the script tries to find an appropriate 
segmentation via a ball of radius 50 voxels around the center of the image and computes 
the 90th percentile of intensities in this part of the image. The center is found 
by one of the following two methods:

If a talairach.xfm transform (--tal) is passed, the center of the ball
is placed at the origin of the Talairach space.

If a brain mask (--mask) is passed, the center is placed at the centroid of the
brainmask. 

One of --mask, --tal, --aseg must be passed to achieve rescaling.

Original Author: Martin Reuter
Date: Mar-18-2022

Modified: David Kügler
Date: Feb-27-2024
"""

HELP_VERBOSITY = "Logging verbosity: 0 (none), 1 (normal), 2 (debug)"
HELP_INVOL = "path to input.nii.gz"
HELP_OUTVOL = "path to corrected.nii.gz"
HELP_UCHAR = ("sets the output dtype to uchar (only applies to outvol, rescalevol "
              "is uchar by default.)")
HELP_RESCALED = "path to rescaled.nii.gz"
HELP_MASK = "optional: path to mask.nii.gz"
HELP_ASEG = "optional: path to aseg or aseg+dkt image to find the white matter mask"
HELP_SHRINK_FACTOR = "<int> shrink factor, default: 4"
HELP_LEVELS = "<int> number of fitting levels, default: 4"
HELP_NUM_ITER = "<int> max number of iterations per level, default: 50"
HELP_THRESHOLD = "<float> convergence threshold, default: 0.0"
HELP_TALAIRACH = "<Path> file name of talairach.xfm if using this for finding origin"
HELP_THREADS = "<int> number of threads, default: 1"
LiteralSkipRescaling = Literal["skip rescaling"]
SKIP_RESCALING: LiteralSkipRescaling = "skip rescaling"
LiteralDoNotSave = Literal["do not save"]
DO_NOT_SAVE: LiteralDoNotSave = "do not save"

logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=str)


def path_or_(*constants: _T) -> Callable[[str], _T]:
    def wrapper(a: str) -> Path | _T:
        if a in constants:
            return a
        return Path(a)
    return wrapper


def options_parse():
    """
    Command line option parser.

    Returns
    -------
    options
        Namespace object holding options.
    """
    parser = argparse.ArgumentParser(
        description=HELPTEXT,
    )
    parser.add_argument(
        "-v", "--verbosity",
        dest="verbosity",
        choices=(0, 1, 2),
        default=-1,
        help=HELP_VERBOSITY,
        type=int,
    )
    parser.add_argument(
        "--in",
        dest="invol",
        type=Path,
        help=HELP_INVOL,
        required=True,
    )
    parser.add_argument(
        "--out",
        dest="outvol",
        help=HELP_OUTVOL,
        default=DO_NOT_SAVE,
        type=path_or_(DO_NOT_SAVE),
    )
    parser.add_argument(
        "--uchar",
        dest="dtype",
        action="store_const",
        const="uint8",
        help=HELP_UCHAR,
        default="keep",
    )
    parser.add_argument(
        "--rescale",
        dest="rescalevol",
        help=HELP_RESCALED,
        default=SKIP_RESCALING,
        type=path_or_(SKIP_RESCALING),
    )
    parser.add_argument(
        "--mask",
        dest="mask",
        help=HELP_MASK,
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--aseg",
        dest="aseg",
        help=HELP_ASEG,
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--shrink",
        dest="shrink",
        help=HELP_SHRINK_FACTOR,
        default=4,
        type=int,
    )
    parser.add_argument(
        "--levels",
        dest="levels",
        help=HELP_LEVELS,
        default=4,
        type=int,
    )
    parser.add_argument(
        "--numiter",
        dest="numiter",
        help=HELP_NUM_ITER,
        default=50,
        type=int,
    )
    parser.add_argument(
        "--thres",
        dest="thres",
        help=HELP_THRESHOLD,
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--tal",
        dest="tal",
        help=HELP_TALAIRACH,
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--threads",
        dest="threads",
        help=HELP_THREADS,
        default=1,
        type=int,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="$Id: N4_bias_correct.py,v 2.1 2024/02/27 20:02:08 mreuter,dkuegler Exp $"
    )
    return parser.parse_args()


def itk_n4_bfcorrection(
        itk_image: sitk.Image,
        itk_mask: sitk.Image | None = None,
        shrink: int = 4,
        levels: int = 4,
        numiter: int = 50,
        thres: float = 0.0,
) -> sitk.Image:
    """
    Perform the bias field correction.

    Parameters
    ----------
    itk_image : sitk.Image
        N-dimensional image.
    itk_mask : Optional[sitk.Image]
        Image mask. Defaults to None. Optional.
    shrink : int
        Shrink factors. Defaults to 4.
    levels : int
        Number of levels for maximum number of iterations. Defaults to 4.
    numiter : int
        Maximum number if iterations. Defaults 50.
    thres : float
        Convergence threshold. Defaults to 0.0.

    Returns
    -------
    itk_bfcorr_image
        Bias field corrected image.
    """
    _logger = logging.getLogger(__name__ + ".itk_n4_bfcorrection")
    # if no mask is passed, create a simple mask from the image
    if itk_mask:
        # binarize mask
        itk_mask = itk_mask > 0
    else:
        itk_mask = sitk.Abs(itk_image) >= 0
        itk_mask.CopyInformation(itk_image)
        _logger.info("- mask: ones (default)")

    itk_orig = itk_image

    # subsample image
    if shrink > 1:
        itk_image = sitk.Shrink(itk_image, [shrink] * itk_image.GetDimension())
        itk_mask = sitk.Shrink(itk_mask, [shrink] * itk_image.GetDimension())

    # init corrector
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([numiter] * levels)
    corrector.SetConvergenceThreshold(thres)

    # bias correct image
    sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(1e-04)
    corrector.Execute(itk_image, itk_mask)

    # we need to apply bias field to original input
    log_bias_field = corrector.GetLogBiasFieldAsImage(itk_orig)
    itk_bfcorr_image = itk_orig / sitk.Exp(log_bias_field)

    # should already be Float 32
    # itk_bfcorr_image = sitk.Cast(itk_bfcorr_image, sitk.sitkFloat32)
    if itk_image.GetPixelIDTypeAsString() != "32-bit float":
        _logger.info(f"The data type is: {itk_image.GetPixelIDTypeAsString()}")
    return itk_bfcorr_image


def normalize_wm_mask_ball(
        itk_image: sitk.Image,
        itk_mask: sitk.Image | None = None,
        radius: float = 50.,
        centroid: np.ndarray | None = None,
        target_wm: float = 110.,
        target_bg: float = 3.
) -> sitk.Image:
    """
    Normalize WM image by Mask and optionally ball around talairach center.

    Parameters
    ----------
    itk_image : sitk.Image
        N-dimensional itk image.
    itk_mask : sitk.Image, optional
        Image mask.
    radius : float, int, default=50
        Radius of ball around centroid. Defaults to 50.
    centroid : np.ndarray
        Brain centroid.
    target_wm : float | int
        Target white matter intensity. Defaults to 110.
    target_bg : float | int
        Target background intensity. Defaults to 3 (1% of 255).

    Returns
    -------
    normalized_image : sitk.Image
        Normalized WM image.
    """
    _logger = logging.getLogger(__name__ + ".normalize_wm_mask_ball")
    _logger.info(f"- centroid: {centroid}")
    _logger.info(f"- size: {itk_image.GetSize()}")
    _logger.info(f"- spacing: {' '.join(f'{f:.2f}' for f in itk_image.GetSpacing())}")

    # distance image to centroid
    isize = itk_image.GetSize()
    ispace = itk_image.GetSpacing()

    def get_distance(axis):
        ii = np.arange(isize[2 - axis])
        for i in range(3):
            if i != axis:
                ii = np.expand_dims(ii, 0 if i < axis else -1)
        xx = ispace[axis] * (ii - centroid[axis])
        return xx * xx

    zz, yy, xx = map(get_distance, range(3))
    distance = xx + yy + zz
    ball = distance < radius * radius

    # get ndarray from sitk image
    image = sitk.GetArrayFromImage(itk_image)
    # get 90th percentiles of intensities in ball (to identify WM intensity)
    source_intensity_bg, source_intensity_wm = np.percentile(image[ball], [1, 90])

    _logger.info(
        f"- source background intensity: {source_intensity_bg:.2f}"
        f"- source white matter intensity: {source_intensity_wm:.2f}"
    )

    return normalize_img(
        itk_image,
        itk_mask,
        (source_intensity_bg, source_intensity_wm),
        (target_bg, target_wm),
    )


def normalize_wm_aseg(
        itk_image: sitk.Image,
        itk_mask: sitk.Image | None,
        itk_aseg: sitk.Image,
        target_wm: float = 110.,
        target_bg: float = 3.
) -> sitk.Image:
    """
    Normalize WM image so the white matter has a mean intensity of target_wm and the
    background has intensity target_bg.

    Parameters
    ----------
    itk_image : sitk.Image
        N-dimensional itk image.
    itk_mask : sitk.Image | None
        Image mask.
    itk_aseg : sitk.Image
        Aseg-like segmentation image to find WM.
    radius : float, int, default=50
        Radius of ball around centroid. Defaults to 50.
    centroid : np.ndarray, optional
        Image centroid. Defaults to None.
    target_wm : float | int
        Target white matter intensity. Defaults to 110.
    target_bg : float | int
        Target background intensity Defaults to 3 (1% of 255).

    Returns
    -------
    normed : sitk.Image
        Normalized WM image.
    """
    _logger = logging.getLogger(__name__ + ".normalize_wm_aseg")

    # get 1 and 90 percentiles of intensities in ball
    img = sitk.GetArrayFromImage(itk_image)
    aseg = sitk.GetArrayFromImage(itk_aseg)
    # Left and Right White Matter
    mask = (aseg == 2) | (aseg == 41)
    source_wm_intensity = np.mean(img[mask]).item()
    # mask = (aseg == 4) | (aseg == 43)
    # source_bg = np.mean(img[mask]).item()
    source_bg = np.percentile(img.flat[::100], 1)

    _logger.info(
        f"- source white matter intensity: {source_wm_intensity:.2f}"
    )

    return normalize_img(
        itk_image,
        itk_mask,
        (source_bg, source_wm_intensity),
        (target_bg, target_wm),
    )


def normalize_img(
        itk_image: sitk.Image,
        itk_mask: sitk.Image | None,
        source_intensity: tuple[float, float],
        target_intensity: tuple[float, float]
) -> sitk.Image:
    """
    Normalize image by source and target intensity values.

    Parameters
    ----------
    itk_image : sitk.Image
        Input image to be normalized.
    itk_mask : sitk.Image | None
        Brain mask, voxels inside the mask are guaranteed to be > 0,
        None is optional.
    source_intensity : tuple[float, float]
        Source intensity range.
    target_intensity : tuple[float, float]
        Target intensity range.

    Returns
    -------
    sitk.Image
        Rescaled image.
    """
    _logger = logging.getLogger(__name__ + ".normalize_wm")
    # compute intensity transformation
    m = (
        (target_intensity[0] - target_intensity[1])
        / (source_intensity[0] - source_intensity[1])
    )
    _logger.info(f"- m: {m:.4f}")

    # itk_image already is Float32 and output should be also Float32, we clamp outside
    normalized = (itk_image - source_intensity[0]) * m + target_intensity[0]

    if itk_mask:
        # ensure normalized image is > 0, where mask is true
        correction_mask = cast(sitk.Image, (normalized < 1) & itk_mask)
        return normalized + sitk.Cast(correction_mask, normalized.GetPixelID())
    else:
        return normalized


def read_talairach_xfm(fname: Path | str) -> np.ndarray:
    """
    Read Talairach transform.

    Parameters
    ----------
    fname : str
        Filename to Talairach transform.

    Returns
    -------
    tal
        Talairach transform matrix.

    Raises
    ------
    ValueError
        if the file is of an invalid format.
    """
    _logger = logging.getLogger(__name__ + ".read_talairach_xfm")
    _logger.info(f"reading talairach transform from {fname}")
    with open(fname) as f:
        lines = f.readlines()

    try:
        transform_iter = iter(lines)
        # advance transform_iter to linear header
        _ = next(ln for ln in transform_iter if ln.lower().startswith("linear_"))
        # return the next 3 lines in transform_lines
        transform_lines = (ln for ln, _ in zip(transform_iter, range(3), strict=False))
        tal_str = [ln.replace(";", " ") for ln in transform_lines]
        tal = np.genfromtxt(tal_str)
        tal = np.vstack([tal, [0, 0, 0, 1]])

        _logger.info(f"- tal: {tal}")

        return tal
    except StopIteration:
        _logger.error(msg := f"Could not find 'linear_' in {fname}.")
        raise ValueError(msg) from None
    except Exception as e:
        err = ValueError(f"Could not find taiairach transform in {fname}.")
        _logger.exception(err)
        raise err from e


def get_tal_origin_voxel(tal: npt.ArrayLike, image: sitk.Image) -> np.ndarray:
    """
    Get the origin of Talairach space in voxel coordinates.

    Parameters
    ----------
    tal : npt.ArrayLike
        Talairach transform matrix.
    image : sitk.Image
        Image.

    Returns
    -------
    vox_origin : np.ndarray
        Voxel coordinate of Talairach origin.
    """
    tal_inv = np.linalg.inv(tal)
    tal_origin = np.array(tal_inv[0:3, 3]).ravel()
    _logger = logging.getLogger(__name__ + ".get_talairach_origin_voxel")
    _logger.debug(f"Talairach Physical Origin: {tal_origin}")

    vox_origin = image.TransformPhysicalPointToIndex(tal_origin)
    _logger.info(f"Talairach Voxel Origin: {vox_origin}")
    return vox_origin


def print_options(options: dict):
    msg = ("",
           "N4 Bias Correction Parameters:",
           "",
           "- verbosity: {verbosity}",
           "- input volume: {invol}",
           "- output volume: {outvol}",
           "- rescaled volume: {rescalevol}",
           "- mask: {mask}" if options.get("mask") else "- mask: default (>0)",
           "- aseg: {aseg}" if options.get("aseg") else "- aseg: not defined",
           "- shrink factor: {shrink}",
           "- number fitting levels: {levels}",
           "- number iterations: {numiter}",
           "- convergence threshold: {thres}",
           "- talairach: {tal}" if options.get("tal") else None,
           "- threads: {threads}")

    _logger = logging.getLogger(__name__ + ".print_options")
    for m in msg:
        if m is not None:
            _logger.info(m.format(**options))


def get_image_mean(image: sitk.Image, mask: sitk.Image | None = None) -> float:
    """
    Get the mean of a sitk Image.

    Parameters
    ----------
    image : sitk.Image
        Image to get mean of.
    mask : sitk.Image, optional
        Optional mask to apply first.

    Returns
    -------
    mean : float
        The mean value of the image.
    """
    img = sitk.GetArrayFromImage(image)
    if mask is not None:
        mask = sitk.GetArrayFromImage(mask)
        img = img[mask]
    return np.mean(img).item()


def get_brain_centroid(itk_mask: sitk.Image) -> np.ndarray:
    """
    Get the brain centroid from a binary image.

    Parameters
    ----------
    itk_mask : sitk.Image
        Binary image to compute the centroid of its labeled region.

    Returns
    -------
    np.ndarray
        Brain centroid.
    """
    _logger = logging.getLogger(__name__ + ".get_brain_centroid")
    _logger.debug("No talairach center passed, estimating center from mask.")
    # extract centroid from mask
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(itk_mask)
    centroid_world = label_stats.GetCentroid(1)
    _logger.debug(f"centroid physical: {centroid_world}")
    centroid_index = itk_mask.TransformPhysicalPointToContinuousIndex(centroid_world)
    _logger.debug(f"centroid voxel: {centroid_index}")
    return itk_mask.TransformPhysicalPointToIndex(centroid_world)


def main(
    invol: Path,
    outvol: LiteralDoNotSave | Path = DO_NOT_SAVE,
    rescalevol: LiteralSkipRescaling | Path = SKIP_RESCALING,
    dtype: str = "keep",
    threads: int = 1,
    mask: Path | None = None,
    aseg: Path | None = None,
    shrink: int = 4,
    levels: int = 4,
    numiter: int = 50,
    thres: float = 0.0,
    tal: Path | None = None,
    verbosity: int = -1,
) -> int | str:
    if rescalevol == "skip rescaling" and outvol == DO_NOT_SAVE:
        return (
            "Neither the rescaled nor the unrescaled volume are saved, "
            "aborting."
        )

    # set number of threads
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)

    # read image (only nii supported) and convert to float32
    logger.debug(f"reading input volume {invol}")
    # itk_image = sitk.ReadImage(options.invol, sitk.sitkFloat32)
    itk_image, image_header = iio.readITKimage(
        str(invol),
        sitk.sitkFloat32,
        with_header=True,
    )

    # read mask (as uchar)
    has_mask = bool(mask)
    if has_mask:
        logger.debug(f"reading mask {mask}")
        itk_mask: sitk.Image | None = iio.readITKimage(
            str(mask),
            sitk.sitkUInt8,
            with_header=False
        )
        # binarize mask
        itk_mask = cast(sitk.Image, itk_mask > 0)
    else:
        itk_mask = None

    # call N4 correct
    logger.info("executing N4 correction ...")
    itk_bfcorr_image = itk_n4_bfcorrection(
        itk_image,
        itk_mask,
        shrink,
        levels,
        numiter,
        thres,
    )

    if outvol != DO_NOT_SAVE:
        logger.info("Skipping WM normalization, ignoring talairach and aseg inputs")

        # normalize to average input intensity
        kw_mask = {"mask": itk_mask} if has_mask else {}

        logger.info("- rescale")
        out_dtype = dtype.lower()
        if out_dtype == "uint8":
            from FastSurferCNN.data_loader.conform import getscale
            image = sitk.GetArrayFromImage(itk_bfcorr_image)
            l_image, m_image = 0, 255
            l_bf_img, scale = getscale(image, l_image, m_image)
            m_bf_img = l_bf_img + (m_image - l_image) / scale
            logger.info(f"   lower bound corrected: {l_bf_img:.4f}, upper bound corrected {m_bf_img:.4f})")
        else:
            m_bf_img = get_image_mean(itk_bfcorr_image, **kw_mask)
            m_image = get_image_mean(itk_image, **kw_mask)
            logger.info(f"   mean input: {m_image:.4f}, mean corrected {m_bf_img:.4f})")
            l_bf_img, l_image = 0.0, 0.0
            # rescale keeping the zero-point and the mean image intensity

        itk_outvol = normalize_img(itk_image, itk_mask, (l_bf_img, m_bf_img), (l_image, m_image))

        if out_dtype in ("uint8", "int8", "uint16", "int16"):
            dtype_info = np.iinfo(np.dtype(out_dtype))
            itk_outvol = sitk.Clamp(itk_outvol, lowerBound=dtype_info.min, upperBound=dtype_info.max)

        if out_dtype != "keep":
            logger.info(f"converting outvol to {dtype.upper()}")
            cap_dtype = out_dtype
            for prefix in ("i", "ui", "f"):
                if cap_dtype.startswith(prefix):
                    cap_dtype = prefix.upper() + cap_dtype[len(prefix):]
            sitk_dtype = getattr(sitk, "sitk" + cap_dtype)
            itk_outvol = sitk.Cast(itk_outvol, sitk_dtype)
            image_header.set_data_dtype(np.dtype(out_dtype))

        # write image
        logger.info(f"writing {type(outvol).__name__}: {outvol}")
        iio.writeITKimage(itk_outvol, str(outvol), image_header)

    if rescalevol == SKIP_RESCALING:
        logger.info("Skipping WM normalization, ignoring talairach and aseg inputs")
    else:
        target_wm = 110.
        # do some rescaling

        if aseg:  # has aseg
            # used to be 110, but we found experimentally, that freesurfer wm-normalized
            # intensity insde the WM mask is closer to 105 (but also quite inconsistent).
            # So when we have a WM mask, we need to use 105 and not 110 as for the
            # percentile approach above.
            target_wm = 105.

            logger.info(f"normalize WM to {target_wm:.1f} (find WM from aseg)")
            # only grab the white matter
            itk_aseg = iio.readITKimage(str(aseg), with_header=False)

            itk_bfcorr_image = normalize_wm_aseg(
                itk_bfcorr_image,
                itk_mask,
                itk_aseg,
                target_wm=target_wm
            )
        else:
            logger.info(f"normalize WM to {target_wm:.1f} (find WM from mask & talairach)")
            if tal:
                talairach_center = read_talairach_xfm(tal)
                brain_centroid = get_tal_origin_voxel(talairach_center, itk_image)
            elif has_mask:
                brain_centroid = get_brain_centroid(itk_mask)
            else:
                return ("Neither --tal, --mask, nor --aseg are passed, but "
                        "rescaling is requested.")

            itk_bfcorr_image = normalize_wm_mask_ball(
                itk_bfcorr_image,
                itk_mask,
                centroid=brain_centroid,
                target_wm=target_wm
            )

        logger.info("converting rescaled to UCHAR")
        itk_bfcorr_image = sitk.Cast(
            sitk.Clamp(itk_bfcorr_image, lowerBound=0, upperBound=255), sitk.sitkUInt8
        )

        # write image
        logger.info(f"writing {type(rescalevol).__name__}: {rescalevol}")
        iio.writeITKimage(itk_bfcorr_image, str(rescalevol), image_header)

    return 0


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()
    LOGLEVEL = (logging.WARNING, logging.INFO, logging.DEBUG)
    FORMAT = "%(message)s"
    if options.verbosity >= 0:
        FORMAT = "%(levelname)s (%(module)s:%(lineno)s): " + FORMAT

    logging.basicConfig(stream=sys.stdout, format=FORMAT)
    logging.getLogger().setLevel(LOGLEVEL[abs(options.verbosity)])

    if options.rescalevol == "skip rescaling" and options.outvol == "do not save":
        logger.error("Neither the rescaled nor the unrescaled volume are saved, aborting.")
        sys.exit(1)

    args = vars(options)
    print_options(args)
    invol = args.pop("invol")

    sys.exit(main(invol, **args))
