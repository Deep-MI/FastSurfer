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
import argparse
import sys
from pathlib import Path
from typing import Optional, cast, Tuple
import logging

import SimpleITK as sitk
import numpy as np
from numpy import typing as npt

import image_io as iio


HELPTEXT = """

Script to call SITK N4 Bias Correction

USAGE:
N4_bias_correct  --in <img.nii> [--out <corrected.nii>] [--rescaled <rescaled.nii>] OPTIONS


Dependencies:
    Python 3.8

    SimpleITK https://simpleitk.org/ (v2.1.1)


N4 Bias Correction (performed if either --out <file> or --rescaled <file> is passed):

The input image will be N4 bias corrected. Optimally you would pass a brainmask via
--mask. If --mask auto is given, all 0 values will be masked to speed-up correction 
and avoid influence of flat zero regions on the bias field. If no mask is given, no 
mask is applied. 


WM Normalization and UCHAR (done if --rescale <filename> is passed):

There are several options for normalization, depending on parameters and passed inputs. 
The biasfield corrected image will always be converted and saved as UCHAR.

After bias field computation and application, the image will be rescaled. This procedure 
is called WM normalization. The goal is to either, achieve a target white matter 
intensity of 110 or similar mean image intensity as the input image.

The most reliable way to achieve this white matter normalization is to pass a brain 
segmentation (via --aseg). Then, the white matter regions from the aseg are used to 
rescale the intensity such that their average intensity will be 110.

If no brain segmentation (--aseg) is passed, the script tries to find an appropriate 
segmentation via a ball of radius 50 voxels around the center of the image and computes 
the 90th percentile of intensities in this part of the image.
To find the center of the brain, if a talairach.xfm transform (--tal) is passed, we 
use the origin of the Talairach space as center for the ball, or, if a brain mask 
(--mask) is passed, the center is computed based on the brain mask. 

One of --mask, --tal, --aseg must be passed to achieve rescaling.

Original Author: Martin Reuter
Date: Mar-18-2022

Modified: David Kügler
Date: Oct-25-2023
"""

h_verbosity = "Logging verbosity: 0 (none), 1 (normal), 2 (debug)"
h_invol = "path to input.nii.gz"
h_outvol = "path to corrected.nii.gz"
h_rescaled = "path to rescaled.nii.gz"
h_mask = "optional: path to mask.nii.gz"
h_aseg = "optional: path to aseg or aseg+dkt image to find the white matter mask"
h_shrink = "<int> shrink factor, default: 4"
h_levels = "<int> number of fitting levels, default: 4"
h_numiter = "<int> max number of iterations per level, default: 50"
h_thres = "<float> convergence threshold, default: 0.0"
h_tal = "<string> file name of talairach.xfm if using this for finding origin"
h_threads = "<int> number of threads, default: 1"


def options_parse():
    """Command line option parser.

    Returns
    -------
    options
        object holding options

    """
    parser = argparse.ArgumentParser(
        description=HELPTEXT,
    )
    parser.add_argument(
        "-v", "--verbosity",
        dest="verbosity", choices=(0, 1, 2), default=-1, help=h_verbosity, type=int
    )
    parser.add_argument("--in", dest="invol", help=h_invol, required=True)
    parser.add_argument("--out", dest="outvol", help=h_outvol, default="do not save")
    parser.add_argument("--rescale", dest="rescalevol", help=h_rescaled, default="skip rescaling")
    parser.add_argument("--mask", dest="mask", help=h_mask, default=None)
    parser.add_argument("--aseg", dest="aseg", help=h_aseg, default=None)
    parser.add_argument("--shrink", dest="shrink", help=h_shrink, default=4, type=int)
    parser.add_argument("--levels", dest="levels", help=h_levels, default=4, type=int)
    parser.add_argument(
        "--numiter", dest="numiter", help=h_numiter, default=50, type=int
    )
    parser.add_argument("--thres", dest="thres", help=h_thres, default=0.0, type=float)
    parser.add_argument("--tal", dest="tal", help=h_tal, default=None)
    parser.add_argument(
        "--threads", dest="threads", help=h_threads, default=1, type=int
    )
    parser.add_argument(
        "--version",
        action="version",
        version="$Id: N4_bias_correct.py,v 2.0 2023/10/25 20:02:08 mreuter,dkuegler Exp $"
    )
    return parser.parse_args()


def itk_n4_bfcorrection(
        itk_image: sitk.Image,
        itk_mask: Optional[sitk.Image] = None,
        shrink: int = 4,
        levels: int = 4,
        numiter: int = 50,
        thres: float = 0.0,
) -> sitk.Image:
    """Perform the bias field correction.

    Parameters
    ----------
    itk_image : sitk.Image
        n-dimensional image
    itk_mask : Optional[sitk.Image]
        Image mask. Defaults to None. Optional
    shrink : int
        Shrink factors. Defaults to 4
    levels : int
        Number of levels for maximum number of iterations. Defaults to 4
    numiter : int
        Maximum number if iterations. Defaults 50
    thres : float
        Convergence threshold. Defaults to 0.0

    Returns
    -------
    itk_bfcorr_image
        Bias field corrected image

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
        itk_mask: Optional[sitk.Image] = None,
        radius: float = 50.,
        centroid: Optional[np.ndarray] = None,
        target_wm: float = 110.,
        target_bg: float = 3.
) -> sitk.Image:
    """Normalize WM image by Mask and optionally ball around talairach center.

    Parameters
    ----------
    itk_image : sitk.Image
        n-dimensional itk image
    itk_mask : sitk.Image, optional
        Image mask.
    radius : float | int
        Defaults to 50 [MISSING]
    centroid : np.ndarray
        brain centroid.
    target_wm : float | int
        Target white matter intensity. Defaults to 110.
    target_bg : float | int
        target background intensity. Defaults to 3 (1% of 255)

    Returns
    -------
    normalized_image : sitk.Image
        Normalized WM image

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
    source_intensity = np.percentile(image[ball], [1, 90])

    _logger.info(
        f"- source background intensity: {source_intensity[0]:.2f}"
        f"- source white matter intensity: {source_intensity[1]:.2f}"
    )

    return normalize_img(itk_image, itk_mask, tuple(source_intensity.tolist()), (target_bg, target_wm))


def normalize_wm_aseg(
        itk_image: sitk.Image,
        itk_mask: Optional[sitk.Image],
        itk_aseg: sitk.Image,
        target_wm: float = 110.,
        target_bg: float = 30.
) -> sitk.Image:
    """Normalize WM image [MISSING].

    Parameters
    ----------
    itk_image : sitk.Image
        n-dimensional itk image
    itk_mask : sitk.Image | None
        Image mask.
    itk_aseg : sitk.Image
        aseg-like segmentation image to find WM.
    radius : float | int
        Defaults to 50 [MISSING]
    centroid : Optional[np.ndarray]
        Image centroid. Defaults to None
    target_wm : float | int
        target white matter intensity. Defaults to 110
    target_bg : float | int
        target background intensity Defaults to 30 (1% of 255)

    Returns
    -------
    normed : sitk.Image
        Normalized WM image

    """
    _logger = logging.getLogger(__name__ + ".normalize_wm_aseg")

    # get 1 and 90 percentiles of intensities in ball
    img = sitk.GetArrayFromImage(itk_image)
    aseg = sitk.GetArrayFromImage(itk_aseg)
    # Left and Right White Matter
    mask = (aseg == 2) | (aseg == 41)
    source_wm_intensity = np.mean(img[mask]).item()
    mask = (aseg == 4) | (aseg == 43)
    source_bg = np.mean(img[mask]).item()

    _logger.info(
        f"- source white matter intensity: {source_wm_intensity:.2f}"
    )

    return normalize_img(itk_image, itk_mask, (source_bg, source_wm_intensity), (target_bg, target_wm))


def normalize_img(
        itk_image: sitk.Image,
        itk_mask: Optional[sitk.Image],
        source_intensity: Tuple[float, float],
        target_intensity: Tuple[float, float]
) -> sitk.Image:
    """
    Normalize image by source and target intensity values.

    Parameters
    ----------
    itk_image : sitk.Image
    itk_mask : sitk.Image | None
    source_intensity : Tuple[float, float]
    target_intensity : Tuple[float, float]

    Returns
    -------
    Rescaled image
    """
    _logger = logging.getLogger(__name__ + ".normalize_wm")
    # compute intensity transformation
    m = (target_intensity[0] - target_intensity[1]) / (source_intensity[0] - source_intensity[1])
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
    """Read Talairach transform.

    Parameters
    ----------
    fname : str
        Filename to Talairach transform

    Returns
    -------
    tal
        Talairach transform matrix

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
        transf_start = [l.lower().startswith("linear_") for l in lines].index(True) + 1
        tal_str = [l.replace(";", " ") for l in lines[transf_start:transf_start + 3]]
        tal = np.genfromtxt(tal_str)
        tal = np.vstack([tal, [0, 0, 0, 1]])

        _logger.info(f"- tal: {tal}")

        return tal
    except Exception as e:
        err = ValueError(f"Could not find taiairach transform in {fname}.")
        _logger.exception(err)
        raise err from e


def get_tal_origin_voxel(tal: npt.ArrayLike, image: sitk.Image) -> np.ndarray:
    """Get the origin of Talairach space in voxel coordinates.

    Parameters
    ----------
    tal : npt.ArrayLike
        Talairach transform matrix.
    image : sitk.Image
        Image.

    Returns
    -------
    vox_origin : np.ndarray
        Voxel coordinate of Talairach origin

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
           "- rescaled volume: {rescaledvol}",
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


def get_image_mean(image: sitk.Image, mask: Optional[sitk.Image] = None) -> float:
    """
    Get the mean of a sitk Image.

    Parameters
    ----------
    image : sitk.Image
        image to get mean of
    mask : sitk.Image, optional
        optional mask to apply first

    Returns
    -------
    mean : float
    """
    img = sitk.GetArrayFromImage(image)
    if mask is not None:
        mask = sitk.GetArrayFromImage(mask)
        img = img[mask]
    return np.mean(img).item()


def get_brain_centroid(itk_mask: sitk.Image) -> np.ndarray:
    """
    Get the brain centroid from the itk_mask

    Parameters
    ----------
    itk_mask : sitk.Image

    Returns
    -------
    brain centroid

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


if __name__ == "__main__":

    # Command Line options are error checking done here
    options = options_parse()
    LOGLEVEL = (logging.WARNING, logging.INFO, logging.DEBUG)
    FORMAT = "" if options.verbosity < 0 else "%(levelname)s (%(module)s:%(lineno)s): "
    FORMAT += "%(message)s"
    logging.basicConfig(stream=sys.stdout, format=FORMAT)
    logging.getLogger().setLevel(LOGLEVEL[abs(options.verbosity)])
    logger = logging.getLogger(__name__)
    print_options(vars(options))

    if options.rescalevol == "skip rescaling" and options.outvol == "do not save":
        logger.error("Neither the rescaled nor the unrescaled volume are saved, aborting.")
        sys.exit(1)

    # set number of threads
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(options.threads)

    # read image (only nii supported) and convert to float32
    logger.debug(f"reading input volume {options.invol}")
    # itk_image = sitk.ReadImage(options.invol, sitk.sitkFloat32)
    itk_image, image_header = iio.readITKimage(
        options.invol, sitk.sitkFloat32, with_header=True
    )

    # read mask (as uchar)
    has_mask = bool(options.mask)
    if has_mask:
        logger.debug(f"reading mask {options.mask}")
        itk_mask: Optional[sitk.Image] = iio.readITKimage(
            options.mask,
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
        options.shrink,
        options.levels,
        options.numiter,
        options.thres,
    )

    target_wm = 110.

    if options.outvol != "do not save":
        logger.info("Skipping WM normalization, ignoring talairach and aseg inputs")

        # normalize to average input intensity
        kw_mask = {"mask": itk_mask} if has_mask else {}
        m_bf_img = get_image_mean(itk_bfcorr_image, **kw_mask)
        m_image = get_image_mean(itk_image, **kw_mask)
        logger.info("- rescale")
        logger.info(f"   mean input: {m_image:.4f}, mean corrected {m_bf_img:.4f})")
        # rescale keeping the zero-point and the mean image intensity
        itk_outvol = normalize_img(itk_image, itk_mask, (0., m_bf_img), (0., m_image))

        logger.info("converting outvol to UCHAR")
        itk_outvol = sitk.Cast(
            sitk.Clamp(itk_outvol, lowerBound=0, upperBound=255), sitk.sitkUInt8
        )

        # write image
        logger.info(f"writing: {options.outvol}")
        iio.writeITKimage(itk_outvol, options.outvol, image_header)


    if options.rescalevol == "skip rescaling":
        logger.info("Skipping WM normalization, ignoring talairach and aseg inputs")

    else:
        # do some rescaling
        if options.aseg:
            logger.info(f"normalize WM to {target_wm:.1f} (find WM from aseg)")
            # only grab the white matter
            itk_aseg = iio.readITKimage(options.aseg, with_header=False)

            itk_bfcorr_image = normalize_wm_aseg(
                itk_bfcorr_image,
                itk_mask,
                itk_aseg,
                target_wm=target_wm
            )
        else:
            logger.info(f"normalize WM to {target_wm:.1f} (find WM from mask & talairach)")
            if options.tal:
                talairach_center = read_talairach_xfm(options.tal)
                brain_centroid = get_tal_origin_voxel(talairach_center, itk_image)
            elif has_mask:
                brain_centroid = get_brain_centroid(itk_mask)
            else:
                logger.error("Neither --tal, --mask, nor --aseg are passed, but rescaling is requested.")
                sys.exit(1)

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
        logger.info(f"writing: {options.rescalevol}")
        iio.writeITKimage(itk_bfcorr_image, options.rescalevol, image_header)

    sys.exit(0)
