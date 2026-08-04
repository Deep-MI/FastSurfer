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


# IMPORTS
import argparse
import logging
import sys
from typing import cast

import nibabel as nib
import numpy as np
from skimage.morphology import dilation

from FastSurferCNN.utils import ShapeType

logger = logging.getLogger(__name__)

HELPTEXT = """
Script to perform quick quality checks for the input segmentation to identify gross errors.

USAGE:
quick_qc.py --asegdkt_segfile <aparc+aseg.mgz>


"""

VENT_LABELS = {
    "Left-Lateral-Ventricle": 4,
    "Right-Lateral-Ventricle": 43,
    "Left-choroid-plexus": 31,
    "Right-choroid-plexus": 63,
}
BG_LABEL = 0


def make_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for quick_qc.

    Returns
    -------
    argparse.ArgumentParser
        The argument parser object.
    """
    parser = argparse.ArgumentParser(
        description=HELPTEXT,
    )
    parser.add_argument(
        "--asegdkt_segfile",
        "--aparc_aseg_segfile",
        dest="asegdkt_segfile",
        help="Input aparc+aseg segmentation to be checked",
        required=True,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase output verbosity (can be used twice times for DEBUG output)",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0 2022/09/28 11:34:08 mreuter Exp $")
    return parser


def check_volume(asegdkt_segfile: np.ndarray, voxvol: float, thres: float = 0.70) -> bool:
    """
    Check if total volume is bigger or smaller than threshold.

    Parameters
    ----------
    asegdkt_segfile : np.ndarray
        The segmentation file.
    voxvol : float
        The volume of a voxel.
    thres : float, default=0.7
        The threshold for the total volume.

    Returns
    -------
    bool
        Whether or not total volume is bigger than the threshold `thres`.
    """
    logger.debug("Checking total volume ...")
    mask = asegdkt_segfile > 0
    total_vol = np.sum(mask) * voxvol / 1000000
    logger.debug(f"Voxel size in mm3: {voxvol}")
    logger.info(f"Total segmentation volume in liter: {np.round(total_vol, 2)}")
    return bool(total_vol > thres)


def get_region_bg_intersection_mask(
    seg_array: np.ndarray[ShapeType, np.dtype[np.integer]],
    region_labels: dict[str, int] = VENT_LABELS,
    bg_label=BG_LABEL,
):
    """
    Return a mask of the intersection between the voxels of a given region and background voxels.

    This is obtained by dilating the region by 1 voxel and computing the intersection with the background mask.

    The region can be defined by passing in the region_labels dict.

    Parameters
    ----------
    seg_array : numpy.ndarray
        Segmentation array.
    region_labels : dict, default=<dict VENT_LABELS>
        Dictionary whose values correspond to the desired region's labels (see Note).
    bg_label : int, default=<BG_LABEL>
        Label id of the background.

    Returns
    -------
    bg_intersect : numpy.ndarray
        Region and background intersection mask array.

    Notes
    -----
    VENT_LABELS is a dictionary containing labels for four regions related to the ventricles:
    "Left-Lateral-Ventricle", "Right-Lateral-Ventricle", "Left-choroid-plexus", "Right-choroid-plexus"
    along with their corresponding integer label values (see also FreeSurferColorLUT.txt).
    """
    from FastSurferCNN.utils.brainvolstats import mask_in_array

    region_array = mask_in_array(seg_array, list(region_labels.values()))
    region_array_dilated = dilation(region_array)

    bg_array = seg_array == bg_label
    return np.logical_and(bg_array, region_array_dilated)


def get_ventricle_bg_intersection_volume(
    seg_array: np.ndarray[ShapeType, np.dtype[np.integer]],
    voxvol: float,
) -> float:
    """
    Return a volume estimate for the intersection of ventricle voxels with background voxels.

    Parameters
    ----------
    seg_array : numpy.ndarray
        Segmentation array.
    voxvol : float
        Voxel volume.

    Returns
    -------
    intersection_volume : float
        Estimated volume of voxels in ventricle and background intersection.
    """
    bg_intersect_mask = get_region_bg_intersection_mask(seg_array)
    intersection_volume = bg_intersect_mask.sum() * voxvol

    return intersection_volume


if __name__ == "__main__":
    from FastSurferCNN.utils import nibabelImage
    from FastSurferCNN.utils.logging import setup_logging

    parser = make_parser()
    options = parser.parse_args()

    # default configuration of the logger, only stdout, no logfile
    setup_logging(log_level=options.verbose)

    logger.info(f"Reading in aparc+aseg: {options.asegdkt_segfile} ...")
    inseg = cast(nibabelImage, nib.load(options.asegdkt_segfile))
    inseg_data = np.asanyarray(inseg.dataobj)
    inseg_voxvol = float(np.prod(inseg.header.get_zooms()))

    # Ventricle-BG intersection volume check:
    logger.debug("Estimating ventricle-background intersection volume...")
    ventricle_volume = get_ventricle_bg_intersection_volume(inseg_data, inseg_voxvol)
    logger.info(f"Ventricle-background intersection volume in mm3: {ventricle_volume:.2f}")

    # Total volume check:
    if not check_volume(inseg_data, inseg_voxvol):
        logger.warning("Total segmentation volume is very small. Segmentation may be corrupted! Please check!")
    sys.exit(0)
