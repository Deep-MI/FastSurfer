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
import sys
from functools import partial
from pathlib import Path
from typing import TypeVar, cast

import nibabel as nib
import numpy as np
from numpy import typing as npt
from scipy import ndimage

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.data.constants import FORNIX_LABEL, SUBSEGMENT_LABELS
from FastSurferCNN.data_loader.conform import is_conform
from FastSurferCNN.reduce_to_aseg import reduce_to_aseg_and_save
from FastSurferCNN.utils.arg_types import path_or_none
from FastSurferCNN.utils.brainvolstats import mask_in_array
from FastSurferCNN.utils.parallel import thread_executor

_T = TypeVar("_T", bound=np.number)

logger = logging.get_logger(__name__)

HELPTEXT = """
Script to add corpus callosum segmentation (CC, FreeSurfer IDs 251-255) to
deep-learning prediction (e.g. aparc.DKTatlas+aseg.deep.mgz).


USAGE:
paint_cc_into_pred  -in_cc <input_seg_with_cc> -in_pred <input_seg_without_cc> -out <output_seg>


Dependencies:
    Python 3.8+

    Nibabel to read and write FreeSurfer data
    http://nipy.org/nibabel/

Original Author: Leonie Henschel
Date: Jul-10-2020

"""


def argument_parse():
    """Create a command line interface and return command line options.
    """
    parser = make_parser()

    args = parser.parse_args()

    if args.input_cc is None or args.input_pred is None or args.output is None:
        sys.exit("ERROR: Please specify input and output segmentations")

    return args


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(usage=HELPTEXT)
    parser.add_argument(
        "--input_cc",
        "-in_cc",
        dest="input_cc",
        type=Path,
        required=True,
        help="path to input segmentation with Corpus Callosum (IDs 251-255 in FreeSurfer space)",
    )
    parser.add_argument(
        "--input_pred",
        "-in_pred",
        dest="input_pred",
        type=Path,
        required=True,
        help="path to input segmentation Corpus Callosum should be added to.",
    )
    parser.add_argument(
        "--output",
        "-out",
        dest="output",
        type=Path,
        required=True,
        help="path to output (input segmentation + added CC)",
    )
    parser.add_argument(
        "--reduce_to_aseg",
        "-aseg",
        dest="aseg",
        type=path_or_none,
        required=False,
        help="optionally also reduce the resulting segmentation to aseg and save separately.",
        default=None,
    )
    return parser


def paint_in_cc(pred: npt.NDArray[np.int_], 
                aseg_cc: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Paint corpus callosum segmentation into aseg+dkt segmentation map.

    Parameters
    ----------
    pred : npt.NDArray[np.int_]
        Deep-learning segmentation map.
    aseg_cc : npt.NDArray[np.int_]
        Aseg segmentation with CC.

    Returns
    -------
    npt.NDArray[np.int_]
        Segmentation map with added CC.

    Notes
    -----
    This function modifies the original array and does not create a copy.
    The CC labels (251-255) from aseg_cc are copied into pred.
    """
    cc_mask = mask_in_array(aseg_cc, SUBSEGMENT_LABELS)
    
    # Count what's being replaced
    replaced_labels = pred[cc_mask]
    num_wm_replaced = np.sum((replaced_labels == 2) | (replaced_labels == 41))
    num_other_replaced = np.sum((replaced_labels != 0) & (replaced_labels != 2) & (replaced_labels != 41))
    num_background_replaced = np.sum(replaced_labels == 0)
    
    logger.info(f"Painting CC: {np.sum(cc_mask)} voxels (replacing {num_wm_replaced} WM, "
                f"{num_background_replaced} background, {num_other_replaced} other)")
    
    pred[cc_mask] = aseg_cc[cc_mask]
    return pred

def _fill_gaps_in_direction(
    corrected_pred: npt.NDArray[np.int_],
    potential_fill: npt.NDArray[np.bool_],
    source_binary: npt.NDArray[np.bool_],
    target_binary: npt.NDArray[np.bool_],
    x_slice: int,
    direction: str,
    max_gap_voxels: int,
    fillable_labels: set[int]
) -> int:
    """Fill gaps between source and target masks in a specific direction.
    
    Parameters
    ----------
    corrected_pred : npt.NDArray[np.int_]
        The segmentation array to modify in place.
    potential_fill : npt.NDArray[np.bool_]
        2D mask of potential fill regions for this slice.
    source_binary : npt.NDArray[np.bool_]
        2D binary mask of source structure (e.g., CC).
    target_binary : npt.NDArray[np.bool_]
        2D binary mask of target structure (e.g., ventricle).
    x_slice : int
        The x-coordinate of the current slice.
    direction : str
        Either 'inferior-superior' (iterate over z) or 'anterior-posterior' (iterate over y).
    max_gap_voxels : int
        Maximum gap size in voxels for this direction.
    fillable_labels : set[int]
        Set of label values that can be replaced (e.g., {0, 2, 41} for background and WM).
    
    Returns
    -------
    int
        Number of voxels filled.
    """
    voxels_filled = 0
    
    if direction == 'inferior-superior':
        # Iterate over z dimension
        for z in range(potential_fill.shape[1]):
            potential_fill_line = potential_fill[:, z]
            labeled_gaps, num_gaps = ndimage.label(potential_fill_line)
            source_line = source_binary[:, z]
            target_line = target_binary[:, z]

            for gap_label in range(1, num_gaps + 1):
                gap_mask = labeled_gaps == gap_label

                # Check that both source and target are connected to the gap
                dilated_gap_mask = ndimage.binary_dilation(gap_mask, iterations=1)
                if not np.any(source_line & dilated_gap_mask):
                    continue
                if not np.any(target_line & dilated_gap_mask):
                    continue

                # Get the target label from adjacent target voxels
                target_label_location = np.where(target_line & dilated_gap_mask)[0]
                if len(target_label_location) == 0:
                    continue
                target_label = corrected_pred[x_slice, target_label_location[0], z]

                # Check gap size
                if np.sum(gap_mask) > max_gap_voxels:
                    continue

                # Fill voxels that have fillable labels
                current_labels = corrected_pred[x_slice, :, z]
                fill_mask = gap_mask & np.isin(current_labels, list(fillable_labels))
                voxels_filled += np.sum(fill_mask)
                corrected_pred[x_slice, :, z][fill_mask] = target_label
                
    elif direction == 'anterior-posterior':
        # Iterate over y dimension
        for y in range(potential_fill.shape[0]):
            potential_fill_line = potential_fill[y, :]
            labeled_gaps, num_gaps = ndimage.label(potential_fill_line)
            source_line = source_binary[y, :]
            target_line = target_binary[y, :]

            for gap_label in range(1, num_gaps + 1):
                gap_mask = labeled_gaps == gap_label

                # Check that both source and target are connected to the gap
                dilated_gap_mask = ndimage.binary_dilation(gap_mask, iterations=1)
                if not np.any(source_line & dilated_gap_mask):
                    continue
                if not np.any(target_line & dilated_gap_mask):
                    continue

                # Get the target label from adjacent target voxels
                target_label_location = np.where(target_line & dilated_gap_mask)[0]
                if len(target_label_location) == 0:
                    continue
                target_label = corrected_pred[x_slice, y, target_label_location[0]]

                # Check gap size
                if np.sum(gap_mask) > max_gap_voxels:
                    continue

                # Fill voxels that have fillable labels
                current_labels = corrected_pred[x_slice, y, :]
                fill_mask = gap_mask & np.isin(current_labels, list(fillable_labels))
                voxels_filled += np.sum(fill_mask)
                corrected_pred[x_slice, y, :][fill_mask] = target_label
    
    return voxels_filled


def _fill_gaps_between_structures(
    corrected_pred: npt.NDArray[np.int_],
    source_mask: npt.NDArray[np.bool_],
    target_mask: npt.NDArray[np.bool_],
    voxel_size: tuple[float, float, float],
    close_gap_size_mm: float,
    fillable_labels: set[int],
    description: str
) -> int:
    """Fill small gaps between two structures.
    
    Parameters
    ----------
    corrected_pred : npt.NDArray[np.int_]
        The segmentation array to modify in place.
    source_mask : npt.NDArray[np.bool_]
        3D binary mask of source structure (e.g., CC).
    target_mask : npt.NDArray[np.bool_]
        3D binary mask of target structure (e.g., ventricle or background).
    voxel_size : tuple[float, float, float]
        Voxel size in mm.
    close_gap_size_mm : float
        Maximum gap size in mm.
    fillable_labels : set[int]
        Set of label values that can be replaced.
    description : str
        Description for logging.
    
    Returns
    -------
    int
        Number of voxels filled.
    """
    # Convert mm gap size to voxels
    max_gap_vox_anterior_posterior = int(np.ceil(close_gap_size_mm / voxel_size[1]))
    max_gap_vox_inferior_superior = int(np.ceil(close_gap_size_mm / voxel_size[2]))
    max_gap_vox_max = max(max_gap_vox_anterior_posterior, max_gap_vox_inferior_superior)
    
    voxels_filled = 0
    
    # Process each slice independently
    for x in range(corrected_pred.shape[0]):
        source_slice = source_mask[x]
        target_slice = target_mask[x]

        # Skip slices without both structures
        if not (source_slice.any() and target_slice.any()):
            continue
        
        # Create binary masks for this slice
        source_binary = source_slice.astype(bool)
        target_binary = target_slice.astype(bool)

        # Dilate both masks to find potential connection points
        source_dilated = ndimage.binary_dilation(source_binary, iterations=max_gap_vox_max)
        target_dilated = ndimage.binary_dilation(target_binary, iterations=max_gap_vox_max)

        # Find voxels that are adjacent to both structures but not part of either
        potential_fill = (source_dilated & target_dilated) & ~(source_binary | target_binary)

        # Fill gaps in inferior-superior direction
        voxels_filled += _fill_gaps_in_direction(
            corrected_pred, potential_fill, source_binary, target_binary,
            x, 'inferior-superior', max_gap_vox_inferior_superior, fillable_labels
        )

        # Fill gaps in anterior-posterior direction
        voxels_filled += _fill_gaps_in_direction(
            corrected_pred, potential_fill, source_binary, target_binary,
            x, 'anterior-posterior', max_gap_vox_anterior_posterior, fillable_labels
        )
    
    if voxels_filled > 0:
        logger.info(f"Filled {voxels_filled} voxels {description}")
    
    return voxels_filled


def correct_wm_ventricles(
    aseg_cc: npt.NDArray[np.int_],
    fornix_mask: npt.NDArray[np.bool_],
    voxel_size: tuple[float, float, float],
    close_gap_size_mm: float = 3.0
) -> npt.NDArray[np.int_]:
    """Fill small gaps between corpus callosum, ventricles, and background.

    This function performs two gap-filling operations:
    1. Fills WM and background gaps between CC and ventricles with ventricle labels
    2. Fills WM gaps between CC and background with background label
    
    Note: Fornix and non-CC-connected WM component removal are intentionally not implemented
    in this function as they have been removed from the processing pipeline.

    Parameters
    ----------
    aseg_cc : npt.NDArray[np.int_]
        Aseg segmentation with CC already painted in.
    fornix_mask : npt.NDArray[np.bool_]
        Mask of the fornix. Not currently used (kept for interface compatibility).
    voxel_size : tuple[float, float, float]
        Voxel size of the aseg image in mm.
    close_gap_size_mm : float, default=3.0
        Maximum size of the gap to fill in millimeters.

    Returns
    -------
    npt.NDArray[np.int_]
        Corrected segmentation map with filled gaps.
    """
    # Create a copy to avoid modifying the original
    corrected_pred = aseg_cc.copy()
    
    # Get CC mask (labels 251-255)
    cc_mask = mask_in_array(aseg_cc, SUBSEGMENT_LABELS)

    # Get ventricle masks (left=4, right=43)
    ventricle_mask = (aseg_cc == 4) | (aseg_cc == 43)
    
    # Get background mask
    background_mask = aseg_cc == 0
    
    # 1. Fill gaps between CC and ventricles (replace WM and background with ventricle labels)
    _fill_gaps_between_structures(
        corrected_pred, cc_mask, ventricle_mask, voxel_size, close_gap_size_mm,
        fillable_labels={0, 2, 41},  # background and WM
        description="between CC and ventricles (WM/background → ventricle)"
    )
    
    # 2. Fill WM gaps between CC and background (replace WM with background)
    _fill_gaps_between_structures(
        corrected_pred, cc_mask, background_mask, voxel_size, close_gap_size_mm,
        fillable_labels={2, 41},  # only WM
        description="between CC and background (WM → background)"
    )

    return corrected_pred


if __name__ == "__main__":
    from FastSurferCNN.utils import nibabelImage

    # Command Line options are error checking done here
    options = argument_parse()

    logging.setup_logging()

    logger.info(f"Reading inputs: {options.input_cc} {options.input_pred}...")
    cc_seg_image = cast(nibabelImage, nib.load(options.input_cc))
    cc_seg_data = np.asanyarray(cc_seg_image.dataobj)
    aseg_image = cast(nibabelImage, nib.load(options.input_pred))
    aseg_data = np.asanyarray(aseg_image.dataobj)

    def _is_conform(img, dtype, verbose):
        return is_conform(img, vox_size=None, img_size=None, verbose=verbose, dtype=dtype)

    conform_args = (cc_seg_image, aseg_image), (np.uint8, np.integer)
    conform_checks = list(thread_executor().map(partial(_is_conform, verbose=False), *conform_args))

    if not all(conform_checks):
        names = []
        dtypes = []
        for conform_check, img, dtype, name in zip(conform_checks, *conform_args, ("CC", "Prediction"), strict=True):
            if not conform_check:
                _is_conform(img, dtype, verbose=True)
                names.append(name)
                dtypes.append(dtype.name if hasattr(dtype, "name") else str(dtype))
        sys.exit(
            f"Error: {' and '.join(names)} input image is not conformed (LIA orientation, {'/'.join(dtypes)} dtype). "
            "Please conform the image(s) using the conform.py script."
        )
    if not np.allclose(cc_seg_image.affine, aseg_image.affine):
        sys.exit("Error: The affine matrices of the aseg and the corpus callosum images are not the same.")

    # Count initial labels before any modifications
    initial_cc = np.sum(mask_in_array(aseg_data, SUBSEGMENT_LABELS))
    initial_fornix = np.sum(aseg_data == FORNIX_LABEL)
    initial_wm = np.sum((aseg_data == 2) | (aseg_data == 41))
    initial_ventricles = np.sum((aseg_data == 4) | (aseg_data == 43))

    # Paint CC into prediction (modifies aseg_data in place)
    paint_in_cc(aseg_data, cc_seg_data)

    # Apply ventricle gap filling corrections
    fornix_mask = cc_seg_data == FORNIX_LABEL
    voxel_size = tuple(aseg_image.header.get_zooms())
    pred_corrected = correct_wm_ventricles(aseg_data, fornix_mask, voxel_size)

    logger.info(f"Writing segmentation with corpus callosum to: {options.output}")
    pred_with_cc_fin = nib.MGHImage(pred_corrected, aseg_image.affine, aseg_image.header)
    io_fut = thread_executor().submit(pred_with_cc_fin.to_filename, options.output)

    if options.aseg is not None:
        rta_fut = thread_executor().submit(
            reduce_to_aseg_and_save,
            pred_corrected,
            aseg_image.affine,
            aseg_image.header,
            options.aseg,
        )
    else:
        rta_fut = None

    # Count final labels
    final_cc = np.sum(mask_in_array(pred_corrected, SUBSEGMENT_LABELS))
    final_fornix = np.sum(pred_corrected == FORNIX_LABEL)
    final_wm = np.sum((pred_corrected == 2) | (pred_corrected == 41))
    final_ventricles = np.sum((pred_corrected == 4) | (pred_corrected == 43))

    wm_change = final_wm - initial_wm
    vent_change = final_ventricles - initial_ventricles
    cc_change = final_cc - initial_cc
    
    logger.info(f"Changes: Corpus Callosum {'+' if cc_change >= 0 else ''}{cc_change}, "
                f"White Matter {'+' if wm_change >= 0 else ''}{wm_change}, "
                f"Ventricles {'+' if vent_change >= 0 else ''}{vent_change}")

    # Wait for all IO operations to complete
    io_fut.result()
    if rta_fut is not None:
        rta_fut.result()

