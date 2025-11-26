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
    pred[cc_mask] = aseg_cc[cc_mask]
    return pred

def correct_wm_ventricles(
    aseg_cc: npt.NDArray[np.int_],
    fornix_mask: npt.NDArray[np.bool_],
    voxel_size: tuple[float, float, float],
    close_gap_size_mm: float = 3.0
) -> npt.NDArray[np.int_]:
    """Correct WM mask and ventricle labels according to the CC and fornix masks.

    The function
    Take non-CC-connected WM components -> remove
    Take FN -> WM
    Fill space in superior inferior direction between CC and left/right Ventricle with corresponding Ventricle labels
    """

    # Create a copy to avoid modifying the original
    corrected_pred = aseg_cc.copy()

    # Get CC mask (labels 251-255)
    cc_mask = mask_in_array(aseg_cc, SUBSEGMENT_LABELS)

    # Get left and right ventricle masks
    all_ventricle_mask = (aseg_cc == 4) | (aseg_cc == 43)

    # Combine all WM labels
    all_wm_mask = (aseg_cc == 2) | (aseg_cc == 41)


    # 1. Fill space between CC and ventricles
    # Only fill small gaps (up to 3 voxels) between CC and ventricle boundaries
    #for ventricle_label, ventricle_mask in [(4, left_ventricle_mask), (43, right_ventricle_mask)]:
    
    # Process each slice independently
    for x in range(corrected_pred.shape[0]):
        cc_slice = cc_mask
        #vent_slice = ventricle_mask
        all_wm_slice = all_wm_mask

        if all_wm_slice.any() and cc_slice.any():

            # Dilate CC mask to find adjacent voxels, then check for overlap with component
            cc_dilated = ndimage.binary_dilation(cc_slice, iterations=1)
            # Label connected components in WM
            labeled_wm, num_components = ndimage.label(all_wm_slice)

            # Find components that are adjacent to CC and remove them
            for label in range(1, num_components + 1):
                component_mask = labeled_wm == label
                # Check if this component is adjacent to (touches) the CC
                if np.any(component_mask & cc_dilated):
                    corrected_pred[x][component_mask] = 0  # Set to background


            if fornix_mask[x].any():
                fornix_slice = fornix_mask[x]
                # count WM labels overlapping with fornix
                left_wm_overlap = np.sum(fornix_slice & (aseg_cc == 2))
                right_wm_overlap = np.sum(fornix_slice & (aseg_cc == 41))
                corrected_pred[x][fornix_slice] = 2 + (left_wm_overlap > right_wm_overlap) * 39  # Left WM / Right WM


            vent_slice = all_ventricle_mask
            potential_fill = np.asarray([False])
            if cc_slice.any() and vent_slice.any():
                # Create binary masks for this slice
                cc_binary = cc_slice.astype(bool)
                vent_binary = vent_slice.astype(bool)

                # Dilate both masks slightly to find potential connection points
                max_gap_vox = int(np.ceil(voxel_size[1] * close_gap_size_mm))
                cc_dilated = ndimage.binary_dilation(cc_binary, iterations=max_gap_vox)
                vent_dilated = ndimage.binary_dilation(vent_binary, iterations=max_gap_vox)

                # Find voxels that are adjacent to both CC and ventricle but not part of either
                potential_fill = (cc_dilated & vent_dilated) & ~(cc_binary | vent_binary)

            # Only fill small gaps between CC and ventricle in inferior-superior direction
            if not potential_fill.any():
                for z in range(potential_fill.shape[1]):
                    potential_fill_line = potential_fill[:, z]
                    labeled_gaps, num_gaps = ndimage.label(potential_fill_line)
                    cc_line = cc_binary[:, z]
                    vent_line = vent_binary[:, z]

                    for gap_label in range(1, num_gaps + 1):
                        gap_mask = labeled_gaps == gap_label

                        # check that CC and ventricle are connected to the gap_mask
                        dilated_gap_mask = ndimage.binary_dilation(gap_mask, iterations=1)
                        if not np.any(cc_line & dilated_gap_mask):
                            continue
                        if not np.any(vent_line & dilated_gap_mask):
                            continue

                        vent_label_location = np.where(vent_line & dilated_gap_mask)[0]
                        vent_label = corrected_pred[x, vent_label_location, z]

                        if np.sum(gap_mask) > max_gap_vox:
                            continue

                        corrected_pred[x, :, z][gap_mask  & (corrected_pred[x, :, z] == 0)] = vent_label

                # Process gaps in z-direction (within each y-row)
                for y in range(potential_fill.shape[0]):
                    potential_fill_line = potential_fill[y, :]
                    labeled_gaps, num_gaps = ndimage.label(potential_fill_line)
                    cc_line = cc_binary[y, :]
                    vent_line = vent_binary[y, :]

                    for gap_label in range(1, num_gaps + 1):
                        gap_mask = labeled_gaps == gap_label

                        # check that CC and ventricle are connected to the gap_mask
                        dilated_gap_mask = ndimage.binary_dilation(gap_mask, iterations=1)
                        if not np.any(cc_line & dilated_gap_mask):
                            continue
                        if not np.any(vent_line & dilated_gap_mask):
                            continue

                        vent_label_location = np.where(vent_line & dilated_gap_mask)[0]
                        if len(vent_label_location) > 0:
                            vent_label = corrected_pred[x, y, vent_label_location[0]]  # Take first match

                            if np.sum(gap_mask) > max_gap_vox:
                                continue

                            corrected_pred[x, y, :][gap_mask  & (corrected_pred[x, y, :] == 0)] = vent_label


    return corrected_pred


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = argument_parse()

    logger.info(f"Reading inputs: {options.input_cc} {options.input_pred}...")
    cc_seg_image = cast(nib.analyze.SpatialImage, nib.load(options.input_cc))
    cc_seg_data = np.asanyarray(cc_seg_image.dataobj)
    aseg_image = cast(nib.analyze.SpatialImage, nib.load(options.input_pred))
    aseg_data = np.asanyarray(aseg_image.dataobj)

    cc_conformed = is_conform(cc_seg_image, vox_size=None, img_size=None, verbose=False)
    pred_conformed = is_conform(aseg_image, vox_size=None, img_size=None, dtype=np.integer, verbose=False)
    if not cc_conformed:
        sys.exit("Error: CC input image is not conformed (LIA orientation, uint8 dtype). \
                 Please conform the image using the conform.py script.")
    if not pred_conformed:
        sys.exit("Error: Prediction input image is not conformed (LIA orientation, integer dtype). \
                  Please conform the image using the conform.py script.")
    if not np.allclose(cc_conformed, pred_conformed):
        sys.exit("Error: The affine matrices of the aseg and the corpus callosum images are not the same.")

    # Paint CC into prediction
    pred_with_cc = paint_in_cc(aseg_data, cc_seg_data)

    # Apply WM and ventricle corrections
    logger.info("Applying white matter and ventricle corrections...")
    fornix_mask = cc_seg_data == FORNIX_LABEL
    voxel_size = tuple(aseg_image.header.get_zooms())
    pred_corrected = correct_wm_ventricles(aseg_data, fornix_mask, voxel_size)

    print(f"Writing segmentation with corpus callosum to: {options.output}")
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

    # Count initial labels
    initial_cc = np.sum(mask_in_array(aseg_data, SUBSEGMENT_LABELS))
    initial_fornix = np.sum(aseg_data == FORNIX_LABEL)
    initial_wm = np.sum((aseg_data == 2) | (aseg_data == 41))
    print(f"Initial segmentation: CC={initial_cc}, Fornix={initial_fornix}, WM={initial_wm}")

    after_paint_cc = np.sum(mask_in_array(pred_with_cc, SUBSEGMENT_LABELS))
    print(f"After painting CC: {after_paint_cc} CC voxels added")

    # Count final labels
    final_cc = np.sum(mask_in_array(pred_corrected, SUBSEGMENT_LABELS))
    final_fornix = np.sum(pred_corrected == FORNIX_LABEL)
    final_wm = np.sum((pred_corrected == 2) | (pred_corrected == 41))
    final_ventricles = np.sum((pred_corrected == 4) | (pred_corrected == 43))

    logger.info(f"Final segmentation: CC={final_cc}, Fornix={final_fornix}, WM={final_wm}, Ventricles={final_ventricles}")
    logger.info(f"Changes: CC +{final_cc-initial_cc}, Fornix {final_fornix-initial_fornix}, WM {final_wm-initial_wm}")

    if rta_fut is not None:
        _ = rta_fut.result()

    sys.exit(0)

