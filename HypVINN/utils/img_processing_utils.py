# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

from pathlib import Path

import numpy as np
from numpy import typing as npt
import nibabel as nib
from skimage.measure import label
from scipy import ndimage

import FastSurferCNN.utils.logging as logging
from HypVINN.data_loader.data_utils import hypo_map_subseg_2_fsseg

LOGGER = logging.get_logger(__name__)


def img2axcodes(img):
    return nib.aff2axcodes(img.affine)


def save_segmentation(
        prediction: np.ndarray,
        orig_path: Path,
        ras_affine: npt.NDArray[float],
        ras_header,
        subject_dir: Path,
        seg_file: Path,
        save_mask: bool = False,
) -> float:
    from time import time
    starttime = time()
    from HypVINN.data_loader.data_utils import reorient_img
    from HypVINN.config.hypvinn_files import HYPVINN_MASK_NAME, HYPVINN_SEG_NAME

    pred_arr, labels_cc = get_clean_labels(np.array(prediction, dtype=np.uint8))
    # Mapped HypVINN labelst to FreeSurfer Hypvinn Labels
    pred_arr = hypo_map_subseg_2_fsseg(pred_arr)
    orig_img = nib.load(orig_path)
    LOGGER.info(f"Orig data orientation : {img2axcodes(orig_img)}")

    if save_mask:
        mask_img = nib.Nifti1Image(labels_cc, affine=ras_affine, header=ras_header)
        LOGGER.info(f"HypoVINN Mask orientation: {img2axcodes(mask_img)}")
        mask_img = reorient_img(mask_img, orig_img)
        LOGGER.info(
            f"HypoVINN Mask after re-orientation: {img2axcodes(mask_img)}"
        )
        nib.save(mask_img, subject_dir / "mri" / HYPVINN_MASK_NAME)

    pred_img = nib.Nifti1Image(pred_arr, affine=ras_affine, header=ras_header)
    LOGGER.info(f"HypoVINN Prediction orientation: {img2axcodes(pred_img)}")
    pred_img = reorient_img(pred_img, orig_img)
    LOGGER.info(
        f"HypoVINN Prediction after re-orientation: {img2axcodes(pred_img)}"
    )
    pred_img.set_data_dtype(np.int16)  # Maximum value 939
    nib.save(pred_img, subject_dir / seg_file)
    return time() - starttime


def save_logits(
        logits: npt.NDArray[float],
        orig_path: Path,
        ras_affine: npt.NDArray[float],
        ras_header,
        save_dir: Path,
        mode: str,
) -> Path:
    from HypVINN.data_loader.data_utils import reorient_img
    orig_img = nib.load(orig_path)
    LOGGER.info(f"Orig data orientation: {img2axcodes(orig_img)}")
    nifti_img = nib.Nifti1Image(
        logits.astype(np.float32),
        affine=ras_affine,
        header=ras_header,
    )
    LOGGER.info(f"HypoVINN logits orientation: {img2axcodes(nifti_img)}")
    nifti_img = reorient_img(nifti_img, orig_img)
    LOGGER.info(
        f"HypoVINN logits after re-orientation: {img2axcodes(nifti_img)}"
    )
    nifti_img.set_data_dtype(np.float32)
    save_as = save_dir / f"HypVINN_logits_{mode}.nii.gz"
    nib.save(nifti_img, save_as)
    return save_as


def get_clean_mask(segmentation, optic=False):
    savemask = False

    # Remove not connected components
    if optic:
        iterations = 7
        # Remove not connected from optics components
        copy_segmentation = np.zeros_like(segmentation)
        copy_segmentation[segmentation == 1] = 1
        copy_segmentation[segmentation == 2] = 2
        copy_segmentation[segmentation == 4] = 4
        copy_segmentation[segmentation == 5] = 5
    else:
        iterations = 5
        copy_segmentation = segmentation.copy()
        # remove optic structures
        copy_segmentation[segmentation == 1] = 0
        copy_segmentation[segmentation == 2] = 0
        copy_segmentation[segmentation == 4] = 0
        copy_segmentation[segmentation == 5] = 0

    struct1 = ndimage.generate_binary_structure(3, 3)
    mask = ndimage.binary_dilation(
        copy_segmentation,
        structure=struct1,
        iterations=iterations,
    ).astype(np.uint8)
    labels_cc = label(mask, connectivity=3, background=0)
    bincount = np.bincount(labels_cc.flat)

    if len(bincount) > 2:
        if optic:
            LOGGER.info("Check Optic Labels")
        else:
            LOGGER.info("Check Hypothalamus Labels")
        savemask = True

    background = np.argmax(bincount)
    bincount[background] = -1
    largest_cc = labels_cc == np.argmax(bincount)
    clean_seg = copy_segmentation * largest_cc

    # remove globus pallidus
    clean_seg[clean_seg == 13] = 0
    clean_seg[clean_seg == 20] = 0

    return clean_seg, labels_cc, savemask


def get_clean_labels(segmentation):
    """
    Function to find the largest connected component of the segmentation.

    Parameters
    ----------
    segmentation: np.ndarray
        The segmentation mask.
    """

    # Mask largest CC without optic labels
    clean_seg, labels_cc, savemask = get_clean_mask(segmentation)
    # Mask largest CC from optic labels
    optic_clean_seg, optic_labels_cc, optic_savemask = get_clean_mask(segmentation, optic=True)

    # clean segmentation from both largest_cc
    clean_seg = clean_seg + optic_clean_seg

    # mask from both largest_cc
    optic_mask = optic_labels_cc > 0
    other_mask = labels_cc > 0
    # multiplication times one to change from boolean
    non_intersect = (optic_mask * 1 - other_mask * 1) * optic_mask

    optic_labels_cc += np.max(np.unique(labels_cc))
    labels_cc = labels_cc + optic_labels_cc * non_intersect

    # remove small group of voxels less than 3
    small_mask = clean_seg > 0
    labels_small = label(small_mask, connectivity=3, background=0)
    bincount_small = np.bincount(labels_small.flat)
    idx = np.where(bincount_small <= 3)
    if idx[0].any():
        for i in idx[0]:
            small_mask[labels_small == i] = False

    clean_seg = clean_seg * small_mask

    return clean_seg, labels_cc
