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
from typing import cast

import nibabel as nib
import numpy as np
from nibabel import Nifti1Image
from nibabel.orientations import aff2axcodes
from scipy import ndimage
from skimage.measure import label

import FastSurferCNN.utils.logging as logging
from FastSurferCNN.data_loader.conform import Reorientation, does_vox2vox_rot_require_interpolation
from FastSurferCNN.utils import AffineMatrix4x4, Image4d, nibabelHeader, nibabelImage
from HypVINN.data_loader.data_utils import hypo_map_subseg_2_fsseg

LOGGER = logging.get_logger(__name__)


def save_segmentation(
        prediction: np.ndarray,
        orig_path: Path,
        ras_affine: AffineMatrix4x4,
        ras_header: nib.nifti1.Nifti1Header | nib.nifti2.Nifti2Header | nib.freesurfer.mghformat.MGHHeader,
        subject_dir: Path,
        seg_file: str,
        mask_file: str,
        save_mask: bool = False,
) -> float:
    """
    Save the segmentation results.

    This function takes the prediction results, cleans the labels, maps them to FreeSurfer Hypvinn Labels, and saves
    the results. It also reorients the mask and prediction images to match the original image's orientation.

    Parameters
    ----------
    prediction : np.ndarray
        The prediction results.
    orig_path : Path
        The path to the original image.
    ras_affine : npt.NDArray[float]
        The affine transformation of the RAS orientation.
    ras_header : nibabel header object
        The header of the RAS orientation.
    subject_dir : Path
        The directory where the subject's data is stored.
    seg_file : Path
        The file where the segmentation will be saved (relative to subject_dir/mri).
    mask_file : str
        The file where the mask will be saved (relative to subject_dir/mri).
    save_mask : bool, default=False
        Whether to save the mask or not. Default is False.

    Returns
    -------
    float
        The time taken to save the segmentation.

    """
    from time import time
    starttime = time()

    pred_arr, labels_cc = get_clean_labels(np.array(prediction, dtype=np.uint8))
    # Mapped HypVINN labels to FreeSurfer Hypvinn Labels
    pred_arr = hypo_map_subseg_2_fsseg(pred_arr)
    orig_img = cast(nibabelImage, nib.load(orig_path))

    reorient = Reorientation.from_target_affine(ras_affine, orig_img.affine, labels_cc.shape)
    LOGGER.info(f"Orig data orientation : {aff2axcodes(orig_img.affine)}")

    for data, name in ((pred_arr, "segmentation"), (labels_cc, "mask")):
        if not np.allclose(reorient.reorder_axes(np.asarray(data.shape)), orig_img.shape):
            raise RuntimeError(f"Hypothalamus {name} and orig image have different shapes!")

    if does_vox2vox_rot_require_interpolation(reorient.vox2vox):
        LOGGER.warning("Hypothalamus mask and segmentation reorientation requires lossy interpolation.")

    if save_mask:
        mask_header: nibabelHeader = Nifti1Image.header_class.from_header(orig_img.header)
        mask_header.set_data_dtype(np.uint8)
        mask_img = nib.Nifti1Image(
            reorient(labels_cc.astype(np.uint8), order=0),
            affine=orig_img.affine,
            header=mask_header,
        )
        mask_img.set_data_dtype(np.float32)
        LOGGER.info(f"HypVINN Mask after re-orientation: {aff2axcodes(mask_img.affine)}")
        nib.save(mask_img, subject_dir / "mri" / mask_file)

    pred_header: nibabelHeader = Nifti1Image.header_class.from_header(orig_img.header)
    pred_header.set_data_dtype(np.uint8)
    pred_img = nib.Nifti1Image(
        reorient(pred_arr.astype(np.int16), order=0),
        affine=orig_img.affine,
        header=pred_header,
    )
    LOGGER.info(f"HypVINN Prediction after re-orientation: {aff2axcodes(pred_img.affine)}")
    pred_img.set_data_dtype(np.int16)  # Maximum value 984
    nib.save(pred_img, subject_dir / "mri" / seg_file)
    return time() - starttime


def save_logits(
        logits: Image4d,
        orig_path: Path,
        ras_affine: AffineMatrix4x4,
        ras_header: nib.nifti1.Nifti1Header | nib.nifti2.Nifti2Header | nib.freesurfer.mghformat.MGHHeader,
        save_dir: Path,
        mode: str,
) -> Path:
    """
    Save the logits (raw model outputs) as a NIfTI image.

    This function takes the logits, reorients the image to match the original image's orientation, and saves the
    results.

    Parameters
    ----------
    logits : np.ndarray
        The raw model outputs.
    orig_path : Path
        The path to the original image.
    ras_affine : AffineMatrix4x4
        The affine transformation of the RAS orientation.
    ras_header : nib.nifti1.Nifti1Header
        The header of the RAS orientation.
    save_dir : Path
        The directory where the logits will be saved.
    mode : str
        The mode of operation.

    Returns
    -------
    save_as: Path
        The path where the logits were saved.

    """
    orig_img = cast(nibabelImage, nib.load(orig_path))
    LOGGER.info(f"Orig data orientation: {aff2axcodes(orig_img.affine)}")
    header: nibabelHeader = Nifti1Image.header_class.from_header(orig_img.header)
    header.set_data_type(np.float32)
    reorient = Reorientation.from_target_affine(ras_affine, orig_img.affine, logits.shape)
    nifti_img = nib.Nifti1Image(
        reorient(logits.astype(np.float32)),
        affine=orig_img.affine,
        header=header,
    )
    LOGGER.info(f"HypVINN logits after re-orientation: {aff2axcodes(nifti_img.affine)}")
    nifti_img.set_data_dtype(np.float32)
    save_as = save_dir / f"HypVINN_logits_{mode}.nii.gz"
    nib.save(nifti_img, save_as)
    return save_as


def get_clean_mask(segmentation: np.ndarray, optic=False) \
        -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Get a clean mask by removing non-connected components from a dilated mask.

    This function takes a segmentation mask and an optional boolean flag indicating whether to consider optic labels.
    It removes not connected components from the segmentation mask and returns the cleaned segmentation mask, the
    labels of the connected components, and a flag indicating whether to save the mask.

    Parameters
    ----------
    segmentation : np.ndarray
        The input segmentation mask.
    optic : bool, default=False
        A flag indicating whether to consider optic labels. Default is False.

    Returns
    -------
    clean_seg : np.ndarray
        The cleaned segmentation mask.
    labels_cc : np.ndarray
        The labels of the connected components in the segmentation mask.
    savemask : bool
        A flag indicating whether to save the mask.

    """
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


def get_clean_labels(segmentation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get clean labels by removing non-connected components from a dilated mask and any connected component with size
    less than 3.

    Parameters
    ----------
    segmentation : np.ndarray
        The segmentation mask.

    Returns
    -------
    clean_seg: np.ndarray
        The cleaned segmentation mask.
    labels_cc: np.ndarray
        The labels of the connected components in the segmentation mask.
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
