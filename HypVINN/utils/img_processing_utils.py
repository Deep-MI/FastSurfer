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


def img2axcodes(img: nib.Nifti1Image) -> tuple:
    """
    Convert the affine matrix of an image to axis codes.

    This function takes an image as input and returns the axis codes corresponding to the affine matrix of the image.

    Parameters
    ----------
    img : nibabel image object
        The input image.

    Returns
    -------
    tuple
        The axis codes corresponding to the affine matrix of the image.
    """
    return nib.aff2axcodes(img.affine)


def save_segmentation(
        prediction: np.ndarray,
        orig_path: Path,
        ras_affine: npt.NDArray[float],
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
    from HypVINN.data_loader.data_utils import reorient_img

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
        nib.save(mask_img, subject_dir / "mri" / mask_file)

    pred_img = nib.Nifti1Image(pred_arr, affine=ras_affine, header=ras_header)
    LOGGER.info(f"HypoVINN Prediction orientation: {img2axcodes(pred_img)}")
    pred_img = reorient_img(pred_img, orig_img)
    LOGGER.info(
        f"HypoVINN Prediction after re-orientation: {img2axcodes(pred_img)}"
    )
    pred_img.set_data_dtype(np.int16)  # Maximum value 939
    nib.save(pred_img, subject_dir / "mri" / seg_file)
    return time() - starttime


def save_logits(
        logits: npt.NDArray[float],
        orig_path: Path,
        ras_affine: npt.NDArray[float],
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
    logits : npt.NDArray[float]
        The raw model outputs.
    orig_path : Path
        The path to the original image.
    ras_affine : npt.NDArray[float]
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
    segmentation: np.ndarray
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
