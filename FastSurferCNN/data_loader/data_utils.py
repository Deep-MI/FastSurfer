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
import sys

import numpy as np
import torch
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion, binary_closing, filters, uniform_filter, generate_binary_structure
import scipy.ndimage.morphology as morphology
import nibabel as nib
import pandas as pd

from FastSurferCNN.utils import logging
from FastSurferCNN.data_loader.conform import is_conform, conform, check_affine_in_nifti


##
# Global Vars
##
SUPPORTED_OUTPUT_FILE_FORMATS = ['mgz', 'nii', 'nii.gz']
LOGGER = logging.getLogger(__name__)

##
# Helper Functions
##


# Conform an MRI brain image to UCHAR, RAS orientation, and 1mm or minimal isotropic voxels
def load_and_conform_image(img_filename, interpol=1, logger=LOGGER, conform_min = False):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm or minimum isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :param logger logger: Logger to write output to (default = STDOUT)
    :param: boolean: conform_min: conform image to minimal voxel size (for high-res) (Default = False)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = nib.load(img_filename)
    # is_conform and conform accept numeric values and the string 'min' instead of the bool value
    _conform_vox_size = 'min' if conform_min else 1.
    if not is_conform(orig, conform_vox_size=_conform_vox_size):

        logger.info('Conforming image to UCHAR, RAS orientation, and minimum isotropic voxels')

        if len(orig.shape) > 3 and orig.shape[3] != 1:
            sys.exit('ERROR: Multiple input frames (' + format(orig.shape[3]) + ') not supported!')

        # Check affine if image is nifti image
        if img_filename[-7:] == ".nii.gz" or img_filename[-4:] == ".nii":
            if not check_affine_in_nifti(orig, logger=logger):
                sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

        # conform
        orig = conform(orig, interpol, conform_vox_size=_conform_vox_size)

    # Collect header and affine information
    header_info = orig.header
    affine_info = orig.affine
    orig = np.asanyarray(orig.dataobj)

    return header_info, affine_info, orig


# Save image routine
def save_image(header_info, affine_info, img_array, save_as, dtype=None):
    """
    Save an image (nibabel MGHImage), according to the desired output file format.
    Supported formats are defined in supported_output_file_formats.
    :param numpy.ndarray img_array: an array containing image data
    :param numpy.ndarray affine_info: image affine information
    :param nibabel.freesurfer.mghformat.MGHHeader header_info: image header information
    :param str save_as: name under which to save prediction; this determines output file format
    :param type dtype: image array type; if provided, the image object is explicitly set to match this type
    :return None: saves predictions to save_as
    """

    assert any(save_as.endswith(file_ext) for file_ext in SUPPORTED_OUTPUT_FILE_FORMATS), \
        'Output filename does not contain a supported file format (' + ', '.join(
            file_ext for file_ext in SUPPORTED_OUTPUT_FILE_FORMATS) + ')!'

    mgh_img = None
    if save_as.endswith('mgz'):
        mgh_img = nib.MGHImage(img_array, affine_info, header_info)
    elif any(save_as.endswith(file_ext) for file_ext in ['nii', 'nii.gz']):
        mgh_img = nib.nifti1.Nifti1Pair(img_array, affine_info, header_info)

    if dtype is not None:
        mgh_img.set_data_dtype(dtype)

    if any(save_as.endswith(file_ext) for file_ext in ['mgz', 'nii']):
        nib.save(mgh_img, save_as)
    elif save_as.endswith('nii.gz'):
        # For correct outputs, nii.gz files should be saved using the nifti1 sub-module's save():
        nib.nifti1.save(mgh_img, save_as)


# Transformation for mapping
def transform_axial(vol, coronal2axial=True):
    """
    Function to transform volume into Axial axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2axial: transform from coronal to axial = True (default),
                               transform from axial to coronal = False
    :return:
    """
    if coronal2axial:
        return np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])


def transform_sagittal(vol, coronal2sagittal=True):
    """
    Function to transform volume into Sagittal axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2sagittal: transform from coronal to sagittal = True (default),
                                transform from sagittal to coronal = False
    :return:
    """
    if coronal2sagittal:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])


# Thick slice generator (for eval) and blank slices filter (for training)
def get_thick_slices(img_data, slice_thickness=3):
    """
    Function to extract thick slices from the image
    (feed slice_thickness preceeding and suceeding slices to network,
    label only middle one)
    :param np.ndarray img_data: 3D MRI image read in with nibabel
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3)
    :return:
    """
    h, w, d = img_data.shape
    img_data_pad = np.expand_dims(np.pad(img_data, ((0, 0), (0, 0), (slice_thickness, slice_thickness)), mode='edge'),
                                  axis=3)
    img_data_thick = np.ndarray((h, w, d, 0), dtype=np.uint8)

    for slice_idx in range(2 * slice_thickness + 1):
        img_data_thick = np.append(img_data_thick, img_data_pad[:, :, slice_idx:d + slice_idx, :], axis=3)

    return img_data_thick


def filter_blank_slices_thick(img_vol, label_vol, weight_vol, threshold=50):
    """
    Function to filter blank slices from the volume using the label volume
    :param np.ndarray img_vol: orig image volume
    :param np.ndarray label_vol: label images (ground truth)
    :param np.ndarray weight_vol: weight corresponding to labels
    :param int threshold: threshold for number of pixels needed to keep slice (below = dropped)
    :return:
    """
    # Get indices of all slices with more than threshold labels/pixels
    select_slices = (np.sum(label_vol, axis=(0, 1)) > threshold)

    # Retain only slices with more than threshold labels/pixels
    img_vol = img_vol[:, :, select_slices, :]
    label_vol = label_vol[:, :, select_slices]
    weight_vol = weight_vol[:, :, select_slices]

    return img_vol, label_vol, weight_vol


# weight map generator
def create_weight_mask(mapped_aseg, max_weight=5, max_edge_weight=5, max_hires_weight=None, ctx_thresh=33,
                       mean_filter=False, cortex_mask=True, gradient=True):
    """
    Function to create weighted mask - with median frequency balancing and edge-weighting
    :param np.ndarray mapped_aseg: segmentation to create weight mask from
    :param int max_weight: maximal weight on median weights (cap at this value)
    :param int max_edge_weight: maximal weight on gradient weight (cap at this value)
    :param int max_hires_weight: maximal weight on hires weight (cap at this value)
    :param int ctx_thresh: label value of cortex (above = cortical parcels)
    :param bool mean_filter: flag, set to add mean_filter mask (default = False)
    :param bool cortex_mask: flag, set to create cortex weight mask (default=True)
    :param bool gradient: flag, set to create gradient mask (default = True)
    :return np.ndarray: weights
    """
    unique, counts = np.unique(mapped_aseg, return_counts=True)

    # Median Frequency Balancing
    class_wise_weights = np.median(counts) / counts
    class_wise_weights[class_wise_weights > max_weight] = max_weight
    (h, w, d) = mapped_aseg.shape

    weights_mask = np.reshape(class_wise_weights[mapped_aseg.ravel()], (h, w, d))

    # Gradient Weighting
    if gradient:
        (gx, gy, gz) = np.gradient(mapped_aseg)
        grad_weight = max_edge_weight * np.asarray(
            np.power(np.power(gx, 2) + np.power(gy, 2) + np.power(gz, 2), 0.5) > 0,
            dtype='float')

        weights_mask += grad_weight

    if max_hires_weight is not None:
        # High-res Weighting
        print("Adding hires weight mask deep sulci and WM with weight ", max_hires_weight)
        mask1 = deep_sulci_and_wm_strand_mask(mapped_aseg, structure=np.ones((3, 3, 3)), ctx_thresh=ctx_thresh)
        weights_mask += mask1 * max_hires_weight

        if cortex_mask:
            print("Adding cortex mask with weight ", max_hires_weight)
            mask2 = cortex_border_mask(mapped_aseg, structure=np.ones((3, 3, 3)), ctx_thresh=ctx_thresh)
            weights_mask += mask2 * (max_hires_weight) // 2

    if mean_filter:
        weights_mask = uniform_filter(weights_mask, size=3)

    return weights_mask


def cortex_border_mask(label, structure, ctx_thresh=33):
    """
    Function to erode the cortex of a given mri image to create
    the inner gray matter mask (outer most cortex voxels)
    :param np.ndarray label: ground truth labels.
    :param np.ndarray structure: structuring element to erode with
    :param int ctx_thresh: label value of cortex (above = cortical parcels)
    :return: np.ndarray outer GM layer
    """
    # create aseg brainmask, erode it and subtract from itself
    bm = np.clip(label, a_max=1, a_min=0)
    eroded = binary_erosion(bm, structure=structure)
    diff_im = np.logical_xor(eroded, bm)

    # only keep values associated with the cortex
    diff_im[(label <= ctx_thresh)] = 0  # > 33 (>19) = > 1002 in FS space (full (sag)),
    print("Remaining voxels cortex border: ", np.unique(diff_im, return_counts=True))
    return diff_im


def deep_sulci_and_wm_strand_mask(volume, structure, iteration=1, ctx_thresh=33):
    """
    Function to get a binary mask of deep sulci and small white matter strands
     by using binary closing (erosion and dilation)

    :param np.ndarray volume: loaded image (aseg, label space)
    :param np.ndarray structure: structuring element (e.g. np.ones((3, 3, 3)))
    :param int iteration: number of times mask should be dilated + eroded (default=1)
    :param int ctx_thresh: label value of cortex (above = cortical parcels)
    :return np.ndarray: sulcus + wm mask
    """
    # Binarize label image (cortex = 1, everything else = 0)
    empty_im = np.zeros(shape=volume.shape)
    empty_im[volume > ctx_thresh] = 1  # > 33 (>19) = >1002 in FS LUT (full (sag))

    # Erode the image
    eroded = binary_closing(empty_im, iterations=iteration, structure=structure)

    # Get difference between eroded and original image
    diff_image = np.logical_xor(empty_im, eroded)
    print("Remaining voxels sulci/wm strand: ", np.unique(diff_image, return_counts=True))
    return diff_image


# Label mapping functions (to aparc (eval) and to label (train))
def read_classes_from_lut(lut_file):
    """
    Function to read in FreeSurfer-like LUT table
    :param str lut_file: path and name of FreeSurfer-style LUT file with classes of interest
                         Example entry:
                         ID LabelName  R   G   B   A
                         0   Unknown   0   0   0   0
                         1   Left-Cerebral-Exterior 70  130 180 0
    :return pd.Dataframe: DataFrame with ids present, name of ids, color for plotting
    """
    # Read in file
    separator = {"tsv": "\t", "csv": ",", "txt": " "}
    return pd.read_csv(lut_file, sep=separator[lut_file[-3:]])


def map_label2aparc_aseg(mapped_aseg, labels):
    """
    Function to perform look-up table mapping from sequential label space to LUT space
    :param torch.Tensor mapped_aseg: label space segmentation (aparc.DKTatlas + aseg)
    :param np.ndarray labels: list of labels defining LUT space
    :return:
    """
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    labels = labels.to(mapped_aseg.device)
    return labels[mapped_aseg]


def clean_cortex_labels(aparc):
    """
    Function to clean up aparc segmentations:
        Map undetermined and optic chiasma to BKG
        Map Hypointensity classes to one
        Vessel to WM
        5th Ventricle to CSF
        Remaining cortical labels to BKG
    :param np.array aparc:
    :return np.array: cleaned aparc
    """
    aparc[aparc == 80] = 77  # Hypointensities Class
    aparc[aparc == 85] = 0  # Optic Chiasma to BKG
    aparc[aparc == 62] = 41  # Right Vessel to Right WM
    aparc[aparc == 30] = 2  # Left Vessel to Left WM
    aparc[aparc == 72] = 24  # 5th Ventricle to CSF
    aparc[aparc == 29] = 0  # left-undetermined to 0
    aparc[aparc == 61] = 0  # right-undetermined to 0

    aparc[aparc == 3] = 0  # Map Remaining Cortical labels to background
    aparc[aparc == 42] = 0
    return aparc


def fill_unknown_labels_per_hemi(gt, unknown_label, cortex_stop):
    """
    Function to replace label 1000 (lh unknown) and 2000 (rh unknown) with closest class for each voxel.
    :param np.ndarray gt: ground truth segmentation with class unknown
    :param int unknown_label: class label for unknown (lh: 1000, rh: 2000)
    :param int cortex_stop: class label at which cortical labels of this hemi stop (lh: 2000, rh: 3000)
    :return:
    """
    # Define shape of image and dilation element
    h, w, d = gt.shape
    struct1 = generate_binary_structure(3, 2)

    # Get indices of unknown labels, dilate them to get closest sorrounding parcels
    unknown = gt == unknown_label
    unknown = (morphology.binary_dilation(unknown, struct1) ^ unknown)
    list_parcels = np.unique(gt[unknown])

    # Mask all subcortical structues (fill unknown with closest cortical parcels only)
    mask = (list_parcels > unknown_label) & (list_parcels < cortex_stop)
    list_parcels = list_parcels[mask]

    # For each closest parcel, blur label with gaussian filter (spread), append resulting blurred images
    blur_vals = np.ndarray((h, w, d, 0), dtype=np.float)
    for idx in range(len(list_parcels)):
        aseg_blur = filters.gaussian_filter(1000 * np.asarray(gt == list_parcels[idx], dtype=np.float), sigma=5)
        blur_vals = np.append(blur_vals, np.expand_dims(aseg_blur, axis=3), axis=3)

    # Get for each position parcel with maximum value after blurring (= closest parcel)
    unknown = np.argmax(blur_vals, axis=3)
    unknown = np.reshape(list_parcels[unknown.ravel()], (h, w, d))

    # Assign the determined closest parcel to the unknown class (case-by-case basis)
    mask = gt == unknown_label
    gt[mask] = unknown[mask]

    return gt


def fuse_cortex_labels(aparc):
    """
    Fuse cortical parcels on left/right hemisphere (reduce aparc classes)
    :param np.ndarray aparc: anatomical segmentation with cortical parcels
    :return: anatomical segmentation with reduced number of cortical parcels
    """
    aparc_temp = aparc.copy()

    # Map undetermined classes
    aparc = clean_cortex_labels(aparc)

    # Fill label unknown
    if np.any(aparc == 1000):
        aparc = fill_unknown_labels_per_hemi(aparc, 1000, 2000)
    if np.any(aparc == 2000):
        aparc = fill_unknown_labels_per_hemi(aparc, 2000, 3000)

    # De-lateralize parcels
    cortical_label_mask = (aparc >= 2000) & (aparc <= 2999)
    aparc[cortical_label_mask] = aparc[cortical_label_mask] - 1000

    # Re-lateralize Cortical parcels in close proximity
    aparc[aparc_temp == 2014] = 2014
    aparc[aparc_temp == 2028] = 2028
    aparc[aparc_temp == 2012] = 2012
    aparc[aparc_temp == 2016] = 2016
    aparc[aparc_temp == 2002] = 2002
    aparc[aparc_temp == 2023] = 2023
    aparc[aparc_temp == 2017] = 2017
    aparc[aparc_temp == 2024] = 2024
    aparc[aparc_temp == 2010] = 2010
    aparc[aparc_temp == 2013] = 2013
    aparc[aparc_temp == 2025] = 2025
    aparc[aparc_temp == 2022] = 2022
    aparc[aparc_temp == 2021] = 2021
    aparc[aparc_temp == 2005] = 2005

    return aparc


def split_cortex_labels(aparc):
    """
    Splot cortex labels to completely de-lateralize structures
    :param np.ndarray aparc: anatomical segmentation and parcellation from network
    :return np.ndarray: re-lateralized aparc
    """
    # Post processing - Splitting classes
    # Quick Fix for 2026 vs 1026; 2029 vs. 1029; 2025 vs. 1025
    rh_wm = get_largest_cc(aparc == 41)
    lh_wm = get_largest_cc(aparc == 2)
    rh_wm = regionprops(label(rh_wm, background=0))
    lh_wm = regionprops(label(lh_wm, background=0))
    centroid_rh = np.asarray(rh_wm[0].centroid)
    centroid_lh = np.asarray(lh_wm[0].centroid)

    labels_list = np.array([1003, 1006, 1007, 1008, 1009, 1011,
                            1015, 1018, 1019, 1020, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035])

    for label_current in labels_list:

        label_img = label(aparc == label_current, connectivity=3, background=0)

        for region in regionprops(label_img):

            if region.label != 0:  # To avoid background

                if np.linalg.norm(np.asarray(region.centroid) - centroid_rh) < np.linalg.norm(
                        np.asarray(region.centroid) - centroid_lh):
                    mask = label_img == region.label
                    aparc[mask] = label_current + 1000

    # Quick Fixes for overlapping classes
    aseg_lh = filters.gaussian_filter(1000 * np.asarray(aparc == 2, dtype=np.float), sigma=3)
    aseg_rh = filters.gaussian_filter(1000 * np.asarray(aparc == 41, dtype=np.float), sigma=3)

    lh_rh_split = np.argmax(np.concatenate((np.expand_dims(aseg_lh, axis=3), np.expand_dims(aseg_rh, axis=3)), axis=3),
                            axis=3)

    # Problematic classes: 1026, 1011, 1029, 1019
    for prob_class_lh in [1011, 1019, 1026, 1029]:
        prob_class_rh = prob_class_lh + 1000
        mask_prob_class = (aparc == prob_class_lh) | (aparc == prob_class_rh)
        mask_lh = np.logical_and(mask_prob_class, lh_rh_split == 0)
        mask_rh = np.logical_and(mask_prob_class, lh_rh_split == 1)

        aparc[mask_lh] = prob_class_lh
        aparc[mask_rh] = prob_class_rh

    return aparc


def unify_lateralized_labels(lut, combi=("Left-", "Right-")):
    """
    Function to generate lookup dictionary of left-right labels
    :param str or pd.DataFrame lut: either lut-file string to load or pandas dataframe
                                    Example entry:
                                    ID LabelName  R   G   B   A
                                    0   Unknown   0   0   0   0
                                    1   Left-Cerebral-Exterior 70  130 180 0
    :param list(str) combi: Prefix or labelnames to combine. Default: Left- and Right-
    :return dict: dictionary mapping between left and right hemispheres
    """
    if isinstance(lut, str):
        lut = read_classes_from_lut(lut)
    left = lut[["ID", "LabelName"]][lut["LabelName"].str.startswith(combi[0])]
    right = lut[["ID", "LabelName"]][lut["LabelName"].str.startswith(combi[1])]
    left["LabelName"] = left["LabelName"].str.removeprefix(combi[0])
    right["LabelName"] = right["LabelName"].str.removeprefix(combi[1])
    mapp = left.merge(right, on="LabelName")
    return pd.Series(mapp.ID_y.values, index=mapp.ID_x).to_dict()


def get_labels_from_lut(lut, label_extract=("Left-", "ctx-rh")):
    """
    Function to extract
    :param str of pd.DataFrame lut: FreeSurfer like LookUp Table (either path to it
                                    or already loaded as pandas DataFrame.
                                    Example entry:
                                    ID LabelName  R   G   B   A
                                    0   Unknown   0   0   0   0
                                    1   Left-Cerebral-Exterior 70  130 180 0
    :param tuple(str) label_extract: suffix of label names to mask for sagittal labels
                                     Default: "Left-" and "ctx-rh"
    :return np.ndarray: full label list
    :return np.ndarray: sagittal label list
    """
    if isinstance(lut, str):
        lut = read_classes_from_lut(lut)
    mask = lut["LabelName"].str.startswith(label_extract)
    return lut["ID"].values, lut["ID"][~mask].values


def map_aparc_aseg2label(aseg, labels, labels_sag, sagittal_lut_dict, aseg_nocc=None, processing="aparc"):
    """
    Function to perform look-up table mapping of aparc.DKTatlas+aseg.mgz data to label space
    :param np.ndarray aseg: ground truth aparc+aseg
    :param np.ndarray labels: labels to use (extracted from LUT with get_labels_from_lut)
    :param np.ndarray labels_sag: sagittal labels to use (extracted from LUT with
                                  get_labels_from_lut)
    :param dict(int) sagittal_lut_dict: left-right label mapping (can be extracted with
                                        unify_lateralized_labels from LUT)
    :param str processing: should be set to "aparc" or "aseg" for additional mappings (hard-coded)
    :param None/np.ndarray aseg_nocc: ground truth aseg without corpus callosum segmentation
    :return:
    """
    # If corpus callosum is not removed yet, do it now
    if aseg_nocc is not None:
        cc_mask = (aseg >= 251) & (aseg <= 255)
        aseg[cc_mask] = aseg_nocc[cc_mask]

    if processing == "aparc":
        LOGGER.info("APARC PROCESSING")
        aseg = fuse_cortex_labels(aseg)

    elif processing == "aseg":
        LOGGER.info("ASEG PROCESSING")
        aseg[aseg == 1000] = 3  # Map unknown to cortex
        aseg[aseg == 2000] = 42
        aseg[aseg == 80] = 77  # Hypointensities Class
        aseg[aseg == 85] = 0  # Optic Chiasma to BKG
        aseg[aseg == 62] = 41  # Right Vessel to Right WM
        aseg[aseg == 30] = 2  # Left Vessel to Left WM
        aseg[aseg == 72] = 24  # 5th Ventricle to CSF

        assert not np.any(251 <= aseg), "Error: CC classes (251-255) still exist in aseg {}".format(np.unique(aseg))
        assert np.any(aseg == 3) and np.any(aseg == 42), "Error: no cortical marker detected {}".format(np.unique(aseg))

    assert set(labels).issuperset(
        np.unique(aseg)), "Error: segmentation image contains classes not listed in the labels: \n{}\n{}".format(
        np.unique(aseg), labels)

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels) + 1, dtype='int')
    for idx, value in enumerate(labels):
        lut_aseg[value] = idx

    # Remap Label Classes - Perform LUT Mapping - Coronal, Axial
    mapped_aseg = lut_aseg.ravel()[aseg.ravel()]
    mapped_aseg = mapped_aseg.reshape((h, w, d))

    if processing == "aparc":
        cortical_label_mask = (aseg >= 2000) & (aseg <= 2999)
        aseg[cortical_label_mask] = aseg[cortical_label_mask] - 1000

    # For sagittal, all Left hemispheres will be mapped to right, ctx the otherway round
    # If you use your own LUT, make sure all per-hemi labels have the corresponding prefix
    # Map Sagittal Labels
    for left, right in sagittal_lut_dict.items():
        aseg[aseg == left] = right

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels_sag) + 1, dtype='int')
    for idx, value in enumerate(labels_sag):
        lut_aseg[value] = idx

    # Remap Label Classes - Perform LUT Mapping - Coronal, Axial
    mapped_aseg_sag = lut_aseg.ravel()[aseg.ravel()]
    mapped_aseg_sag = mapped_aseg_sag.reshape((h, w, d))

    return mapped_aseg, mapped_aseg_sag


def sagittal_coronal_remap_lookup(x):
    """
    Dictionary mapping to convert left labels to corresponding right labels for aseg
    :param int x: label to look up
    :return:
    """
    return {
        2: 41,
        3: 42,
        4: 43,
        5: 44,
        7: 46,
        8: 47,
        10: 49,
        11: 50,
        12: 51,
        13: 52,
        17: 53,
        18: 54,
        26: 58,
        28: 60,
        31: 63,
    }[x]


def infer_mapping_from_lut(num_classes_full, lut):
    labels, labels_sag = unify_lateralized_labels(lut)
    idx_list = np.ndarray(shape=(num_classes_full,), dtype=np.int16)
    for idx in range(len(labels)):
        idx_in_sag = np.where(labels_sag == labels[idx])[0]
        if idx_in_sag.size == 0:  # Empty not subcortical
            idx_in_sag = np.where(labels_sag == (labels[idx] - 1000))[0]

        if idx_in_sag.size == 0:
            current_label_sag = sagittal_coronal_remap_lookup(labels[idx])
            idx_in_sag = np.where(labels_sag == current_label_sag)[0]

        idx_list[idx] = idx_in_sag
    return idx_list


def map_prediction_sagittal2full(prediction_sag, num_classes=51, lut=None):
    """
    Function to remap the prediction on the sagittal network to full label space used by coronal and axial networks
    (full aparc.DKTatlas+aseg.mgz)
    :param prediction_sag: sagittal prediction (labels)
    :param int num_classes: number of SAGGITAL classes (96 for full classes, 51 for hemi split, 21 for aseg)
    :param str/None lut: look-up table listing class labels
    :return: Remapped prediction
    """
    if num_classes == 96:
        idx_list = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 14, 15, 4, 16,
                               17, 18, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 20, 21, 22,
                               23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], dtype=np.int16)

    elif num_classes == 51:
        idx_list = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 14, 15, 4, 16,
                               17, 18, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 20, 22, 27,
                               29, 30, 31, 33, 34, 38, 39, 40, 41, 42, 45], dtype=np.int16)

    elif num_classes == 21:
        idx_list = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 15, 16, 4,
                               17, 18, 19, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20], dtype=np.int16)
    elif num_classes == 1:
        idx_list = np.asarray([0, 1, 1])
    else:
        assert lut is not None, 'lut is not defined!'
        idx_list = infer_mapping_from_lut(num_classes, lut)
    prediction_full = prediction_sag[:, idx_list, :, :]
    return prediction_full


# Clean up and class separation
def bbox_3d(img):
    """
    Function to extract the three-dimensional bounding box coordinates.
    :param np.ndarray img: mri image
    :return:
    """

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def get_largest_cc(segmentation):
    """
    Function to find largest connected component of segmentation.
    :param np.ndarray segmentation: segmentation
    :return:
    """
    labels = label(segmentation, connectivity=3, background=0)

    bincount = np.bincount(labels.flat)
    background = np.argmax(bincount)
    bincount[background] = -1

    largest_cc = labels == np.argmax(bincount)

    return largest_cc
