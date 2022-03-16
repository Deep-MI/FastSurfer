
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
import nibabel as nib
import numpy as np
import h5py
import scipy.ndimage.morphology as morphology
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import sys

from skimage.measure import label
from torch.utils.data.dataset import Dataset
from .conform import is_conform, conform, check_affine_in_nifti

supported_output_file_formats = ['mgz', 'nii', 'nii.gz']

##
# Helper Functions
##


# Conform an MRI brain image to UCHAR, RAS orientation, and 1mm isotropic voxels
def load_and_conform_image(img_filename, interpol=1, logger=None):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return: nibabel.MGHImage header_info: header information of the conformed image
    :return: np.ndarray affine_info: affine information of the conformed image
    :return: nibabel.MGHImage orig: conformed image
    """
    orig = nib.load(img_filename)

    if not is_conform(orig):

        if logger is not None:
            logger.info('Conforming image to UCHAR, RAS orientation, and 1mm isotropic voxels')
        else:
            print('Conforming image to UCHAR, RAS orientation, and 1mm isotropic voxels')

        if len(orig.shape) > 3 and orig.shape[3] != 1:
            sys.exit('ERROR: Multiple input frames (' + format(orig.shape[3]) + ') not supported!')

        # Check affine if image is nifti image
        if img_filename[-7:] == ".nii.gz" or img_filename[-4:] == ".nii":
            if not check_affine_in_nifti(orig, logger=logger):
                sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

        # conform
        orig = conform(orig, interpol)

    # Collect header and affine information
    header_info = orig.header
    affine_info = orig.affine
    orig = np.asanyarray(orig.dataobj)

    return header_info, affine_info, orig

def save_image(img_array, affine_info, header_info, save_as):
    """
    Save an image (nibabel MGHImage), according to the desired output file format.
    Supported formats are defined in supported_output_file_formats.

    :param numpy.ndarray img_array: an array containing image data
    :param numpy.ndarray affine_info: image affine information
    :param nibabel.freesurfer.mghformat.MGHHeader header_info: image header information
    :param str save_as: name under which to save prediction; this determines output file format

    :return None: saves predictions to save_as
    """

    assert any(save_as.endswith(file_ext) for file_ext in supported_output_file_formats), \
            'Output filename does not contain a supported file format (' + ', '.join(file_ext for file_ext in supported_output_file_formats) + ')!'

    mgh_img = None
    if save_as.endswith('mgz'):
        mgh_img = nib.MGHImage(img_array, affine_info, header_info)
    elif any(save_as.endswith(file_ext) for file_ext in ['nii', 'nii.gz']):
        mgh_img = nib.nifti1.Nifti1Pair(img_array, affine_info, header_info)

    if any(save_as.endswith(file_ext) for file_ext in ['mgz', 'nii']):
        nib.save(mgh_img, save_as)
    elif save_as.endswith('nii.gz'):
        ## For correct outputs, nii.gz files should be saved using the nifti1 sub-module's save():
        nib.nifti1.save(mgh_img, save_as)


# Transformation for mapping
def transform_axial(vol, coronal2axial=True):
    """
    Function to transform volume into Axial axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2axial: transform from coronal to axial = True (default),
                               transform from axial to coronal = False
    :return: np.ndarray: transformed image volume
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
    :return: np.ndarray: transformed image volume
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
    :return: np.ndarray img_data_thick: image array containing the extracted slices
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
    :return: np.ndarray img_vol: filtered orig image volume
    :return: np.ndarray label_vol: filtered label images (ground truth)
    :return: np.ndarray weight_vol: filtered weight corresponding to labels
    """
    # Get indices of all slices with more than threshold labels/pixels
    select_slices = (np.sum(label_vol, axis=(0, 1)) > threshold)

    # Retain only slices with more than threshold labels/pixels
    img_vol = img_vol[:, :, select_slices, :]
    label_vol = label_vol[:, :, select_slices]
    weight_vol = weight_vol[:, :, select_slices]

    return img_vol, label_vol, weight_vol


# weight map generator
def create_weight_mask(mapped_aseg, max_weight=5, max_edge_weight=5):
    """
    Function to create weighted mask - with median frequency balancing and edge-weighting
    :param np.ndarray mapped_aseg: label space segmentation
    :param int max_weight: an upper bound on weight values
    :param int max_edge_weight: edge-weighting factor
    :return: np.ndarray weights_mask: generated weights mask
    """
    unique, counts = np.unique(mapped_aseg, return_counts=True)

    # Median Frequency Balancing
    class_wise_weights = np.median(counts) / counts
    class_wise_weights[class_wise_weights > max_weight] = max_weight
    (h, w, d) = mapped_aseg.shape

    weights_mask = np.reshape(class_wise_weights[mapped_aseg.ravel()], (h, w, d))

    # Gradient Weighting
    (gx, gy, gz) = np.gradient(mapped_aseg)
    grad_weight = max_edge_weight * np.asarray(np.power(np.power(gx, 2) + np.power(gy, 2) + np.power(gz, 2), 0.5) > 0,
                                               dtype='float')

    weights_mask += grad_weight

    return weights_mask


# class unknown filler (cortex)
def fill_unknown_labels_per_hemi(gt, unknown_label, cortex_stop):
    """
    Function to replace label 1000 (lh unknown) and 2000 (rh unknown) with closest class for each voxel.
    :param np.ndarray gt: ground truth segmentation with class unknown
    :param int unknown_label: class label for unknown (lh: 1000, rh: 2000)
    :param int cortex_stop: class label at which cortical labels of this hemi stop (lh: 2000, rh: 3000)
    :return: np.ndarray gt: ground truth segmentation with replaced unknown class labels
    """
    # Define shape of image and dilation element
    h, w, d = gt.shape
    struct1 = ndimage.generate_binary_structure(3, 2)

    # Get indices of unknown labels, dilate them to get closest sorrounding parcels
    unknown = gt == unknown_label
    unknown = (morphology.binary_dilation(unknown, struct1) ^ unknown)
    list_parcels = np.unique(gt[unknown])

    # Mask all subcortical structures (fill unknown with closest cortical parcels only)
    mask = (list_parcels > unknown_label) & (list_parcels < cortex_stop)
    list_parcels = list_parcels[mask]

    # For each closest parcel, blur label with gaussian filter (spread), append resulting blurred images
    blur_vals = np.ndarray((h, w, d, 0), dtype=float)

    for idx in range(len(list_parcels)):
        aseg_blur = filters.gaussian_filter(1000 * np.asarray(gt == list_parcels[idx], dtype=float), sigma=5)
        blur_vals = np.append(blur_vals, np.expand_dims(aseg_blur, axis=3), axis=3)

    # Get for each position parcel with maximum value after blurring (= closest parcel)
    unknown = np.argmax(blur_vals, axis=3)
    unknown = np.reshape(list_parcels[unknown.ravel()], (h, w, d))

    # Assign the determined closest parcel to the unknown class (case-by-case basis)
    mask = gt == unknown_label
    gt[mask] = unknown[mask]

    return gt


# Label mapping functions (to aparc (eval) and to label (train))
def map_label2aparc_aseg(mapped_aseg):
    """
    Function to perform look-up table mapping from label space to aparc.DKTatlas+aseg space
    :param np.ndarray mapped_aseg: label space segmentation
    :return: np.ndarray aseg: segmentation in aparc+aseg space
    """
    aseg = np.zeros_like(mapped_aseg)
    labels = np.array([0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                       15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                       46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                       77, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                       1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
                       1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
                       2002, 2005, 2010, 2012, 2013, 2014, 2016, 2017, 2021, 2022, 2023,
                       2024, 2025, 2028])
    h, w, d = aseg.shape

    aseg = labels[mapped_aseg.ravel()]

    aseg = aseg.reshape((h, w, d))

    return aseg


def map_aparc_aseg2label(aseg, aseg_nocc=None):
    """
    Function to perform look-up table mapping of aparc.DKTatlas+aseg.mgz data to label space
    :param np.ndarray aseg: ground truth aparc+aseg
    :param None/np.ndarray aseg_nocc: ground truth aseg without corpus callosum segmentation
    :return: np.ndarray mapped_aseg: label space segmentation (coronal and axial)
    :return: np.ndarray mapped_aseg_sag: label space segmentation (sagittal)
    """
    aseg_temp = aseg.copy()
    aseg[aseg == 80] = 77  # Hypointensities Class
    aseg[aseg == 85] = 0  # Optic Chiasma to BKG
    aseg[aseg == 62] = 41  # Right Vessel to Right GM
    aseg[aseg == 30] = 2  # Left Vessel to Left GM
    aseg[aseg == 72] = 24  # 5th Ventricle to CSF

    # If corpus callosum is not removed yet, do it now
    if aseg_nocc is not None:
        cc_mask = (aseg >= 251) & (aseg <= 255)
        aseg[cc_mask] = aseg_nocc[cc_mask]

    aseg[aseg == 3] = 0  # Map Remaining Cortical labels to background
    aseg[aseg == 42] = 0

    # If ctx-unknowns are not filled yet, do it now
    if np.any(np.in1d([1000, 2000], aseg.ravel())):
        aseg = fill_unknown_labels_per_hemi(aseg, 1000, 2000)
        aseg = fill_unknown_labels_per_hemi(aseg, 2000, 3000)

    cortical_label_mask = (aseg >= 2000) & (aseg <= 2999)
    aseg[cortical_label_mask] = aseg[cortical_label_mask] - 1000

    # Preserve Cortical Labels
    aseg[aseg_temp == 2014] = 2014
    aseg[aseg_temp == 2028] = 2028
    aseg[aseg_temp == 2012] = 2012
    aseg[aseg_temp == 2016] = 2016
    aseg[aseg_temp == 2002] = 2002
    aseg[aseg_temp == 2023] = 2023
    aseg[aseg_temp == 2017] = 2017
    aseg[aseg_temp == 2024] = 2024
    aseg[aseg_temp == 2010] = 2010
    aseg[aseg_temp == 2013] = 2013
    aseg[aseg_temp == 2025] = 2025
    aseg[aseg_temp == 2022] = 2022
    aseg[aseg_temp == 2021] = 2021
    aseg[aseg_temp == 2005] = 2005

    labels = np.array([0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                       15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                       46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                       77, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                       1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
                       1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
                       2002, 2005, 2010, 2012, 2013, 2014, 2016, 2017, 2021, 2022, 2023,
                       2024, 2025, 2028])

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels) + 1, dtype='int')
    for idx, value in enumerate(labels):
        lut_aseg[value] = idx

    # Remap Label Classes - Perform LUT Mapping - Coronal, Axial

    mapped_aseg = lut_aseg.ravel()[aseg.ravel()]

    mapped_aseg = mapped_aseg.reshape((h, w, d))

    # Map Sagittal Labels
    aseg[aseg == 2] = 41
    aseg[aseg == 3] = 42
    aseg[aseg == 4] = 43
    aseg[aseg == 5] = 44
    aseg[aseg == 7] = 46
    aseg[aseg == 8] = 47
    aseg[aseg == 10] = 49
    aseg[aseg == 11] = 50
    aseg[aseg == 12] = 51
    aseg[aseg == 13] = 52
    aseg[aseg == 17] = 53
    aseg[aseg == 18] = 54
    aseg[aseg == 26] = 58
    aseg[aseg == 28] = 60
    aseg[aseg == 31] = 63

    cortical_label_mask = (aseg >= 2000) & (aseg <= 2999)
    aseg[cortical_label_mask] = aseg[cortical_label_mask] - 1000

    labels_sag = np.array([0, 14, 15, 16, 24, 41, 43, 44, 46, 47, 49,
                           50, 51, 52, 53, 54, 58, 60, 63, 77, 1002,
                           1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014,
                           1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025,
                           1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035])

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels_sag) + 1, dtype='int')
    for idx, value in enumerate(labels_sag):
        lut_aseg[value] = idx

    # Remap Label Classes - Perform LUT Mapping - Sagittal

    mapped_aseg_sag = lut_aseg.ravel()[aseg.ravel()]

    mapped_aseg_sag = mapped_aseg_sag.reshape((h, w, d))

    return mapped_aseg, mapped_aseg_sag


def sagittal_coronal_remap_lookup(x):
    """
    Dictionary mapping to convert left labels to corresponding right labels for aseg
    :param int x: label to look up
    :return: dict: left-to-right aseg label mapping dict
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


def map_prediction_sagittal2full(prediction_sag, num_classes=79):
    """
    Function to remap the prediction on the sagittal network to full label space used by coronal and axial networks
    (full aparc.DKTatlas+aseg.mgz)
    :param np.ndarray prediction_sag: sagittal prediction (labels)
    :param int num_classes: number of classes (96 for full classes, 79 for hemi split)
    :return: np.ndarray prediction_full: Remapped prediction
    """
    if num_classes == 96:
        idx_list = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 14, 15, 4, 16,
                               17, 18, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 20, 21, 22,
                               23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50], dtype=np.int16)

    else:
        idx_list = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 14, 15, 4, 16,
                               17, 18, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 20, 22, 27,
                               29, 30, 31, 33, 34, 38, 39, 40, 41, 42, 45], dtype=np.int16)

    prediction_full = prediction_sag[:, idx_list, :, :]
    return prediction_full


# Clean up and class separation
def bbox_3d(img):
    """
    Function to extract the three-dimensional bounding box coordinates.
    :param np.ndarray img: mri image
    :return: float rmin
    :return: float rmax
    :return: float cmin
    :return: float cmax
    :return: float zmin
    :return: float zmax
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
    :return: np.ndarray largest_cc: largest connected component of the segmentation array
    """
    labels = label(segmentation, connectivity=3, background=0)

    bincount = np.bincount(labels.flat)
    background = np.argmax(bincount)
    bincount[background] = -1

    largest_cc = labels == np.argmax(bincount)

    return largest_cc


# Class Operator for image loading (orig only)
class OrigDataThickSlices(Dataset):
    """
    Class to load a given image and segmentation and prepare it
    for network training.
    """
    def __init__(self, img_filename, orig, plane='Axial', slice_thickness=3, transforms=None):

        try:
            self.img_filename = img_filename
            self.plane = plane
            self.slice_thickness = slice_thickness

            # Transform Data as needed
            if plane == 'Sagittal':
                orig = transform_sagittal(orig)
                print('Loading Sagittal')

            elif plane == 'Axial':
                orig = transform_axial(orig)
                print('Loading Axial')

            else:
                print('Loading Coronal.')

            # Create Thick Slices
            orig_thick = get_thick_slices(orig, self.slice_thickness)

            # Make 4D
            orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
            self.images = orig_thick

            self.count = self.images.shape[0]

            self.transforms = transforms

            print("Successfully loaded Image from {}".format(img_filename))

        except Exception as e:
            print("Loading failed. {}".format(e))

    def __getitem__(self, index):

        img = self.images[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return {'image': img}

    def __len__(self):
        return self.count


##
# Dataset loading (for training)
##

# Operator to load hdf5-file for training
class AsegDatasetWithAugmentation(Dataset):
    """
    Class for loading aseg file with augmentations (transforms)
    """
    def __init__(self, params, transforms=None):

        # Load the h5 file and save it to the dataset
        try:
            self.params = params

            # Open file in reading mode
            with h5py.File(self.params['dataset_name'], "r") as hf:
                self.images = np.array(hf.get('orig_dataset'))
                self.labels = np.array(hf.get('aseg_dataset'))
                self.weights = np.array(hf.get('weight_dataset'))
                self.subjects = np.array(hf.get("subject"))

            self.count = self.images.shape[0]
            self.transforms = transforms

            print("Successfully loaded {} with plane: {}".format(params["dataset_name"], params["plane"]))

        except Exception as e:
            print("Loading failed: {}".format(e))

    def get_subject_names(self):
        return self.subjects

    def __getitem__(self, index):

        img = self.images[index]
        label = self.labels[index]
        weight = self.weights[index]

        if self.transforms is not None:
            tx_sample = self.transforms({'img': img, 'label': label, 'weight': weight})
            img = tx_sample['img']
            label = tx_sample['label']
            weight = tx_sample['weight']

        return {'image': img, 'label': label, 'weight': weight}

    def __len__(self):
        return self.count

