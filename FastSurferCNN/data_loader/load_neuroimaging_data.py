
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

from skimage.measure import label
from torch.utils.data.dataset import Dataset
from .conform import is_conform, conform

##
# Helper Functions
##


# Conform an MRI brain image to UCHAR, RAS orientation, and 1mm isotropic voxels
def load_and_conform_image(img_filename, interpol=1):
    """
    Function to load MRI image and conform it to UCHAR, RAS orientation and 1mm isotropic voxels size
    (if it does not already have this format)
    :param str img_filename: path and name of volume to read
    :param int interpol: interpolation order for image conformation (0=nearest,1=linear(default),2=quadratic,3=cubic)
    :return:
    """
    orig = nib.load(img_filename)

    if not is_conform(orig):
        print('Conforming image to UCHAR, RAS orientation, and 1mm isotropic voxels')
        orig = conform(orig, interpol)

    # Collect header and affine information
    header_info = orig.header
    affine_info = orig.affine
    orig = np.asarray(orig.get_data(), dtype=np.uint8)

    return header_info, affine_info, orig


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
def create_weight_mask(mapped_aseg, max_weight=5, max_edge_weight=5):
    """
    Function to create weighted mask - with median frequency balancing and edge-weighting
    :param mapped_aseg:
    :param max_weight:
    :param max_edge_weight:
    :return:
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


# Label mapping functions (to aparc (eval) and to label (train))
def map_label2aparc_aseg(mapped_aseg):
    """
    Function to perform look-up table mapping from label space to aparc.DKTatlas+aseg space
    :param np.ndarray mapped_aseg: label space segmentation (aparc.DKTatlas + aseg)
    :return:
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
    :return:
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


def map_prediction_sagittal2full(prediction_sag, num_classes=79):
    """
    Function to remap the prediction on the sagittal network to full label space used by coronal and axial networks
    (full aparc.DKTatlas+aseg.mgz)
    :param prediction_sag: sagittal prediction (labels)
    :param int num_classes: number of classes (96 for full classes, 79 for hemi split)
    :return: Remapped prediction
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

