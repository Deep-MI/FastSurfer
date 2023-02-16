
# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
from typing import Literal, TypeVar

import numpy as np
import torch
from numpy import typing as npt

Plane = Literal['axial', 'coronal', 'sagittal']
AT = TypeVar('AT', np.ndarray, torch.Tensor)

# CLASSES for final evaluation
EVAL_CLASS_NAMES = {
    "Background":             0,
    "Left_I_IV":              1,
    "Right_I_IV":             2,
    "Left_V":                 3,
    "Right_V":                4,
    "Left_VI":                5,
    "Vermis_VI":              6,
    "Right_VI":               7,
    "Left_CrusI":             8,
    "Right_CrusI":            10,
    "Left_CrusII":            11,
    "Right_CrusII":           13,
    "Left_VIIb":              14,
    "Right_VIIb":             16,
    "Vermis_VII":             12,
    "Left_VIIIa":             17,
    "Right_VIIIa":            19,
    "Left_VIIIb":             20,
    "Right_VIIIb":            22,
    "Vermis_VIII":            18,
    "Left_IX":                23,
    "Vermis_IX":              24,
    "Right_IX":               25,
    "Left_X":                 26,
    "Vermis_X":               27,
    "Right_X":                28,
    # Dummy label id
    "Vermis":                100,
    "L_Gray_Matter":         101,
    "R_Gray_Matter":         102,

    "Left_Corpus_Medullare":  37,
    "Right_Corpus_Medullare": 38,
}

GRAY_MATTER = {
    "Left": {
        "Left_I_IV": 1,
        "Left_V": 3,
        "Left_VI": 5,
        "Left_CrusI": 8,
        "Left_CrusII": 11,
        "Left_VIIb": 14,
        "Left_VIIIa": 17,
        "Left_VIIIb": 20,
        "Left_IX": 23,
        "Left_X": 26,
    },
    "Right": {
        "Right_I_IV": 2,
        "Right_V": 4,
        "Right_VI": 7,
        "Right_CrusI": 10,
        "Right_CrusII": 13,
        "Right_VIIb": 16,
        "Right_VIIIa": 19,
        "Right_VIIIb": 22,
        "Right_IX": 25,
        "Right_X": 28,
    }
}

VERMIS_NAMES = {
    "Vermis_VI":  6,
    "Vermis_VII": 12,
    "Vermis_VIII": 18,
    "Vermis_IX": 24,
    "Vermis_X": 27,
}

sag_right2left = {
    2:  1,
    4:  3,
    7:  5,
    9:  8,
    11: 10,
    13: 12,
    16: 15,
    18: 17,
    22: 20,
    25: 23,
    27: 26
}

LABELS_SAG = {
    "cereb_subseg": np.array([ 0,  1,  3,  5,  6,
                               8, 10, 12, 14, 15,
                               17, 19, 20, 21, 23, 24, 26])
}


FLIPPED_LABELS = np.array([
    0,
    2, 1,
    4, 3,
    7, 6, 5,
    9, 8,
    11, 10,
    13, 12, 14,
    16, 15,
    18, 17, 19,
    22, 21, 20,
    25, 24, 23,
    27, 26
])

SAG2FULL_MAP = {
    "cereb_subseg": np.array([0,
                              1, 1,
                              2, 2,
                              3, 4, 3,
                              5, 5,
                              6, 6,
                              7, 7, 8,
                              9, 9,
                              10, 10, 11,
                              12, 13, 12,
                              14, 15, 14,
                              16, 16
                              ])
}


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
        return np.moveaxis(vol, [0, 1, 2, 3], [0, 2, 3, 1])
    else:
        return np.moveaxis(vol, [0, 1, 2, 3], [0, 3, 1, 2])


def transform_sagittal(vol, coronal2sagittal=True):
    """
    Function to transform volume into Sagittal axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2sagittal: transform from coronal to sagittal = True (default),
                                transform from sagittal to coronal = False
    :return:
    """
    if coronal2sagittal:
        return np.moveaxis(vol, [0, 1, 2, 3], [0, 3, 2, 1])
    else:
        return np.moveaxis(vol, [0, 1, 2, 3], [0, 3, 2, 1])


def transform_coronal(vol, axial2coronal=True):
    """
    Function to transform volume into coronal axis and back
    :param np.ndarray vol: image volume to transform
    :param bool axial2coronal: transform from axial to coronal = True (default),
                                transform from coronal to axial = False
    :return:
    """
    if axial2coronal:
        if len(vol.shape) == 4:
            return np.moveaxis(vol, [0, 1, 2, 3], [0, 1, 3, 2])
        if len(vol.shape) == 5:
            return np.moveaxis(vol, [0, 1, 2, 3, 4], [0, 1, 3, 2, 4])
    else:
        if len(vol.shape) == 4:
            return np.moveaxis(vol, [0, 1, 2, 3], [0, 1, 3, 2])
        if len(vol.shape) == 5:
            return np.moveaxis(vol, [0, 1, 2, 3, 4], [0, 1, 3, 2, 4])


def transform_axial2sagittal(vol, axial2sagittal=True):
    """
    Function to transform volume into Sagittal axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2sagittal: transform from coronal to sagittal = True (default),
                                transform from sagittal to coronal = False
    :return:
    """
    if axial2sagittal:
        if len(vol.shape) == 4:
            return np.moveaxis(vol, [0, 1, 2, 3], [0, 3, 1, 2])
        if len(vol.shape) == 5:
            return np.moveaxis(vol, [0, 1, 2, 3, 4], [0, 3, 1, 2, 4])
    else:
        if len(vol.shape) == 4:
            return np.moveaxis(vol, [0, 1, 2, 3], [0, 2, 3, 1])
        if len(vol.shape) == 5:
            return np.moveaxis(vol, [0, 1, 2, 3, 4], [0, 2, 3, 1, 4])


def get_plane_transform(plane, primary_slice_dir='coronal'):
    if primary_slice_dir == 'coronal':
        if plane == 'sagittal':
            return transform_sagittal
        elif plane == 'axial':
            return transform_axial
        else:
            return lambda v: v
    # primary slice_direction is axial
    elif primary_slice_dir == 'axial':
        if plane == 'sagittal':
            return transform_axial2sagittal
        elif plane == 'coronal':
            return transform_coronal
        else:
            return lambda v: v


def filter_blank_slices_thick(data_dict, img_key="img", lbl_key="label", threshold=10):
    """
    Function to filter blank slices from the volume using the label volume
    :param dict data_dict: dictionary containing all volumes need to be filtered
    :param img_key
    :param lbl_key
    :param threshold
    :return:
    """
    # Get indices of all slices with more than threshold labels/pixels
    selected_slices = (np.sum(data_dict[lbl_key], axis=(1, 2)) > threshold)
    data_dict[img_key] = data_dict[img_key][selected_slices]
    data_dict[lbl_key] = data_dict[lbl_key][selected_slices]


def create_weight_mask2d(label_map, class_wise_weights, max_edge_weight=5):
    """
    Function to create weighted mask - with median frequency balancing and edge-weighting
    :param label_map:
    :param class_wise_weights:
    :param max_edge_weight:
    :return:
    """
    (h, w) = label_map.shape
    weights_mask = np.reshape(class_wise_weights[label_map.ravel()], (h, w))

    # Gradient Weighting
    (gx, gy) = np.gradient(label_map)
    grad_weight = max_edge_weight * np.asarray(np.power(np.power(gx, 2) + np.power(gy, 2), 0.5) > 0,
                                               dtype='float')

    weights_mask += grad_weight
    return weights_mask


def map_sag2label(lbl_data, label_type='cereb_subseg'):
    """
    Mapping right ids to left and relabeling
    Args:
        lbl_data:
        label_type:

    Returns:

    """
    for r_lbl, l_lbl in sag_right2left.items():
        lbl_data[lbl_data == r_lbl] = l_lbl

    lbls = LABELS_SAG[label_type]
    lut = np.zeros(max(lbls) + 1, dtype='int')
    for idx, value in enumerate(lbls):
        lut[value] = idx

    mapped_labels = lut.ravel()[lbl_data.ravel()]
    mapped_labels = mapped_labels.reshape(lbl_data.shape)
    return mapped_labels


def map_prediction_sagittal2full(prediction_sag, lbl_type):
    """
    Function to remap the prediction on the sagittal network to full label space used by coronal and axial networks
    :param prediction_sag: sagittal prediction (labels)
    :param lbl_type: type of label
    :return: Remapped prediction
    """

    idx_list = SAG2FULL_MAP[lbl_type]
    prediction_full = prediction_sag[:, idx_list, :, :]
    return prediction_full


def get_aseg_cereb_mask(aseg_map: npt.NDArray[int]) -> npt.NDArray[bool]:
    """Get a boolean mask of the cerebellum from a segmentation image."""
    wm_cereb_mask = np.logical_or(aseg_map == 46, aseg_map == 7)
    gm_cereb_mask = np.logical_or(aseg_map == 47, aseg_map == 8)
    return np.logical_or(wm_cereb_mask, gm_cereb_mask)


def unpad_volume(vol, borders):
    unpad_vol = vol[borders[0, 0]:borders[0, 1],
                borders[1, 0]:borders[1, 1],
                borders[2, 0]:borders[2, 1]]
    return unpad_vol


def uncrop_volume(vol, uncrop_shape, roi):
    uncrop_vol = np.zeros(uncrop_shape)
    uncrop_vol[roi] = vol
    return uncrop_vol


def get_binary_map(lbl_map, class_names):
    bin_map = np.logical_or.reduce(list(map(lambda l: lbl_map == l, class_names)))
    return bin_map


def slice_lia2ras(plane: Plane, data: AT, /, thick_slices: bool = False) -> AT:
    """Maps the data from LIA to RAS orientation.

    Args:
        plane: the slicing direction (usually moved into batch dimension)
        data: the data array of shape [plane, Channels, H, W]
        thick_slices: whether the channels are thick slices and should also be flipped (default: False).

    Returns:
        data reoriented from LIA to RAS of [plane, Channels, H, W] (plane: 'sagittal' or 'coronal') or
            [plane, Channels, W, H] (plane: 'axial').
    """
    if isinstance(data, np.ndarray):
        flip, swapaxes = np.flip, np.swapaxes
    else:  # Tensor
        from torch import flip, swapaxes

    if thick_slices and plane in ["sagittal", "coronal"]:
        data = flip(data, (1,))
    if plane == 'sagittal':
        return flip(data, (-1,))
    elif plane == "coronal":
        return flip(data, (-1, -2))
    elif plane == "axial":
        return flip(swapaxes(data, -1, -2), (-2,))
    else:
        raise ValueError("invalid plane")


def slice_ras2lia(plane: Plane, data: AT, /, thick_slices: bool = False) -> AT:
    """Maps the data from RAS to LIA orientation.

    Args:
        plane: the slicing direction (usually moved into batch dimension)
        data: the data array of shape [plane, Channels, H, W]
        thick_slices: whether the channels are thick slices and should also be flipped (default: False).

    Returns:
        data reoriented from RAS to LIA of [plane, Channels, H, W] (plane: 'sagittal' or 'coronal') or
            [plane, Channels, W, H] (plane: 'axial').
    """
    if isinstance(data, np.ndarray):
        flip, swapaxes = np.flip, np.swapaxes
    else:  # Tensor
        from torch import flip, swapaxes

    if thick_slices and plane in ["sagittal", "coronal"]:
        data = flip(data, (1,))
    if plane == 'sagittal':
        return flip(data, (-1,))
    elif plane == "coronal":
        return flip(data, (-1, -2))
    elif plane == "axial":
        return swapaxes(flip(data, (-2,)), -1, -2)
    else:
        raise ValueError("invalid plane")
