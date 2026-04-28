# Copyright 2024
# AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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
import numpy as np

from FastSurferCNN.data_loader.conform import AXCODES, conform, getscale, scalecrop
from FastSurferCNN.utils import ShapeType
from HypVINN.config.hypvinn_global_var import FS_CLASS_NAMES, HYPVINN_CLASS_NAMES, SAG2FULL_MAP, hyposubseg_labels

##
# Helper Functions
##


def reorient_img(img, ref_img):
    """
    Reorient a Nibabel image based on the orientation of a reference nibabel image.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Nibabel Image to reorient.
    ref_img : nibabel.Nifti1Image
        Reference orientation nibabel image.

    Returns
    -------
    img : nibabel.Nifti1Image
        Reoriented image.
    """
    # if the affines are the same, no reorientation is required and we can skip this
    if np.array_equal(ref_img.affine, img.affine):
        return img
    from nibabel.orientations import aff2axcodes
    target_orientation = "soft " + "".join(aff2axcodes(ref_img.affine, AXCODES))
    # returns the same class as img
    return conform(img, orientation=target_orientation, vox_size=None, img_size=None, dtype=None, rescale=None)


def transform_axial2coronal(vol: np.ndarray, axial2coronal: bool = True) -> np.ndarray:
    """
    Transforms a volume into the coronal axis and back.

    This function is used to transform a volume into the coronal axis and back. The
    transformation is done by moving the axes of the volume. If the `axial2coronal`
    parameter is set to True, the function will transform from axial to coronal. If it
    is set to False, the function will transform from coronal to axial.

    Parameters
    ----------
    vol : np.ndarray
        The image volume to transform.
    axial2coronal : bool, optional
        A flag to determine the direction of the transformation. If True, transform from
        axial to coronal. If False, transform from coronal to axial. (Default: True).

    Returns
    -------
    np.ndarray
        The transformed volume.
    """
    # TODO check compatibility with axis transform from CerebNet
    if axial2coronal:
        return np.moveaxis(vol, [0, 1, 2], [0, 2, 1])
    else:
        return np.moveaxis(vol, [0, 1, 2], [0, 2, 1])


def transform_axial2sagittal(vol: np.ndarray,
                             axial2sagittal: bool = True) -> np.ndarray:
    """
    Transforms a volume into the sagittal axis and back.

    This function is used to transform a volume into the sagittal axis and back. The
    transformation is done by moving the axes of the volume. If the `axial2sagittal`
    parameter is set to True, the function will transform from axial to sagittal. If it
    is set to False, the function will transform from sagittal to axial.

    Parameters
    ----------
    vol : np.ndarray
        The image volume to transform.
    axial2sagittal : bool, default=True
        A flag to determine the direction of the transformation. If True, transform from
        axial to sagittal. If False, transform from sagittal to axial. (Default: True).

    Returns
    -------
    np.ndarray
        The transformed volume.
    """
    # TODO check compatibility with axis transform from CerebNet
    if axial2sagittal:
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])
    else:
        return np.moveaxis(vol, [0, 1, 2], [1, 2, 0])


def rescale_image(img_data: np.ndarray) -> np.ndarray:
    """
    Rescale the image data to the range [0, 255].

    This function rescales the input image data to the range [0, 255].

    Parameters
    ----------
    img_data : np.ndarray
        The image data to rescale.

    Returns
    -------
    np.ndarray
        The rescaled image data.
    """
    # Conform intensities
    # TODO move function to FastSurferCNN similar: CerebNet.datasets.utils.rescale_image
    src_min, scale = getscale(img_data, 0, 255)
    mapped_data = img_data

    # this used to rescale, if the image was not uint8 and any intensity was > 255
    if not np.allclose([src_min, scale], [0, 1]):
        mapped_data = scalecrop(img_data, 0, 255, src_min, scale)

    return np.rint(mapped_data).astype(np.uint8)


def hypo_map_label2subseg(mapped_subseg: np.ndarray[ShapeType, np.dtype[np.integer]]) \
        -> np.ndarray[ShapeType, np.dtype[np.integer]]:
    """
    Perform look-up table mapping from label space to subseg space.

    This function is used to perform a look-up table mapping from label space to subseg
    space.

    Parameters
    ----------
    mapped_subseg : np.ndarray of integers
        The input array in label space to be mapped to subseg space.

    Returns
    -------
    np.ndarray of integers
        The mapped array in subseg space.
    """
    # TODO can this function be replaced by a Mapper and a mapping file?
    labels, _ = hyposubseg_labels
    subseg = np.zeros_like(mapped_subseg)
    h, w, d = subseg.shape
    subseg = labels[mapped_subseg.ravel()]

    return subseg.reshape((h, w, d))


def hypo_map_prediction_sagittal2full(
        prediction_sag: np.ndarray[ShapeType, np.dtype[np.integer]],
) -> np.ndarray[ShapeType, np.dtype[np.integer]]:
    """
    Remap the prediction on the sagittal network to full label space.

    This function is used to remap the prediction on the sagittal network to the full
    label space used by the coronal and axial networks.

    Parameters
    ----------
    prediction_sag : np.ndarray of integers
        The sagittal prediction in label space to be remapped to full label space.

    Returns
    -------
    np.ndarray of integers
        The remapped prediction in full label space.
    """
    # TODO can this function be replaced by a Mapper and a mapping file?

    idx_list = list(SAG2FULL_MAP.values())
    prediction_full = prediction_sag[:, idx_list, :, :]
    return prediction_full


def hypo_map_subseg_2_fsseg(
        subseg: np.ndarray[ShapeType, np.dtype[np.integer]],
        reverse: bool = False,
) -> np.ndarray[ShapeType, np.dtype[np.int16]]:
    """
    Remap HypVINN internal labels to FastSurfer Labels and vice versa.

    This function is used to remap HypVINN internal labels to FastSurfer Labels and vice
    versa. If the `reverse` parameter is set to False, the function will map HypVINN
    labels to FastSurfer labels. If it is set to True, the function will map FastSurfer
    labels to HypVINN labels.

    Parameters
    ----------
    subseg : np.ndarray of integers
        The input array with HypVINN or FastSurfer labels to be remapped.
    reverse : bool, default=False
        A flag to determine the direction of the remapping. If False, remap HypVINN
        labels to FastSurfer labels. If True, remap FastSurfer labels to HypVINN labels.

    Returns
    -------
    np.ndarray of integers
        The remapped array with FastSurfer or HypVINN labels.
    """
    # TODO can this function be replaced by a Mapper and a mapping file?

    fsseg: np.ndarray[ShapeType, np.dtype[np.int16]] = np.zeros_like(subseg, dtype=np.int16)

    if not reverse:
        for value, name in HYPVINN_CLASS_NAMES.items():
            fsseg[subseg == value] = FS_CLASS_NAMES[name]
    else:
        reverse_hypvinn = dict(map(reversed, HYPVINN_CLASS_NAMES.items()))
        for name, value in FS_CLASS_NAMES.items():
            fsseg[subseg == value] = reverse_hypvinn[name]
    return fsseg
