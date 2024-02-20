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

import os
import numpy as np
import nibabel as nib
from HypVINN.data_loader.data_utils import hypo_map_subseg_2_fsseg
from skimage.measure import label
from scipy import ndimage
import FastSurferCNN.utils.logging as logging

LOGGER = logging.get_logger(__name__)


def save_segmentation(prediction,orig_path,affine,header,save_dir):
    from HypVINN.data_loader.data_utils import reorient_img
    from HypVINN.config.hypvinn_files import HYPVINN_MASK_NAME,HYPVINN_SEG_NAME

    pred_arr, labels_cc, savemask = get_clean_labels(np.array(prediction, dtype=np.uint8))
    #Mapped HypVINN labelst to FreeSurfer Hypvinn Labels
    pred_arr = hypo_map_subseg_2_fsseg(pred_arr)
    LOGGER.info('Orig data orientation : {}'.format(nib.aff2axcodes(nib.load(orig_path).affine)))

    if savemask:
        mask_img = nib.Nifti1Image(labels_cc, affine, header)
        LOGGER.info('HypoVINN Mask orientation : {}'.format(nib.aff2axcodes(mask_img.affine)))
        mask_img = reorient_img(mask_img,nib.load(orig_path))
        LOGGER.info('HypoVINN Mask after re-orientation : {}'.format(nib.aff2axcodes(mask_img.affine)))
        name = HYPVINN_MASK_NAME
        nib.save(mask_img, os.path.join(save_dir, name))

    pred_img = nib.Nifti1Image(pred_arr, affine=affine, header=header)
    LOGGER.info('HypoVINN Prediction orientation : {}'.format(nib.aff2axcodes(pred_img.affine)))
    pred_img = reorient_img(pred_img,nib.load(orig_path))
    LOGGER.info('HypoVINN Prediction after re-orientation : {}'.format(nib.aff2axcodes(pred_img.affine)))
    pred_img.set_data_dtype(np.int16) #Maximum value 939
    save_as = os.path.join(save_dir, HYPVINN_SEG_NAME)
    nib.save(pred_img, save_as)

    return save_as

def save_logits(logits,orig_path,affine,header,save_dir,mode):
    from HypVINN.data_loader.data_utils import reorient_img
    LOGGER.info('Orig data orientation : {}'.format(nib.aff2axcodes(nib.load(orig_path).affine)))
    nifti_img = nib.Nifti1Image(logits.astype(np.float32), affine=affine, header=header)
    LOGGER.info('HypoVINN logits orientation : {}'.format(nib.aff2axcodes(nifti_img.affine)))
    nifti_img = reorient_img(nifti_img,orig_path)
    LOGGER.info('HypoVINN logits after re-orientation : {}'.format(nib.aff2axcodes(nifti_img.affine)))
    nifti_img.set_data_dtype(np.float32)
    save_as = os.path.join(save_dir, 'HypVINN_logits_{}.nii.gz'.format(mode))
    nib.save(nifti_img, save_as)
    return save_as

def get_clean_mask(segmentation,optic = False):

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
    mask = ndimage.binary_dilation(copy_segmentation, structure=struct1, iterations= iterations).astype(np.uint8)
    labels_cc = label(mask, connectivity=3, background=0)
    bincount = np.bincount(labels_cc.flat)

    if len(bincount) > 2:
        if  optic:
            LOGGER.info('Check Optic Labels')
        else:
            LOGGER.info('Check Hypothalamus Labels')
        savemask = True


    background = np.argmax(bincount)
    bincount[background] = -1
    largest_cc = labels_cc == np.argmax(bincount)
    clean_seg = copy_segmentation * largest_cc

    #remove globus pallidus
    clean_seg[clean_seg == 13] = 0
    clean_seg[clean_seg == 20] = 0

    return clean_seg, labels_cc, savemask

def get_clean_labels(segmentation):
    """
    Function to find largest connected component of segmentation.
    :param np.ndarray segmentation: segmentation
    :return:
    """

    # Mask largest CC without optic labels
    clean_seg,labels_cc,savemask = get_clean_mask(segmentation)
    # Mask largest CC from optic labels
    optic_clean_seg,optic_labels_cc,optic_savemask = get_clean_mask(segmentation,optic=True)

    #clean segmentation from both largest_cc
    clean_seg = clean_seg + optic_clean_seg

    # mask from both largest_cc
    optic_mask = optic_labels_cc > 0
    other_mask = labels_cc > 0
    #multiplication times one to change from boolean
    non_intersect = (optic_mask * 1 - other_mask * 1) * optic_mask

    optic_labels_cc += np.max(np.unique(labels_cc))
    labels_cc = labels_cc + optic_labels_cc * non_intersect
    #TO-DO remove in the future, for now always save mask
    savemask= True

    #remove small group of voxels less than 3
    small_mask = clean_seg > 0
    labels_small = label(small_mask, connectivity=3, background=0)
    bincount_small = np.bincount(labels_small.flat)
    idx = np.where(bincount_small <= 3)
    if idx[0].any():
        for i in idx[0]:
            small_mask [labels_small == i] = False

    clean_seg = clean_seg * small_mask

    return clean_seg,labels_cc, savemask











