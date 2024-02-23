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

import numpy as np
import torch
from torch.utils.data import Dataset


from HypVINN.data_loader.data_utils import transform_axial2sagittal,transform_axial2coronal
from FastSurferCNN.data_loader.data_utils import get_thick_slices

import FastSurferCNN.utils.logging as logging

logger = logging.get_logger(__name__)

# Operator to load imaged for inference
class HypoVINN_dataset(Dataset):
    """
    Class to load MRI-Image and process it to correct format for HypVINN network inference
    The HypVINN Dataset passed during Inference the input images,the scale factor for the VINN layer and a weight factor (wT1,wT2)
    The Weight factor determines the running mode of the HypVINN model
    if wT1 =1 and wT2 =0. The HypVINN model will only allow the flow of the T1 information (mode = t1)
    if wT1 =0 and wT2 =1. The HypVINN model will only allow the flow of the T2 information (mode = t2)
    if wT1 !=1 and wT2 !=1. The HypVINN model will automatically weigh the T1 information and the T2 information based on the learned modality weights (mode = multi)
    """
    def __init__(self, subject_name, modalities, orig_zoom, cfg, mode='multi', transforms=None):
        self.subject_name = subject_name
        self.plane = cfg.DATA.PLANE
        #Inference Mode
        self.mode = mode
        #set thickness base on train paramters
        if cfg.MODEL.MODE in ['t1','t2']:
            self.slice_thickness = cfg.MODEL.NUM_CHANNELS//2
        else:
            self.slice_thickness = cfg.MODEL.NUM_CHANNELS//4

        self.base_res = cfg.MODEL.BASE_RES

        if self.mode == 't1':
            orig_thick = self._standarized_img(modalities['t1'],orig_zoom, modalitie='t1')
            orig_thick = np.concatenate((orig_thick, orig_thick), axis=-1)
            self.weight_factor = torch.from_numpy(np.asarray([1.0, 0.0]))

        elif self.mode == 't2':
            orig_thick = self._standarized_img(modalities['t2'],orig_zoom, modalitie='t2')
            orig_thick = np.concatenate((orig_thick, orig_thick), axis=-1)
            self.weight_factor = torch.from_numpy(np.asarray([0.0, 1.0]))
        else:
            t1_orig_thick = self._standarized_img(modalities['t1'], orig_zoom, modalitie='t1')
            t2_orig_thick = self._standarized_img(modalities['t2'],orig_zoom, modalitie='t2')
            orig_thick = np.concatenate((t1_orig_thick, t2_orig_thick), axis=-1)
            self.weight_factor = torch.from_numpy(np.asarray([0.5, 0.5]))
        
        # Transpose from W,H,N,C to N,W,H,C
        orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
        self.images = orig_thick
        self.count = self.images.shape[0]
        self.transforms = transforms

        logger.info(f"Successfully loaded Image from {subject_name} for {self.plane} model")

        if (cfg.MODEL.MULTI_AUTO_W or cfg.MODEL.MULTI_AUTO_W_CHANNELS) and (self.mode == 'multi' or cfg.MODEL.DUPLICATE_INPUT) :
            logger.info(f"For inference T1 block weight and the T2 block are set to the weights learn during training")
        else:
            logger.info(f"For inference T1 block weight was set to : {self.weight_factor.numpy()[0]} and the T2 block was set to: {self.weight_factor.numpy()[1]}")

    def _standarized_img(self,orig_data,orig_zoom,modalitie):
        if self.plane == "sagittal":
            orig_data = transform_axial2sagittal(orig_data)
            self.zoom = orig_zoom[::-1][:2]
            logger.info("Loading {} sagittal with input voxelsize {}".format(modalitie,self.zoom))

        elif self.plane == "coronal":
            orig_data = transform_axial2coronal(orig_data)
            self.zoom = orig_zoom[1:]
            logger.info("Loading {} coronal with input voxelsize {}".format(modalitie,self.zoom))

        else:
            self.zoom = orig_zoom[:2]
            logger.info("Loading {} axial with input voxelsize {}".format(modalitie,self.zoom))

        # Create thick slices
        orig_thick = get_thick_slices(orig_data, self.slice_thickness)

        return orig_thick
    def _get_scale_factor(self):
        """
        Get scaling factor to match original resolution of input image to
        final resolution of FastSurfer base network. Input resolution is
        taken from voxel size in image header.
        ToDO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.
        :param img_zoom:
        :return np.ndarray(float32): scale factor along x and y dimension
        """
        scale = self.base_res / np.asarray(self.zoom)

        return scale

    def __getitem__(self, index):
        img = self.images[index]

        scale_factor = self._get_scale_factor()
        if self.transforms is not None:
            img = self.transforms(img)

        return {'image': img, 'scale_factor': scale_factor,'weight_factor' : self.weight_factor}

    def __len__(self):
        return self.count

