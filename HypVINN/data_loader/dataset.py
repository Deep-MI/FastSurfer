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
from numpy import typing as npt
import torch
from torch.utils.data import Dataset


from HypVINN.data_loader.data_utils import transform_axial2sagittal,transform_axial2coronal
from FastSurferCNN.data_loader.data_utils import get_thick_slices

import FastSurferCNN.utils.logging as logging
from HypVINN.utils import ModalityDict, ModalityMode

logger = logging.get_logger(__name__)


# Operator to load imaged for inference
class HypVINNDataset(Dataset):
    """
    Class to load MRI-Image and process it to correct format for HypVINN network inference.

    The HypVINN Dataset passed during Inference the input images,the scale factor for the VINN layer and a weight factor
    (wT1,wT2).
    The Weight factor determines the running mode of the HypVINN model.
    if wT1 =1 and wT2 =0. The HypVINN model will only allow the flow of the T1 information (mode = t1).
    if wT1 =0 and wT2 =1. The HypVINN model will only allow the flow of the T2 information (mode = t2).
    if wT1 !=1 and wT2 !=1. The HypVINN model will automatically weigh the T1 information and the T2 information based
    on the learned modality weights (mode = t1t2).

    Methods
    -------
    _standarized_img(orig_data: np.ndarray, orig_zoom: npt.NDArray[float], modality: np.ndarray) -> np.ndarray
        Standardize the image based on the original data, original zoom, and modality.
    _get_scale_factor() -> npt.NDArray[float]
        Get the scaling factor to match the original resolution of the input image to the final resolution of the
        FastSurfer base network.
    __getitem__(index: int) -> dict[str, torch.Tensor | np.ndarray]
        Retrieve the image, scale factor, and weight factor for a given index.
    __len__()
        Return the number of images in the dataset.
    """
    def __init__(
            self,
            subject_name: str,
            modalities: ModalityDict,
            orig_zoom: npt.NDArray[float],
            cfg,
            mode: ModalityMode = "t1t2",
            transforms=None,
    ):
        """
        Initialize the HypVINN Dataset.

        Parameters
        ----------
        subject_name : str
            The name of the subject.
        modalities : ModalityDict
            The modalities of the subject.
        orig_zoom : npt.NDArray[float]
            The original zoom of the subject.
        cfg : CfgNode
            The configuration object.
        mode : ModalityMode, default="t1t2"
            The running mode of the HypVINN model. (Default: "t1t2").
        transforms : Callable, optional
            The transformations to apply to the images. (Default: None).

        """
        self.subject_name = subject_name
        self.plane = cfg.DATA.PLANE
        #Inference Mode
        self.mode = mode
        #set thickness base on train paramters
        if cfg.MODEL.MODE in ["t1", "t2"]:
            self.slice_thickness = cfg.MODEL.NUM_CHANNELS//2
        else:
            self.slice_thickness = cfg.MODEL.NUM_CHANNELS//4

        self.base_res = cfg.MODEL.BASE_RES

        if self.mode == "t1":
            orig_thick = self._standarized_img(modalities["t1"], orig_zoom, modality="t1")
            orig_thick = np.concatenate((orig_thick, orig_thick), axis=-1)
            self.weight_factor = torch.from_numpy(np.asarray([1.0, 0.0]))

        elif self.mode == "t2":
            orig_thick = self._standarized_img(modalities["t2"], orig_zoom, modality="t2")
            orig_thick = np.concatenate((orig_thick, orig_thick), axis=-1)
            self.weight_factor = torch.from_numpy(np.asarray([0.0, 1.0]))
        else:
            t1_orig_thick = self._standarized_img(modalities["t1"], orig_zoom, modality="t1")
            t2_orig_thick = self._standarized_img(modalities["t2"], orig_zoom, modality="t2")
            orig_thick = np.concatenate((t1_orig_thick, t2_orig_thick), axis=-1)
            self.weight_factor = torch.from_numpy(np.asarray([0.5, 0.5]))
        
        # Transpose from W,H,N,C to N,W,H,C
        orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
        self.images = orig_thick
        self.count = self.images.shape[0]
        self.transforms = transforms

        logger.info(
            f"Successfully loaded Image from {subject_name} for {self.plane} "
            f"model"
        )

        if ((cfg.MODEL.MULTI_AUTO_W or cfg.MODEL.MULTI_AUTO_W_CHANNELS) and
                (self.mode == 't1t2' or cfg.MODEL.DUPLICATE_INPUT)) :
            logger.info(
                f"For inference T1 block weight and the T2 block are set to "
                f"the weights learn during training"
            )
        else:
            logger.info(
                f"For inference T1 block weight was set to: "
                f"{self.weight_factor.numpy()[0]} and the T2 block was set to: "
                f"{self.weight_factor.numpy()[1]}")

    def _standarized_img(self, orig_data: np.ndarray, orig_zoom: npt.NDArray[float],
                         modality: np.ndarray) -> np.ndarray:
        """
        Standardize the image based on the original data, original zoom, and modality.

        Parameters
        ----------
        orig_data : np.ndarray
            The original data of the image.
        orig_zoom : npt.NDArray[float]
            The original zoom of the image.
        modality : np.ndarray
            The modality of the image.

        Returns
        -------
        orig_thick : np.ndarray
            The standardized image.
        """
        if self.plane == "sagittal":
            orig_data = transform_axial2sagittal(orig_data)
            self.zoom = orig_zoom[::-1][:2]
            logger.info(
                f"Loading {modality} sagittal with input voxelsize {self.zoom}"
            )

        elif self.plane == "coronal":
            orig_data = transform_axial2coronal(orig_data)
            self.zoom = orig_zoom[1:]
            logger.info(
                f"Loading {modality} coronal with input voxelsize {self.zoom}"
            )

        else:
            self.zoom = orig_zoom[:2]
            logger.info(
                f"Loading {modality} axial with input voxelsize {self.zoom}"
            )

        # Create thick slices
        orig_thick = get_thick_slices(orig_data, self.slice_thickness)

        return orig_thick

    def _get_scale_factor(self) -> npt.NDArray[float]:
        """
        Get the scaling factor to match the original resolution of the input image to
        the final resolution of the FastSurfer base network. The input resolution is
        taken from the voxel size in the image header.

        Returns
        -------
        scale : npt.NDArray[float]
            The scaling factor along the x and y dimensions. This is a numpy array of float values.
        """
        # TODO: This needs to be updated based on the plane we are looking at in case we
        #  are dealing with non-isotropic images as inputs.

        scale = self.base_res / np.asarray(self.zoom)

        return scale

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | np.ndarray]:
        """
        Retrieve the image, scale factor, and weight factor for a given index.

        This method retrieves the image at the given index from the images attribute, calculates the scale factor,
        applies any transformations to the image if they are defined, and returns a dictionary containing the image,
        scale factor, and weight factor.

        Parameters
        ----------
        index : int
            The index of the image to retrieve.

        Returns
        -------
        dict[str, torch.Tensor | np.ndarray]
            A dictionary containing the image, scale factor, and weight factor.
        """
        img = self.images[index]

        scale_factor = self._get_scale_factor()
        if self.transforms is not None:
            img = self.transforms(img)

        return {
            "image": img,
            "scale_factor": scale_factor,
            "weight_factor": self.weight_factor,
        }

    def __len__(self):
        return self.count

