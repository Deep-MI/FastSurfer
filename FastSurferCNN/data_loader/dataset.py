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
import time
from typing import Optional, Tuple, Dict

import h5py
import numpy as np
import numpy.typing as npt
import torch
import torchvision
import yacs.config
from torch.utils.data import Dataset
import torchio as tio

from FastSurferCNN.data_loader import data_utils as du
from FastSurferCNN.utils import logging

logger = logging.getLogger(__name__)


# Operator to load imaged for inference
class MultiScaleOrigDataThickSlices(Dataset):
    """Load MRI-Image and process it to correct format for network inference."""

    def __init__(
            self,
            orig_data: npt.NDArray,
            orig_zoom: npt.NDArray,
            cfg: yacs.config.CfgNode,
            transforms: Optional = None
    ):
        """Construct object.

        Parameters
        ----------
        orig_data : npt.NDArray
            Orignal Data
        orig_zoom : npt.NDArray
            Original zoomfactors
        cfg : yacs.config.CfgNode
            Configuration Node
        transforms : Optional
            Transformer for the image. Defaults to None

        """
        assert (
                orig_data.max() > 0.8
        ), f"Multi Dataset - orig fail, max removed {orig_data.max()}"
        self.plane = cfg.DATA.PLANE
        self.slice_thickness = cfg.MODEL.NUM_CHANNELS // 2
        self.base_res = cfg.MODEL.BASE_RES

        if self.plane == "sagittal":
            orig_data = du.transform_sagittal(orig_data)
            self.zoom = orig_zoom[::-1][:2]
            logger.info("Loading Sagittal with input voxelsize {}".format(self.zoom))

        elif self.plane == "axial":
            orig_data = du.transform_axial(orig_data)
            self.zoom = orig_zoom[::-1][:2]
            logger.info("Loading Axial with input voxelsize {}".format(self.zoom))

        else:
            self.zoom = orig_zoom[:2]
            logger.info("Loading Coronal with input voxelsize {}".format(self.zoom))

        # Create thick slices
        orig_thick = du.get_thick_slices(orig_data, self.slice_thickness)
        orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
        self.images = orig_thick
        self.count = self.images.shape[0]
        self.transforms = transforms

    def _get_scale_factor(self) -> npt.NDArray[float]:
        """Get scaling factor to match original resolution of input image to final resolution of FastSurfer base network.

        Input resolution is taken from voxel size in image header.
        ToDO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.

        Returns
        -------
        npt.NDArray[float]
            scale factor along x and y dimension
        
        """
        scale = self.base_res / np.asarray(self.zoom)

        return scale

    def __getitem__(self, index: int) -> Dict:
        """Return a single image and its scale factor.

        Parameters
        ----------
        index : int
            Index of image to get

        Returns
        -------
        dict
            Dictionary of image and scale factor

        """
        img = self.images[index]

        scale_factor = self._get_scale_factor()
        if self.transforms is not None:
            img = self.transforms(img)

        return {"image": img, "scale_factor": scale_factor}

    def __len__(self) -> int:
        """Return length.

        Returns
        -------
        int
            count
        """
        return self.count


# Operator to load hdf5-file for training
class MultiScaleDataset(Dataset):
    """Class for loading aseg file with augmentations (transforms)."""

    def __init__(
            self,
            dataset_path: str,
            cfg: yacs.config.CfgNode,
            gn_noise: bool = False,
            transforms: Optional = None
    ):
        """Construct object.

        Parameters
        ----------
        dataset_path : str
            Path to the dataset
        cfg : yacs.config.CfgNode
            Configuration node
        gn_noise : bool
            Whether to add gaussian noise (Default value = False)
        transforms : Optional
            Transformer to apply to the image (Default value = None)

        """
        self.max_size = cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES
        self.gn_noise = gn_noise

        # Load the h5 file and save it to the datase
        self.images = []
        self.labels = []
        self.weights = []
        self.subjects = []
        self.zooms = []

        # Open file in reading mode
        start = time.time()
        with h5py.File(dataset_path, "r") as hf:
            for size in cfg.DATA.SIZES:
                try:
                    logger.info(f"Processing images of size {size}.")
                    img_dset = list(hf[f"{size}"]["orig_dataset"])
                    logger.info(
                        "Processed origs of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.images.extend(img_dset)
                    self.labels.extend(list(hf[f"{size}"]["aseg_dataset"]))
                    logger.info(
                        "Processed asegs of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.weights.extend(list(hf[f"{size}"]["weight_dataset"]))
                    self.zooms.extend(list(hf[f"{size}"]["zoom_dataset"]))
                    logger.info(
                        "Processed zooms of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    logger.info(
                        "Processed weights of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.subjects.extend(list(hf[f"{size}"]["subject"]))
                    logger.info(
                        "Processed subjects of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    logger.info(f"Number of slices for size {size} is {len(img_dset)}")

                except KeyError as e:
                    print(
                        f"KeyError: Unable to open object (object {size} does not exist)"
                    )
                    continue

            self.count = len(self.images)
            self.transforms = transforms

            logger.info(
                "Successfully loaded {} data from {} with plane {} in {:.3f} seconds".format(
                    self.count, dataset_path, cfg.DATA.PLANE, time.time() - start
                )
            )

    def get_subject_names(self):
        """Get the subject name.

        Returns
        -------
        list
            list of subject names
        """
        return self.subjects

    def _get_scale_factor(
            self,
            img_zoom: torch.Tensor,
            scale_aug: torch.Tensor
    ) -> npt.NDArray[float]:
        """Get scaling factor to match original resolution of input image to final resolution of FastSurfer base network.

        Input resolution is taken from voxel size in image header.
        
        ToDO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.

        Parameters
        ----------
        img_zoom : torch.Tensor
            Image zoom factor
        scale_aug : torch.Tensor
            [MISSING]

        Returns
        -------
        npt.NDArray[float]
            scale factor along x and y dimension
        
        """
        if torch.all(scale_aug > 0):
            img_zoom *= 1 / scale_aug

        scale = self.base_res / img_zoom

        if self.gn_noise:
            scale += (
                torch.randn(1) * 0.1 + 0
            )  # needs to be changed to torch.tensor stuff
            scale = torch.clamp(scale, min=0.1)

        return scale

    def _pad(
            self,
            image: npt.NDArray
    ) ->  np.ndarray:
        """Pad the image with zeros.

        Parameters
        ----------
        image : npt.NDArray
            Image to pad

        Returns
        -------
        padded_image
            Padded image

        """
        if len(image.shape) == 2:
            h, w = image.shape
            padded_img = np.zeros((self.max_size, self.max_size), dtype=image.dtype)
        else:
            h, w, c = image.shape
            padded_img = np.zeros((self.max_size, self.max_size, c), dtype=image.dtype)

        if self.max_size < h:
            sub = h - self.max_size
            padded_img = image[0 : h - sub, 0 : w - sub]
        else:
            padded_img[0:h, 0:w] = image

        return padded_img

    def unify_imgs(
            self,
            img: npt.NDArray,
            label: npt.NDArray,
            weight: npt.NDArray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pad img, label and weight.

        Parameters
        ----------
        img : npt.NDArray
            image to unify
        label : npt.NDArray
            labels of the image
        weight : npt.NDArray
            weights of the image

        Returns
        -------
        np.ndarray
            img
        np.ndarray
            label
        np.ndarray
            weight
        
        """
        img = self._pad(img)
        label = self._pad(label)
        weight = self._pad(weight)

        return img, label, weight

    def __getitem__(self, index):
        """[MISSING].

        Parameters
        ----------
        index :
            [MISSING]

        Returns
        -------
        [MISSING]

        """
        padded_img, padded_label, padded_weight = self.unify_imgs(
            self.images[index], self.labels[index], self.weights[index]
        )
        img = np.expand_dims(padded_img.transpose((2, 0, 1)), axis=3)
        label = padded_label[np.newaxis, :, :, np.newaxis]
        weight = padded_weight[np.newaxis, :, :, np.newaxis]

        subject = tio.Subject(
            {
                "img": tio.ScalarImage(tensor=img),
                "label": tio.LabelMap(tensor=label),
                "weight": tio.LabelMap(tensor=weight),
            }
        )

        zoom_aug = torch.as_tensor([0.0, 0.0])

        if self.transforms is not None:
            tx_sample = self.transforms(subject)  # this returns data as torch.tensors

            img = torch.squeeze(tx_sample["img"].data).float()
            label = torch.squeeze(tx_sample["label"].data).byte()
            weight = torch.squeeze(tx_sample["weight"].data).float()

            # get updated scalefactor, in case of scaling, not ideal - fails if scales is not in dict
            rep_tf = tx_sample.get_composed_history()
            if rep_tf:
                zoom_aug += torch.as_tensor(
                    rep_tf[0]._get_reproducing_arguments()["scales"]
                )[:-1]

            # Normalize image and clamp between 0 and 1
            img = torch.clamp(img / img.max(), min=0.0, max=1.0)

        scale_factor = self._get_scale_factor(
            torch.from_numpy(self.zooms[index]), scale_aug=zoom_aug
        )

        return {
            "image": img,
            "label": label,
            "weight": weight,
            "scale_factor": scale_factor,
        }

    def __len__(self):
        """Return count."""
        return self.count


# Operator to load hdf5-file for validation
class MultiScaleDatasetVal(Dataset):
    """Class for loading aseg file with augmentations (transforms)."""

    def __init__(self, dataset_path, cfg, transforms=None):

        self.max_size = cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES

        # Load the h5 file and save it to the dataset
        self.images = []
        self.labels = []
        self.weights = []
        self.subjects = []
        self.zooms = []

        # Open file in reading mode
        start = time.time()
        with h5py.File(dataset_path, "r") as hf:
            for size in cfg.DATA.SIZES:
                try:
                    logger.info(f"Processing images of size {size}.")
                    img_dset = list(hf[f"{size}"]["orig_dataset"])
                    logger.info(
                        "Processed origs of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.images.extend(img_dset)
                    self.labels.extend(list(hf[f"{size}"]["aseg_dataset"]))
                    logger.info(
                        "Processed asegs of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.weights.extend(list(hf[f"{size}"]["weight_dataset"]))
                    logger.info(
                        "Processed weights of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.zooms.extend(list(hf[f"{size}"]["zoom_dataset"]))
                    logger.info(
                        "Processed zooms of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    self.subjects.extend(list(hf[f"{size}"]["subject"]))
                    logger.info(
                        "Processed subjects of size {} in {:.3f} seconds".format(
                            size, time.time() - start
                        )
                    )
                    logger.info(f"Number of slices for size {size} is {len(img_dset)}")

                except KeyError as e:
                    print(
                        f"KeyError: Unable to open object (object {size} does not exist)"
                    )
                    continue

        self.count = len(self.images)
        self.transforms = transforms
        logger.info(
            "Successfully loaded {} data from {} with plane {} in {:.3f} seconds".format(
                self.count, dataset_path, cfg.DATA.PLANE, time.time() - start
            )
        )

    def get_subject_names(self):
        """Get subject names."""
        return self.subjects

    def _get_scale_factor(self, img_zoom):
        """Get scaling factor to match original resolution of input image to final resolution of FastSurfer base network.

        Input resolution is taken from voxel size in image header.
        
        ToDO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.

        Parameters
        ----------
        img_zoom :
            zooming factor [MISSING]

        Returns
        -------
        np.ndarray : float32
            scale factor along x and y dimension

        """
        scale = self.base_res / img_zoom
        return scale

    def __getitem__(self, index):
        """Get item."""
        img = self.images[index]
        label = self.labels[index]
        weight = self.weights[index]
        scale_factor = self._get_scale_factor(self.zooms[index])

        if self.transforms is not None:
            tx_sample = self.transforms(
                {
                    "img": img,
                    "label": label,
                    "weight": weight,
                    "scale_factor": scale_factor,
                }
            )

            img = tx_sample["img"]
            label = tx_sample["label"]
            weight = tx_sample["weight"]
            scale_factor = tx_sample["scale_factor"]

        return {
            "image": img,
            "label": label,
            "weight": weight,
            "scale_factor": scale_factor,
        }

    def __len__(self):
        """Get count."""
        return self.count
