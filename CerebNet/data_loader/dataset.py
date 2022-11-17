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
import time
# IMPORTS
from typing import Sequence, Tuple, Literal, get_args as _get_args, TypeVar, Dict
from numbers import Number

import nibabel as nib
import torch
import numpy as np
from numpy import typing as npt
import h5py
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose

from CerebNet.data_loader.data_utils import Plane
from FastSurferCNN.utils import logging
from FastSurferCNN.data_loader.data_utils import get_thick_slices, transform_axial, transform_sagittal

from CerebNet.data_loader import data_utils as utils
from CerebNet.data_loader.augmentation import ToTensor
from CerebNet.datasets.load_data import SubjectLoader
from CerebNet.datasets.utils import crop_transform, bounding_volume_offset

ROIKeys = Literal['source_shape', 'offsets', 'target_shape']
LocalizerROI = Dict[ROIKeys, Tuple[int, ...]]

NT = TypeVar('NT', bound=Number)
PLANES = _get_args(Plane)

logger = logging.get_logger(__name__)


# Operator to load hdf5-file for training
class CerebDataset(Dataset):
    """
    Class for loading aseg file with augmentations (transforms)
    """
    def __init__(self, dataset_path, cfg, transforms, load_aux_data):

        # Load the h5 file and save it to the dataset
        # try:
        self.cfg = cfg
        self.slice_thickness = cfg.DATA.THICKNESS
        self.transforms = transforms
        self.load_talairach = cfg.DATA.LOAD_TALAIRACH
        plane_transform = utils.get_plane_transform(cfg.DATA.PLANE,
                                                    cfg.DATA.PRIMARY_SLICE_DIR)
        self.dataset = {}

        # TODO may be need to load warped data to memory as needed
        with h5py.File(dataset_path, "r") as hf:
            for name in hf.keys():
                if 'auxiliary' in name:
                    if not load_aux_data:
                        continue
                    stride = int(1.0 / cfg.DATA.FRACTION)
                    data = np.array(hf.get(name)[::stride])
                    logger.info(f"Loaded {data.shape[0]} {name} with stride {stride}")
                else:
                    if self.cfg.DEBUG:
                        data = np.array(hf.get(name)[:3])
                    else:
                        data = np.array(hf.get(name)[:])

                if name != 'subject':
                    data = plane_transform(data)
                    data = self._stack_slices_in_plane(data,
                                                       thick_slice=(name == 'img' or name == 'auxiliary_img'))  # [n_slices, h, w]
                    if name == 'label' or name == 'auxiliary_lbl':
                        if cfg.DATA.PLANE == 'sagittal':
                            data = utils.map_sag2label(data)

                self.dataset[name] = data

        if load_aux_data:
            logger.info(f"Using {100 * cfg.DATA.FRACTION}% ({self.dataset['auxiliary_img'].shape[0]} slices)"
                        f" of auxiliary data.")
            self.dataset['img'] = np.concatenate([self.dataset['img'], self.dataset['auxiliary_img']], axis=0)
            self.dataset['label'] = np.concatenate([self.dataset['label'], self.dataset['auxiliary_lbl']], axis=0)
            del self.dataset['auxiliary_img']
            del self.dataset['auxiliary_lbl']

        self.class_wise_weights = self._get_class_wise_weights(self.dataset['label'])
        utils.filter_blank_slices_thick(self.dataset)

        self.slice_thickness = cfg.DATA.THICKNESS
        self.count = self.dataset['img'].shape[0]
        # self.warped_count = self.dataset['warped_img'].shape[0] if 'warped_img' in self.dataset else 0
        assert self.count >= self.slice_thickness, f"Not enough slices {self.count}" \
                                                   f" for the given slice thickness {self.slice_thickness}"

        self.subjects = self.dataset['subject']
        del self.dataset['subject']

        logger.info("Successfully loaded {} slices in {} plane from {}".format(self.count,
                                                                                 cfg.DATA.PLANE,
                                                                                 dataset_path,
                                                               ))

        logger.info("Total number of classes is: {}".format(len(np.unique(self.dataset['label']))))

        # except Exception as e:
        #     logger.info("Loading failed: {}".format(e))

    def get_subject_names(self):
        return self.subjects

    def _get_class_wise_weights(self, label_map, max_weight=5):
        unique, counts = np.unique(label_map, return_counts=True)

        # Median Frequency Balancing
        class_wise_weights = np.median(counts) / counts
        class_wise_weights[class_wise_weights > max_weight] = max_weight

        return class_wise_weights

    def _stack_slices_in_plane(self, vol, thick_slice):
        """
            vol: [N, H, W, D]  N  images with plane at last dimension
        """
        if len(vol.shape) == 4:
            vol = np.moveaxis(vol, [0, 1, 2, 3], [0, 2, 3, 1])
            if thick_slice:
                vol = get_thick_slices(vol, self.slice_thickness)
                n_imgs, n_slices, thickness, h, w = vol.shape
                return vol.reshape(n_imgs*n_slices, thickness, h, w)
            n_imgs, n_slices, h, w = vol.shape
            return vol.reshape(n_imgs * n_slices, h, w)
        if len(vol.shape) == 5:
            vol = np.moveaxis(vol, [0, 1, 2, 3, 4], [0, 2, 3, 1, 4])
            n_imgs, n_slices, h, w, c = vol.shape
            vol = vol.reshape(n_imgs * n_slices, h, w, c)
            return np.moveaxis(vol, [0, 1, 2, 3], [0, 2, 3, 1])

    def __getitem__(self, index):
        sample = {}
        sample['image'] = self.dataset['img'][index]
        sample['label'] = self.dataset['label'][index]
        if 'talairach' in self.dataset:
            sample['talairach'] = self.dataset['talairach'][index]

        if self.transforms is not None:
            sample = self.transforms(sample)
        sample['weight'] = utils.create_weight_mask2d(sample['label'], self.class_wise_weights)

        if 'talairach' in sample:
            sample['image'] = np.concatenate((sample['image'], sample['talairach']), axis=0)
            del sample['talairach']
        elif self.load_talairach: ## for validation use zeros instead
            pad_width = self.cfg.MODEL.NUM_CHANNELS - sample['image'].shape[0]
            size = torch.Size([pad_width]) + sample['image'].shape[1:]
            zero_pads = torch.zeros(size, dtype=sample['image'].dtype)
            sample['image'] = torch.cat((zero_pads, sample['image']), dim=0)
        return sample

    def __len__(self):
        return self.count


class TestLoader(Dataset):
    def __init__(self,
                 subject_path,
                 data_cfg,
                 transforms=None):

        data_dict = self._load_images(data_cfg, subject_path)
        self.img_per_plane = data_dict['image']
        self.labels = data_dict['label']
        self.meta_data = data_dict['meta']
        self.transform = transforms
        self.count = self.img_per_plane['axial'].shape[0]

    def _load_images(self, cfg, subj_path):
        subject_loader = SubjectLoader(cfg)
        data_dict = subject_loader.load_test_subject(subj_path)
        return data_dict

    def get_img_metadata(self):
        return self.meta_data

    def get_orig(self):
        return self.img_per_plane

    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        out_dict = {}
        for plane in ['axial', 'coronal', 'sagittal']:
            img = self.img_per_plane[plane][index]
            if self.transform is not None:
                img = self.transform(img)
            out_dict[plane] = img
        return out_dict

    def __len__(self):
        return self.count


class SubjectDataset(Dataset):
    """
    Single subject loader to load and prepare slices for network process.

    """

    roi = LocalizerROI

    def __init__(self,
                 img_org: nib.analyze.SpatialImage,
                 brain_seg: nib.analyze.SpatialImage,
                 patch_size: Tuple[int, ...],
                 slice_thickness: int,
                 primary_slice: str):
        self.slice_thickness = slice_thickness
        self.transforms = Compose([ToTensor()])
        self.img_org = img_org
        self.img_org_data = np.asarray(img_org.dataobj)

        self.brain_seg = brain_seg

        # binarize the cerebellum from brain_seg
        cereb_aseg_mask = utils.get_aseg_cereb_mask(np.asarray(brain_seg.dataobj))

        from numpy.linalg import inv
        affine = inv(brain_seg.affine) @ img_org.affine

        #print(brain_seg.affine, img_org.affine)
        if not np.allclose(affine, np.eye(affine.shape[0])):
            logger.info("The conformed image and the segmentation do not share the same affine. The cerebellum mask "
                        "is being resampled to localize it in the conformed image.")
            from scipy.ndimage import affine_transform
            cereb_aseg = affine_transform(cereb_aseg_mask.astype(np.float32), affine, output_shape=img_org.shape)

            # TODO remove this save statement, if the cereb_aseg_float_mask lines up with the conformed image
            #from FastSurferCNN.data_loader.data_utils import save_image
            #save_image(img_org.header, img_org.affine, cereb_aseg, "/tmp/cereb_mask_on_conformed.mgz", dtype=np.float32)

            cereb_aseg_mask = cereb_aseg > 0.5

        bbox = self.locate_mask_bbox(cereb_aseg_mask)

        # create the roi from cereb_aseg (where labels after interpolation > 0.05 --> membership rounded to 1 decimal)
        self.roi: LocalizerROI = {"source_shape": img_org.shape,
                                  "offsets": bounding_volume_offset(bbox, patch_size, image_shape=cereb_aseg_mask.shape),
                                  "target_shape": patch_size}
        # crop the region of interest
        img = crop_transform(self.img_org_data, offsets=self.roi["offsets"], target_shape=self.roi["target_shape"])

        self.images_per_plane = {}
        self.count = 0
        self._plane: Plane = "axial"
        data = {"axial": transform_axial(img), "coronal": img, "sagittal": transform_sagittal(img)}
        for plane, data in data.items():
            # data is transformed to 'plane'-direction in axis 2
            thick_slices = get_thick_slices(data, self.slice_thickness)  # [H, W, n_slices, C]
            # it seems x and y are flipped with respect to expectations here
            self.images_per_plane[plane] = np.transpose(thick_slices, (2, 0, 1, 3))   # [n_slices, H, W, C]

    def locate_mask_bbox(self, mask: npt.NDArray[bool]):
        """Find the largest connected component of the mask.

        Returns:
            bbox of min0, min1, ..., max0, max1, ...
        """
        # filter disconnected components
        from skimage.measure import regionprops, label
        label_image = label(mask, connectivity=3)
        regions = regionprops(label_image)
        largest_region = np.argmax([r.area for r in regions])
        return regions[largest_region].bbox

    def get_nibabel_img(self):
        return self.img_org

    def get_bounding_offsets(self) -> LocalizerROI:
        return self.roi

    def set_plane(self, plane: Plane):
        """Set the active plane."""
        if plane not in self.images_per_plane.keys():
            raise ValueError(f"Invalid plane name, must be in {tuple(self.images_per_plane.keys())}")
        self._plane = plane

    @property
    def plane(self) -> Plane:
        """Returns the active plane"""
        return self._plane

    def __getitem__(self, index: int) -> Tuple[Plane, np.ndarray]:
        """Get the plane and data belonging to indices given."""

        if not (0 <= index < self.images_per_plane[self.plane].shape[0]):
            raise IndexError(f"Index out of bounds, for active plane {self.plane}. "
                             f"Index should be within [0, {self.images_per_plane[self.plane].shape[0]}).")
        img = self.images_per_plane[self.plane][index]
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __len__(self) -> int:
        return self.images_per_plane[self.plane].shape[0]
