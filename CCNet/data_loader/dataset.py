
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

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio
import copy

from CCNet.data_loader import data_utils as du
from CCNet.utils import logging

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod

class VINNDataset(Dataset, ABC):

    @abstractmethod
    def __init__(self):
        self.images = []
        self.labels = []
        self.weights = []
        self.subjects = []
        self.zooms = []
        self.default_size = []
        self.com = []
        self.localisation = False

        raise NotImplementedError("InferenceDataset is an abstract class and should not be instantiated")

    def unify_imgs(self, input_list, padsize = None):
        if padsize is None:
            padsize = [self.default_size for _ in input_list]

        output = []
        for i in range(len(input_list)):
            output.append(self._pad(input_list[i], padsize[i]))
        return output

    def _pad(self, image, padsize = None):
        if padsize is None:
            padsize = self.default_size

        dif = lambda i : np.max([0, (padsize[i] - image.shape[i])])
        if len(image.shape) == 2:
            padded_img = np.pad(image, pad_width=((0, dif(0)), (0, dif(1))), mode='constant', constant_values = 0)
            print("")
        else:
            padded_img = np.pad(image, pad_width=((0, dif(0)), (0, dif(1)), (0, dif(2))), mode='constant', constant_values = 0)

        return padded_img

    def _get_scale_factor(self, img_zoom):
        """
        Get scaling factor to match original resolution of input image to
        final resolution of FastSurfer base network. Input resolution is
        taken from voxel size in image header.

        TODO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.
        :param img_zoom:
        :return np.ndarray(float32): scale factor along x and y dimension
        """
        scale = self.base_res / img_zoom
        return scale

    def load_subset(self, hf, size, num_slices=5):
        """
        Same as load data, but only loads a few slices per resolution for faster debugging

        :param num_slices: number of slices to load per resolution
        """
        #start = time.time()
        #logger.info(f"Processing images of size {size}.")
        
        self.images.extend(list(hf[f'{size}']['orig_dataset'][:num_slices]))
        #logger.info("Processed origs of size {} in {:.3f} seconds".format(size, time.time()-start))

        if self.localisation:
            # load comissures
            self.com.extend(list(hf[f'{size}']['center_dataset'][:num_slices]))
            #logger.info("Processed comissures of size {} in {:.3f} seconds".format(size, time.time() - start))

        self.labels.extend(list(hf[f'{size}']['aseg_dataset'][:num_slices]))
        #logger.info("Processed asegs of size {} in {:.3f} seconds".format(size, time.time()-start))

        self.weights.extend(list(hf[f'{size}']['weight_dataset'][:num_slices]))
        #logger.info("Processed weights of size {} in {:.3f} seconds".format(size, time.time() - start))

        self.zooms.extend(list(hf[f'{size}']['zoom_dataset'][:num_slices]))
        #logger.info("Processed zooms of size {} in {:.3f} seconds".format(size, time.time() - start))
        
        self.subjects.extend(list(hf[f'{size}']['subject'][:num_slices]))
        #logger.info("Processed subjects of size {} in {:.3f} seconds".format(size, time.time()-start))

        assert len(self.images) == len(self.labels) == len(self.weights) == len(self.zooms) == len(self.subjects), "Number of images, labels, weights, zooms and subjects are not equal"
        logger.info(f"New number of slices is {len(self.images)}")

        # load max size
        max = [self.default_size]
        max.append(np.max([image.shape for image in self.images], axis=0))
        self.pad_shape_images = np.max(max, axis=0)

        max = [self.default_size[:2]]
        max.append(np.max([label.shape for label in self.labels], axis=0))
        self.pad_shape_labels = np.max(max, axis=0)

        max = [self.default_size[:2]]
        max.append(np.max([weight.shape for weight in self.weights], axis=0))
        self.pad_shape_weights = np.max(max, axis=0)
    
    def load_data(self, hf, size):
        """
        Load data from h5 file
        :param hf: h5 file
        :param size: size of the data
        :return:
        """
        start = time.time()
        self.zooms.extend(list(hf[f'{size}']['zoom_dataset']))
        logger.info("Processed zooms of size {} in {:.3f} seconds".format(size, time.time() - start))

        start = time.time()
        logger.info(f"Processing images of size {size}.")
        
        self.images.extend(list(hf[f'{size}']['orig_dataset']))
        logger.info("Processed origs of size {} in {:.3f} seconds".format(size, time.time() - start))
        
        if self.localisation:
            # load comissures
            self.com.extend(list(hf[f'{size}']['center_dataset']))
            logger.info("Processed comissures of size {} in {:.3f} seconds".format(size, time.time() - start))
        
        self.labels.extend(list(hf[f'{size}']['aseg_dataset']))
        logger.info("Processed asegs of size {} in {:.3f} seconds".format(size, time.time() - start))

        self.weights.extend(list(hf[f'{size}']['weight_dataset']))
        logger.info("Processed weights of size {} in {:.3f} seconds".format(size, time.time() - start))

        self.subjects.extend(list(hf[f'{size}']['subject']))
        logger.info("Processed subjects of size {} in {:.3f} seconds".format(size, time.time() - start))

        logger.info(f"Number of slices for size {size} is {len(self.images)}")

        # load max size
        max = [self.default_size]
        max.append(np.max([image.shape for image in self.images], axis=0))
        self.pad_shape_images = np.max(max, axis=0)

        max = [self.default_size[:2]]
        max.append(np.max([label.shape for label in self.labels], axis=0))
        self.pad_shape_labels = np.max(max, axis=0)

        max = [self.default_size[:2]]
        max.append(np.max([weight.shape for weight in self.weights], axis=0))
        self.pad_shape_weights = np.max(max, axis=0)

    def __len__(self):
        return len(self.images)

    def get_subject_names(self):
        return self.subjects

    def __getitem__(self, index) -> dict:
        padded_img, padded_weight, padded_label = self.unify_imgs([self.images[index], self.weights[index], self.labels[index]], [self.pad_shape_images, self.pad_shape_weights, self.pad_shape_labels])

        padded_img = padded_img.transpose((2, 0, 1)) # move slice thickness to first spatial dimension
        padded_img = torch.clamp(torch.from_numpy(padded_img).float() / padded_img.max(), 0, 1) # TODO: if we keep the image as int the data augmentation it will be faster
        padded_weight = torch.from_numpy(padded_weight).float()
        padded_label = torch.from_numpy(padded_label)

        orig_img = padded_img[padded_img.shape[0]//2].detach().clone()

        scale_factor = self._get_scale_factor(torch.from_numpy(self.zooms[index]))

        return {'image': padded_img, 'label': padded_label, 'weight': padded_weight,
                'scale_factor': scale_factor, 'unmodified_center_slice': orig_img,
                'subject_id': self.subjects[index]}


# Operator to load images for inference
class MultiScaleOrigDataThickSlices(VINNDataset):
    """
    Class to load MRI-Image and process it to correct format for network inference
    """
    def __init__(self, img_filename, orig_data, orig_zoom, cfg, transforms=None, lesion_mask=None, pad = True):
        assert orig_data.max() > 0.8, f"Multi Dataset - orig fail, max removed {orig_data.max()}"
        self.img_filename = img_filename
        self.plane = cfg.DATA.PLANE
        self.slice_thickness = cfg.MODEL.NUM_CHANNELS//2
        self.base_res = cfg.MODEL.BASE_RES
        self.default_size = cfg.DATA.PADDED_SIZE

        if self.plane == "sagittal":
            orig_data = du.transform_sagittal(orig_data)
            self.zoom = orig_zoom[::-1][:2]
            logger.info("Loading Sagittal with input voxelsize {}".format(self.zoom))

            if lesion_mask is not None:
                lesion_mask = du.transform_sagittal(lesion_mask)
                logger.info("Loading Sagittal lesion with input voxelsize {}".format(self.zoom))

        elif self.plane == "axial":
            orig_data = du.transform_axial(orig_data)
            self.zoom = orig_zoom[::-1][:2]
            logger.info("Loading Axial with input voxelsize {}".format(self.zoom))

            if lesion_mask is not None:
                lesion_mask = du.transform_axial(lesion_mask)
                logger.info("Loading Axial lesion with input voxelsize {}".format(self.zoom))

        else:
            self.zoom = orig_zoom[:2]
            logger.info("Loading Coronal with input voxelsize {}".format(self.zoom))

            if lesion_mask is not None:
                logger.info("Loading Coronal lesion with input voxelsize {}".format(self.zoom))

        # Create thick slices
        orig_thick = du.get_thick_slices(orig_data, self.slice_thickness, pad = pad)
        orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
        self.images = orig_thick

        if lesion_mask is not None:
            lesion_thick = du.get_thick_slices(lesion_mask, self.slice_thickness)
            lesion_thick = np.transpose(lesion_thick, (2, 0, 1, 3))
            assert(lesion_thick.shape == orig_thick.shape), f"Lesion mask shape {lesion_thick.shape} does not match image shape {orig_thick.shape}"
            self.lesion_mask = lesion_thick

        self.transforms = transforms
        logger.info(f"Successfully loaded Image from {img_filename}")


    def __getitem__(self, index):
        img = self.images[index]
        if hasattr(self, 'lesion_mask'):
            lesion_mask = self.lesion_mask[index]
        else:
            lesion_mask = np.zeros(img.shape, dtype=bool)

        scale_factor = self._get_scale_factor(np.asarray(self.zoom))

        output = {'image': img, 'cutout_mask': lesion_mask, 'scale_factor': scale_factor}
        if self.transforms is not None:
            output = self.transforms(output)

        assert(output['cutout_mask'].shape == output['image'].shape), f"Cutout mask shape {output['cutout_mask'].shape} does not match image shape {output['image'].shape}"

        return output


# Operator to load hdf5-file for training
class MultiScaleDatasetAux(VINNDataset):
    """
    Class for loading aseg file with augmentations (transforms)
    """
    def __init__(self, dataset_path, cfg, scale_aug=False, transforms=None):
        self.default_size = cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES
        self.scale_aug = scale_aug

        # Check if we are dealing with a localisation task
        self.localisation = (cfg.MODEL.MODEL_NAME == "FastSurferLocalisation")

        # Load the h5 file and save it to the datase
        self.images = []
        self.labels = []
        self.weights = []
        self.aux_labels = []
        self.subjects = []
        self.zooms = []

        # Open file in reading mode        
        logger.info(f"Opening file {dataset_path} for reading...")
        assert(h5py.is_hdf5(dataset_path)), f"File \"{dataset_path}\" is not a valid hdf5 file."

        start = time.time()
        with h5py.File(dataset_path, "r") as hf:
            for size in cfg.DATA.SIZES:
                try:
                    if not cfg.TRAIN.DEBUG:
                        self.load_data(hf, size)
                    else:
                        self.load_subset(hf, size, num_slices=20)

                    logger.info("Successfully loaded {} participant volumes from h5 file".format(len(self.subjects)))

                    assert(len(self.images) == len(self.labels) == len(self.weights) == len(self.zooms) == len(self.aux_labels)), "Number of images, labels, weights, auxil and zooms are not equal."

                except KeyError as e:
                    print(f"KeyError: Unable to open object (object {size} does not exist)")
                    continue

            self.transforms = transforms

            logger.info("Successfully loaded {} slices from {} with plane {} in {:.3f} seconds".format(len(self.images), dataset_path, cfg.DATA.PLANE, time.time()-start))

    # override this function to enable scale augmentation (could also be outside of this function)
    def _get_scale_factor(self, img_zoom, scale_aug = torch.tensor(0.0)):
        """
        Get scaling factor to match original resolution of input image to
        final resolution of FastSurfer base network. Input resolution is
        taken from voxel size in image header.

        TODO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.
        :param img_zoom:
        :return np.ndarray(float32): scale factor along x and y dimension
        """
        if torch.all(scale_aug > 0):
            img_zoom *= (1 / scale_aug)

        scale = self.base_res / img_zoom

        if self.scale_aug:
            scale += torch.randn(1) * 0.1 + 0 # needs to be changed to torch.tensor stuff
            scale = torch.clamp(scale, min=0.1)

        return scale

    def apply_transforms(self, img, label, weight, aux_labels=None, cutout_mask=None, orig_slice=None, subject_id=None):
        """
        apply transforms to the image and labels
        :param img: image
        :param label: label
        :param weight: weight
        :return: transformed image, label and weight, and composed torchio history
        """
        img = img[None, ...] # add batch dimension for torchio
        label = label[None, ...]
        weight = weight[None, ...]
        if orig_slice is not None:
            orig_slice = orig_slice[None, ...]
        if cutout_mask is not None:
            cutout_mask = cutout_mask[None, ...]
        if aux_labels is not None:
            aux_labels = aux_labels[None, ...]

        # pad label and weight to match sptial dimensions of image
        slice_thickness = img.shape[1]
        label = np.pad(label, ((0, 0), (slice_thickness//2, slice_thickness//2), (0, 0), (0, 0)), 'constant', constant_values=0)
        weight = np.pad(weight, ((0, 0), (slice_thickness//2, slice_thickness//2), (0, 0), (0, 0)), 'constant', constant_values=0)
        #if cutout_mask is not None:
        #    cutout_mask = np.pad(cutout_mask, ((0, 0), (slice_thickness//2, slice_thickness//2), (0, 0), (0, 0)), 'constant', constant_values=0)
        #if aux_labels is not None:
        #    aux_labels = np.pad(aux_labels, ((0, 0), (slice_thickness//2, slice_thickness//2), (0, 0), (0, 0)), 'constant', constant_values=0)


        subject = tio.Subject({'image': tio.ScalarImage(tensor=img),
                               'label': tio.LabelMap(tensor=label),
                               'weight': tio.LabelMap(tensor=weight),
                               'unmodified_center_slice': tio.ScalarImage(tensor=orig_slice) if orig_slice is not None else None,
                               'cutout_mask': tio.LabelMap(tensor=cutout_mask), # TODO: this converts to UINT8, which is not what we want
                               'aux_label': tio.LabelMap(tensor=aux_labels),
                               'subject_id': subject_id
                               })
        tx_sample = self.transforms(subject) # this returns data as torch.tensors



        img = tx_sample['image'].data.float().squeeze(0)
        label = tx_sample['label'].data.byte()
        weight = tx_sample['weight'].data.float()
        if aux_labels:
            aux_labels = tx_sample['aux_label'].data.byte().squeeze(0)
            #aux_labels = aux_labels[:, slice_thickness//2, :, :].squeeze(0) # retrieve middle slice
        if orig_slice is not None:
            orig_slice = tx_sample['unmodified_center_slice'].data.float().squeeze(0)
        if 'cutout_mask' in tx_sample.keys():
            cutout_mask = tx_sample['cutout_mask'].data.bool().squeeze(0)
        else:
            cutout_mask = torch.zeros(orig_slice.size(), dtype=bool)

        label = label[:, slice_thickness//2, :, :].squeeze(0) # retrieve middle slice
        weight = weight[:, slice_thickness//2, :, :].squeeze(0)
        # if aux_labels is not None:
        #     aux_labels = aux_labels[:, slice_thickness//2, :, :].squeeze(0)

        return img, label, weight, aux_labels, orig_slice, cutout_mask, tx_sample.get_composed_history()


    def __getitem__(self, index):
        sample = super().__getitem__(index)
        img = sample['image']
        label = sample['label']
        weight = sample['weight']
        subject_id = sample['subject_id']

        if 'aux_data' in sample.keys():
            auxiliary_labels = sample['aux_data']
        else:
            auxiliary_labels = None

        if 'cutout_mask' in sample.keys():
             cutout_mask = sample['cutout_mask']
        elif 'aux_data' in sample.keys():
             cutout_mask = auxiliary_labels == 3
        else:
             cutout_mask = torch.zeros_like(img, dtype=bool)


        #### DEBUG

        # if cutout_mask.sum() > 0:
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     fig, ax = plt.subplots(1, 1)
        #     ax.imshow(img[img.shape[0]//2])
        #     ax.imshow(cutout_mask[cutout_mask.shape[0]//2], alpha=0.5)
        #     plt.show()
        

        if self.transforms is not None:
            del(sample['unmodified_center_slice'])
            orig_slice = img.detach().clone() # we need to get the full thickslice to replicate data augmentation on the original slice (i.e. rotation of the plane)
            label  = label[np.newaxis, :, :] # add slice thickness dimension of one
            #auxiliary_labels = auxiliary_labels[np.newaxis, :, :]
            
            weight = weight[np.newaxis, :, :]
            zoom_aug = torch.as_tensor([0., 0.])

            if self.aux_labels:
                img, label, weight, auxiliary_labels, orig_slice, cutout_mask, rep_tf = self.apply_transforms(img, label, weight, aux_labels=auxiliary_labels, orig_slice=orig_slice, cutout_mask=cutout_mask, subject_id=sample['subject_id'])
            else:
                img, label, weight, auxiliary_labels, orig_slice, cutout_mask, rep_tf = self.apply_transforms(img, label, weight, aux_labels=auxiliary_labels, orig_slice=orig_slice, cutout_mask=cutout_mask, subject_id=sample['subject_id'])


            if rep_tf and 'scales' in rep_tf[0]._get_reproducing_arguments().keys(): # get updated scalefactor, incase of scaling
                zoom_aug += torch.as_tensor(rep_tf[0]._get_reproducing_arguments()["scales"])[:-1]

            # Normalize image and clamp between 0 and 1 again after data augmentation
            img = torch.clamp(img / orig_slice.max(), min=0.0, max=1.0) # use original slice to normalize
            orig_slice = torch.clamp(orig_slice / orig_slice.max(), min=0.0, max=1.0)
            orig_slice = orig_slice[orig_slice.shape[0]//2]
            #cutout_mask = cutout_mask[cutout_mask.shape[0]//2]


            #assert(torch.max(img) == 1.0 and torch.min(img) == 0.0), "Image is not normalized between 0 and 1, but between {} and {}".format(torch.min(img), torch.max(img))
        else:
            orig_slice = sample['unmodified_center_slice']
            zoom_aug = torch.as_tensor([0., 0.])
            #cutout_mask = sample['cutout_mask'] # TODO: is not in sample yet

        scale_factor = self._get_scale_factor(torch.from_numpy(self.zooms[index]), scale_aug=zoom_aug)

        assert(cutout_mask.shape == img.shape), f"Cutout mask shape {cutout_mask.shape} does not match image shape {img.shape}"

        return {'image': img, 'label': label, 'weight': weight, 'aux_labels': auxiliary_labels, 
                'scale_factor': scale_factor, 'unmodified_center_slice': orig_slice, 'cutout_mask': cutout_mask, 'subject_id': subject_id}


class MultiScaleDataset(VINNDataset):
    """
    Class for loading aseg file with augmentations (transforms)
    """

    def __init__(self, dataset_path, cfg, scale_aug=False, transforms=None):
        self.default_size= cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES
        self.scale_aug = scale_aug

        self.localisation = cfg.MODEL.MODEL_NAME == "FastSurferLocalisation"

        # Load the h5 file and save it to the datase
        self.images = []
        self.labels = []
        self.weights = []
        self.subjects = []
        self.zooms = []
        self.com = []

        # Open file in reading mode
        logger.info(f"Opening file {dataset_path} for reading...")
        import os
        cwd =  os.path.realpath('.')
        assert (h5py.is_hdf5(dataset_path)), f"File \"{dataset_path}\" is not a valid hdf5 file."

        start = time.time()
        with h5py.File(dataset_path, "r") as hf:
            sizes = hf.keys() if cfg.DATA.SIZES is None else cfg.DATA.SIZES
            for size in hf.keys():
                try:
                    if not cfg.TRAIN.DEBUG:
                        self.load_data(hf, size)
                    else:
                        self.load_subset(hf, size, num_slices=10)

                    logger.info("Successfully loaded {} participant volumes from h5 file".format(len(self.subjects)))

                    assert (len(self.images) == len(self.labels) == len(self.weights) == len(self.zooms)), "Number of images, labels, weights and zooms are not equal."
                    #assert np.array([self.labels[0].shape == label.shape for label in self.labels]).all(), "Not all labels have the same shape"


                except KeyError as e:
                    print(f"KeyError: Unable to open object (object {size} does not exist)")
                    continue


            # TODO: CHECK
            #self.transforms = transforms
            self.transforms = None

            logger.info("Successfully loaded {} slices from {} with plane {} in {:.3f} seconds".format(len(self.images),
                                                                                                       dataset_path,
                                                                                                       cfg.DATA.PLANE,
                                                                                                       time.time() - start))

    # override this function to enable scale augmentation (could also be outside of this function)
    def _get_scale_factor(self, img_zoom, scale_aug=torch.tensor(0.0)):
        """
        Get scaling factor to match original resolution of input image to
        final resolution of FastSurfer base network. Input resolution is
        taken from voxel size in image header.

        TODO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.
        :param img_zoom:
        :return np.ndarray(float32): scale factor along x and y dimension
        """
        if torch.all(scale_aug > 0):
            img_zoom *= (1 / scale_aug)

        scale = self.base_res / img_zoom

        if self.scale_aug:
            scale += torch.randn(1) * 0.1 + 0  # needs to be changed to torch.tensor stuff
            scale = torch.clamp(scale, min=0.1)

        return scale

    def apply_transforms(self, img, label, weight, cutout_mask=None, orig_slice=None, subject_id=None):
        """
        apply transforms to the image and labels
        :param img: image
        :param label: label
        :param weight: weight
        :return: transformed image, label and weight, and composed torchio history
        """
        img = img[None, ...]  # add batch dimension for torchio
        label = label[None, ...]
        weight = weight[None, ...]
        if orig_slice is not None:
            orig_slice = orig_slice[None, ...]
        if cutout_mask is not None:
            cutout_mask = cutout_mask[None, ...]

        # pad label and weight to match sptial dimensions of image
        slice_thickness = img.shape[1]
        label = np.pad(label, ((0, 0), (slice_thickness // 2, slice_thickness // 2), (0, 0), (0, 0)), 'constant',
                       constant_values=0)
        weight = np.pad(weight, ((0, 0), (slice_thickness // 2, slice_thickness // 2), (0, 0), (0, 0)), 'constant',
                        constant_values=0)

        subject = tio.Subject({'image': tio.ScalarImage(tensor=img),
                                   'label': tio.LabelMap(tensor=label),
                                   'weight': tio.LabelMap(tensor=weight),
                                   'unmodified_center_slice': tio.ScalarImage(
                                       tensor=orig_slice) if orig_slice is not None else None,
                               'cutout_mask': tio.LabelMap(tensor=cutout_mask),
                               # TODO: this converts to UINT8, which is not what we want

                               'subject_id': subject_id
                                   })

        tx_sample = self.transforms(subject)  # this returns data as torch.tensors

        img = tx_sample['image'].data.float().squeeze(0)
        label = tx_sample['label'].data.byte()
        weight = tx_sample['weight'].data.float()
        if orig_slice is not None:
            orig_slice = tx_sample['unmodified_center_slice'].data.float().squeeze(0)
        if 'cutout_mask' in tx_sample.keys():
            cutout_mask = tx_sample['cutout_mask'].data.bool().squeeze(0)
        else:
            cutout_mask = torch.zeros(orig_slice.size(), dtype=bool)

        label = label[:, slice_thickness // 2, :, :].squeeze(0)  # retrieve middle slice
        weight = weight[:, slice_thickness // 2, :, :].squeeze(0)

        return img, label, weight, orig_slice, cutout_mask, tx_sample.get_composed_history()

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        img = sample['image']
        label = sample['label']
        weight = sample['weight']
        subject_id = sample['subject_id']

        if 'cutout_mask' in sample.keys():
             cutout_mask = sample['cutout_mask']
        else:
             cutout_mask = torch.zeros_like(img, dtype=bool)

        if self.localisation:
            com = torch.as_tensor(self._pad(self.com[index], self.pad_shape_labels))
            label = torch.cat((label, com), dim=0)

        if self.transforms is not None:
            del (sample['unmodified_center_slice'])
            orig_slice = img.detach().clone()  # we need to get the full thickslice to replicate data augmentation on the original slice (i.e. rotation of the plane)
            label = label[np.newaxis, :, :]  # add slice thickness dimension of one

            weight = weight[np.newaxis, :, :]
            zoom_aug = torch.as_tensor([0., 0.])

            img, label, weight, orig_slice, cutout_mask, rep_tf = self.apply_transforms(img, label, weight, orig_slice=orig_slice, subject_id=sample['subject_id'], cutout_mask=cutout_mask)
            # ERROR: Labels loose weights

            if rep_tf and 'scales' in rep_tf[
                0]._get_reproducing_arguments().keys():  # get updated scalefactor, incase of scaling
                zoom_aug += torch.as_tensor(rep_tf[0]._get_reproducing_arguments()["scales"])[:-1]

            # Normalize image and clamp between 0 and 1 again after data augmentation
            img = torch.clamp(img / orig_slice.max(), min=0.0, max=1.0)  # use original slice to normalize
            orig_slice = torch.clamp(orig_slice / orig_slice.max(), min=0.0, max=1.0)
            orig_slice = orig_slice[orig_slice.shape[0] // 2]

            # assert(torch.max(img) == 1.0 and torch.min(img) == 0.0), "Image is not normalized between 0 and 1, but between {} and {}".format(torch.min(img), torch.max(img))
        else:
            orig_slice = sample['unmodified_center_slice']
            zoom_aug = torch.as_tensor([0., 0.])

        scale_factor = self._get_scale_factor(torch.from_numpy(self.zooms[index]), scale_aug=zoom_aug)

        assert(cutout_mask.shape == img.shape), f"Cutout mask shape {cutout_mask.shape} does not match image shape {img.shape}"


        return {'image': img, 'label': label, 'weight': weight,
                'scale_factor': scale_factor, 'unmodified_center_slice': orig_slice,
                'cutout_mask': cutout_mask, 'subject_id': subject_id}


# # Operator to load hdf5-file for validation
class MultiScaleDatasetVal(VINNDataset):
    """
    Class for loading aseg file with augmentations (transforms)

    only used for legacy FastSurferVINN
    """
    #overrides
    def __init__(self, dataset_path, cfg, transforms=None):

        self.default_size = cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES

        # Load the h5 file and save it to the dataset
        self.images = []
        self.labels = []
        self.weights = []
        self.aux_labels = []
        self.subjects = []
        self.zooms = []

        # Open file in reading mode
        start = time.time()
        logger.info(f"Opening file {dataset_path} for reading...")
        assert(h5py.is_hdf5(dataset_path)), f"File \"{dataset_path}\" is not a valid hdf5 file."
        with h5py.File(dataset_path, "r") as hf:
            for size in hf.keys(): # iterate over image sizes
                if not cfg.TRAIN.DEBUG:
                    self.load_data(hf, size)
                else:
                    self.load_subset(hf, size)
                

        self.transforms = transforms
        logger.info("Successfully loaded {} slices from {} with plane {} in {:.3f} seconds".format(len(self.images),
                                                                                                 dataset_path,
                                                                                                 cfg.DATA.PLANE,
                                                                                                 time.time()-start))

    def __getitem__(self, index) -> dict:
        sample = super().__getitem__(index)

        if self.transforms is not None:
            sample = self.transforms(sample)

        if 'aux_data' in sample.keys():
            sample['cutout_mask'] = sample['aux_data'] == 3
            if sample['cutout_mask'].sum() > 0:
                #mid_slice = sample['image'][sample['image'].shape[0]//2]
                sample['image'][sample['cutout_mask']] = 0
                #sample['image'][sample['image'].shape[0]//2] = mid_slice
        else:
            sample['cutout_mask'] = torch.zeros(sample['label'].size(), dtype=bool)
        return sample


# class InpaintingDataset(VINNDataset):
#     """
#     Class for loading aseg file with augmentations (transforms)
#     """
#     #overrides
#     def __init__(self, original_dataset):
#         start = time.time()

#         self.default_size = original_dataset.max_size
#         self.base_res = original_dataset.base_res

        

#         # Load the h5 file and save it to the dataset
#         self.images = copy.deepcopy(original_dataset.images)
#         self.orig_images = original_dataset.images
#         self.labels = original_dataset.labels
#         self.aux_labels = original_dataset.aux_labels
#         self.weights = original_dataset.weights
#         self.subjects = original_dataset.subjects
#         self.zooms = original_dataset.zooms
#         self.cutout_masks = []#np.zeros((len(self.labels),self.labels[0].shape[0], self.labels[0].shape[1]), dtype=bool)
        
#         self.cutout_images()


#         if original_dataset.transforms is not None:
#             raise NotImplementedError("Transforms are not yet supported for inpainting dataset")
#             #self.transforms = original_dataset.transforms

#         #assert((self.images[0] != original_dataset.images[0]).any()), "Inpainting dataset is a copy of the original dataset"
#         assert(self.images[0] is not original_dataset.images[0]), 'images should point to different objects, but do not'
#         assert(self.orig_images is original_dataset.images), 'orig_images and images from donor dataset should point to the same object, but do not'
#         assert(self.labels[0] is original_dataset.labels[0]), 'labels should point to the same object, but do not'
#         assert(self.aux_labels[0] is original_dataset.aux_labels[0]), 'aux_labels should point to the same object, but do not'

#         logger.info("Successfully augmented {} slices with cutout in {:.3f} seconds".format(len(self.images), time.time()-start))

#     def cutout_images(self):
#         "Duplicate loaded images and generate deterministic cutout for validation"
#         # start_idx_of_cutout_imgs = len(self.images)

#         # self.images.extend(self.images.copy())
#         # self.labels.extend(self.labels.copy())
#         # self.weights.extend(self.weights.copy())
#         # self.zooms.extend(self.zooms.copy())


#         cutout_size = 5 #th of the image size

#         # iterate over all images and cutout a square move it on a grid left to right and top to bottom
#         # cutout size is the grid size
#         for i in range(len(self.images)):
#             grid_x = i%cutout_size
#             x1 = int(grid_x/cutout_size*self.images[i].shape[0])
#             x2 = int((grid_x+1)/cutout_size*self.images[i].shape[0])

            
#             grid_y = ((i)//cutout_size)%cutout_size
#             y1 = int(grid_y/cutout_size*self.images[i].shape[1])
#             y2 = int((grid_y+1)/cutout_size*self.images[i].shape[1])

#             #print(x1,x2,y1,y2)

#             self.images[i][x1:x2,y1:y2] = 0
#             self.weights[i][x1:x2,y1:y2] *= 0.5
#             self.cutout_masks.append(np.zeros(self.labels[i].shape, dtype=bool))
#             self.cutout_masks[i][x1:x2,y1:y2] = True

#             #self.images[i,i/cutout_size*self.images[i].shape[1],i*cutout_size*self.images[i].shape[1]] = 0.0
#             #self.weights[i,i*cutout_size*self.images[i].shape[1],i*cutout_size*self.images[i].shape[1]] *= 0.5

#     def __getitem__(self, index) -> dict:
#         padded_img, padded_label, padded_weight, padded_orig, padded_cutout_mask, padded_auxiliary_labels = self.unify_imgs([self.images[index], self.labels[index], self.weights[index], self.orig_images[index], self.cutout_masks[index], self.aux_labels[index]])
        
#         padded_img = padded_img.transpose((2, 0, 1)) # move slice thickness to first spatial dimension
#         padded_img = torch.clamp(torch.from_numpy(padded_img).float() / padded_orig.max(), 0, 1) # use original max to normalize

#         padded_orig = padded_orig.transpose((2, 0, 1)) # move slice thickness to first spatial dimension
#         padded_orig = torch.clamp(torch.from_numpy(padded_orig).float() / padded_orig.max(), 0, 1)
#         padded_orig = padded_orig[padded_orig.shape[0] // 2] # take center slice

        
#         padded_weight = torch.from_numpy(padded_weight).float()
#         padded_label = torch.from_numpy(padded_label)
#         padded_auxiliary_labels = torch.from_numpy(padded_auxiliary_labels)
#         padded_cutout_mask = torch.from_numpy(padded_cutout_mask).bool()

#         scale_factor = self._get_scale_factor(torch.from_numpy(self.zooms[index]))

#         return {'image': padded_img, 'label': padded_label, 'weight': padded_weight, 
#                 'scale_factor': scale_factor, 'unmodified_center_slice': padded_orig, 
#                 'cutout_mask': padded_cutout_mask, 'subject_id': self.subjects[index], 'aux_labels': padded_auxiliary_labels}

    