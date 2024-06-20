
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
from __future__ import annotations
import itertools
from numbers import Number
import random
#import monai

import numpy as np
import torch
import nibabel as nib
import h5py

from typing import Any, List
from typing import Sequence
from typing import Tuple
from typing import TypeVar

import numpy as np
import torch
import torchvision

import torchio as tio
from torchio.transforms.augmentation import RandomTransform
from torchio.transforms import IntensityTransform, RandomFlip
from torchio.data.subject import Subject
from torchio.typing import TypeTripletInt
from torchio.typing import TypeTuple
from torchio.utils import to_tuple

from scipy.spatial import ConvexHull, Delaunay

TypeLocations = Sequence[Tuple[TypeTripletInt, TypeTripletInt]]
TensorArray = TypeVar('TensorArray', np.ndarray, torch.Tensor)

##
# Transformations for evaluation
##
class ToTensorTest(object):
    """
    Convert np.ndarrays in sample to Tensors.   #TODO: Thats not what is happening here
    """

    def __init__(self, include=['image', 'label', 'weight', 'cutout_mask']) -> None:
        self.include = include

    def __call__(self, img):

        if isinstance(img, dict):
            for key in self.include:
                img[key] = torch.from_numpy(self._clip_and_transpose(img[key]))
        elif isinstance(img, np.ndarray) or isinstance(img, torch.Tensor):
            img = torch.from_numpy(self.x_clip_and_transpose(img))

        return img

    @staticmethod
    def _clip_and_transpose(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)

        # Normalize and clamp between 0 and 1
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))
        return img



class ZeroPad2DTest(object):
    def __init__(self, output_size, pos='top_left'):
        """
         Pad the input with zeros to get output size
        :param output_size:
        :param pos: position to put the input
        """
        if isinstance(output_size, Number):
            output_size = (int(output_size), ) * 2
        self.output_size = output_size
        self.pos = pos

    def _pad(self, image):
        if len(image.shape) == 2:
            h, w = image.shape
            padded_img = np.zeros(self.output_size, dtype=image.dtype)
        else:
            h, w, c = image.shape
            padded_img = np.zeros(self.output_size + (c,), dtype=image.dtype)

        if self.pos == 'top_left':
            padded_img[0: h, 0: w] = image

        return padded_img

    def __call__(self, img):

        if isinstance(img, dict):
            for key in ['image', 'label', 'weight']:
                img[key] = self._pad(img[key])
        elif isinstance(img, np.ndarray) or isinstance(img, torch.Tensor):
            img = self._pad(img)

        return img


##
# Transformations for training
##
class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __init__(self, keys=None):
        self.keys = keys

    def __call__(self, sample):
        return_dict = dict(**sample)
        #img, label, weight = sample['image'], sample['label'], sample['weight']

        if self.keys == None or 'image' in self.keys:
            img = sample['image']

            #img = img.astype(np.float32)

            # Normalize image and clamp between 0 and 1
            img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            # if isinstance(img, np.ndarray):
            #     img = img.transpose((2, 0, 1))
            # elif isinstance(img, torch.Tensor):
            #     return_dict['image'] = img.permute((2, 0, 1))

        if self.keys == None:
            keys = list(sample.keys())
        else:
            keys = self.keys

        for key in keys:
            if isinstance(sample[key], np.ndarray):
                return_dict[key] = torch.from_numpy(sample[key])

        return return_dict


class ZeroPad2D(object):
    def __init__(self, output_size, pos='top_left'):
        """
         Pad the input with zeros to get output size
        :param output_size:
        :param pos: position to put the input
        """
        if isinstance(output_size, Number):
            output_size = (int(output_size), ) * 2
        self.output_size = output_size
        self.pos = pos

    def _pad(self, image):
        if len(image.shape) == 2:
            h, w = image.shape
            padded_img = np.zeros(self.output_size, dtype=image.dtype)

            if self.pos == 'top_left':
                padded_img[0: h, 0: w] = image
        else:
            assert(len(image.shape) == 3)
            d, h, w = image.shape
            padded_img = np.zeros((d,) + self.output_size, dtype=image.dtype)

            if self.pos == 'top_left':
                padded_img[:, 0: h, 0: w] = image

        return padded_img

    def __call__(self, sample):
        return_dict = dict(**sample)
        img, label, weight = sample['image'], sample['label'], sample['weight']

        return_dict['image'] = self._pad(img)
        return_dict['label'] = self._pad(label)
        return_dict['weight'] = self._pad(weight)

        return return_dict  #{'image': img, 'label': label, 'weight': weight}





# TODO: maybe we should reduce weights for removed areas
class RandomAugmentation(object):

    def __init__(self, p=1.0):
        self.probability = p

    def decision(self, prob=None):
        if prob is None:
            return random.random() < self.probability
        else:
            return random.random() < prob

    def downweight(self, weight, mask):
        weight = weight.copy()
        weight[mask] = weight[mask] * 0.5
        return weight
    

class RandomizeScaleFactor(RandomAugmentation):
    def __init__(self, mean=0, std=0.1, p=1.0):
        super().__init__(p)
        self.std = std
        self.mean = mean

    def __call__(self, sample):

        if self.decision():
            return_dict = dict(**sample)
            #img, label, weight, sf = sample['image'], sample['label'], sample['weight'], sample['scale_factor']
            sf = sample['scale_factor']
            # change 1 to sf.size() for isotropic scale factors (now same noise change added to both dims)
            sf = sf + torch.randn(1) * self.std + self.mean
            
            return_dict['scale_factor'] = sf

            # TODO: check that this really changes randomly
            return return_dict
        else:
            return sample




class RandomCutout(RandomTransform, IntensityTransform):
    r"""Randomly set patches to zero within an image.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to swap patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.
        num_cuts: Number patches that will be cut
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(
            self,
            patch_size: TypeTuple = None,
            num_cuts: int = 1,
            downweighting_factor: float = 0.5,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = np.array(to_tuple(patch_size))
        self.num_cuts = self._parse_num_iterations(num_cuts)
        self.downweighting_factor = downweighting_factor

    @staticmethod
    def _parse_num_iterations(num_iterations):
        if not isinstance(num_iterations, int):
            raise TypeError(
                'num_iterations must be an int,'
                f'not {num_iterations}',
            )
        if num_iterations < 0:
            raise ValueError(
                'num_iterations must be positive,'
                f'not {num_iterations}',
            )
        return num_iterations

    def get_params(self,
            tensor: torch.Tensor,
            patch_size: np.ndarray,
            num_cuts: int,
            subject: Subject
    ) -> List[Tuple[TypeTripletInt, TypeTripletInt]]:
        si, sj, sk = tensor.shape[-3:]
        spatial_shape = si, sj, sk  # for mypy
        if patch_size[0] is None:
            patch_size = np.random.randint([0,0,0], spatial_shape, (3,))
        locations = []
        for _ in range(num_cuts):
            first_ini, first_fin = self.get_random_indices_from_shape(
                spatial_shape,
                patch_size.tolist(),
                subject
            )
                
            locations.append((tuple(first_ini), tuple(first_fin)))
        return locations  # type: ignore[return-value]

    def apply_transform(self, subject: Subject) -> Subject:
        THICKSLICE_DIMENSION = 0
        orig_image = self.get_images_dict(subject)['image']
        if self.downweighting_factor > 0:
            wgt = subject['weight'].data.clone()
        
        midslice_no = orig_image.data.shape[THICKSLICE_DIMENSION]//2
        orig_slice= orig_image.data[midslice_no]

        locations = self.get_params(
                orig_image,
                self.patch_size,
                self.num_cuts,
                subject
            )
        for name, image in self.get_images_dict(subject).items():
            
            if name != 'image':
                assert(orig_slice.shape == image.shape), f'expected slice dimension, but got {image.shape}, for {name}'
            
            
            img = image.data.clone()
            for location in locations:
                if name != 'image': # apply to slice if midslice is included in image (applies cutout to weights and segmentation) - experimental
                    if location[0][THICKSLICE_DIMENSION] <= midslice_no and location[1][THICKSLICE_DIMENSION] >= midslice_no:
                        continue
                    else:
                        location[0][THICKSLICE_DIMENSION] = 0 
                        location[1][THICKSLICE_DIMENSION] = 0
                
                img = self._set_constant(img, location[0], location[1], constant=0)  # type: ignore[arg-type]  # noqa: E501
                if self.downweighting_factor > 0:
                    wgt = self._multiply(wgt, location[0], location[1], constant=self.downweighting_factor)
            image.set_data(img)
        if self.downweighting_factor > 0:
            subject['weight'].set_data(wgt)

        if not 'cutout_mask' in subject.keys():
            subject.add_image(tio.LabelMap(tensor=torch.zeros(self.get_images_dict(subject)['label'].data.size(), dtype=bool)), 'cutout_mask')
        subject['cutout_mask'] = self._set_constant(subject['cutout_mask'].data, location[0], location[1], constant=1)
        
        return subject

    @staticmethod
    def _set_constant(
            image: TensorArray,
            index_ini: np.ndarray,
            index_fin: np.ndarray,
            constant: float = 0
    ) -> TensorArray:
        i_ini, j_ini, k_ini = index_ini
        i_fin, j_fin, k_fin = index_fin
        image[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = constant
        return image
    
    @staticmethod
    def _multiply(
            image: TensorArray,
            index_ini: np.ndarray,
            index_fin: np.ndarray,
            constant: float = 0
    ) -> TensorArray:
        i_ini, j_ini, k_ini = index_ini
        i_fin, j_fin, k_fin = index_fin
        #image[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] //= int(1/constant) # throws deprecated warning
        image[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin].div_(int(1/constant), rounding_mode='floor') # inplace integer division
        return image

    def get_random_indices_from_shape(self,
            spatial_shape: Sequence[int],
            patch_size: Sequence[int],
            subject: Subject=None) -> Tuple[np.ndarray, np.ndarray]: # subject used in child class
        assert len(spatial_shape) == 3
        assert len(patch_size) in (1, 3)
        shape_array = np.array(spatial_shape)
        patch_size_array = np.array(patch_size)
        max_index_ini_unchecked = shape_array - patch_size_array
        if (max_index_ini_unchecked < 0).any():
            message = (
                f'Patch size {patch_size} cannot be'
                f' larger than image spatial shape {spatial_shape}'
            )
            raise ValueError(message)
        max_index_ini = max_index_ini_unchecked.astype(np.uint16)
        coordinates = []
        for max_coordinate in max_index_ini.tolist():
            if max_coordinate == 0:
                coordinate = 0
            else:
                coordinate = int(torch.randint(max_coordinate, size=(1,)).item())
            coordinates.append(coordinate)
        index_ini = np.array(coordinates, np.uint16)
        index_fin = index_ini + patch_size_array
        return index_ini, index_fin
    
class SmartRandomCutout(RandomCutout):
    """
    Ensures that at least three of the four corners of the cutout block are within the brain
    and adjusts probability for partial cutout in the slice thickness dimension
    """

    def __init__(self, brain_div_patch_size: float = 2, num_cuts: int = 1, full_random_probability=0.1, **kwargs):
        super().__init__(None, num_cuts, **kwargs)
        self.brain_div_patch_size = brain_div_patch_size
        self.full_random_probability = full_random_probability

    def get_brain_mask(self, subject: Subject) -> np.ndarray:
        if 'aux_label' in subject.keys():
            brain_mask = subject['aux_label'].data > 0 # brainmask from hdf5
        else:
            print(f'WARNING: no brain mask found, using convex hull of aseg instead')
            try: # this can fail if the input mask is of unexpected shape (e.g.: not enough points(1) to construct initial simplex (need 4))
                _, brain_mask = getConvexHull(subject) # TODO: make static in pre-processing
            except:
                return None #super().get_random_indices_from_shape(spatial_shape, patch_size)
        return brain_mask
    
    @staticmethod
    def _get_rectangle_around_mask(mask):

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        #return np.round(np.mean(np.array(mask.nonzero())[[1,2]], axis=1)).astype(int) # get center of tumor mask in slice

        #print('mask shape:', mask.shape)

        # Find indices where we have mass
        _, _, x, y = torch.where(mask)
        # mass_x and mass_y are the list of x indices and y indices of mass pixels

        min_xy = torch.min(x), torch.min(y)
        max_xy = torch.max(x), torch.max(y)

        return min_xy, max_xy

    def get_random_indices_from_shape(self,
            spatial_shape: Sequence[int],
            patch_size: Sequence[int],
            subject: Subject) -> Tuple[np.ndarray, np.ndarray]:
        # get slice thickness dimension
        MIDSLICE = spatial_shape[0]//2
        SLICE_THICKNESS_DIM = np.argmin(spatial_shape)
        SLICE_DIMENSIONS = list(range(3))
        SLICE_DIMENSIONS.remove(SLICE_THICKNESS_DIM)

        brain_mask = self.get_brain_mask(subject)

        if brain_mask is None:
            return super().get_random_indices_from_shape(spatial_shape, patch_size)

        min_coord, max_coord = self._get_rectangle_around_mask(brain_mask)

        brain_spatial_shape = np.zeros(3) 
        brain_spatial_shape[SLICE_DIMENSIONS] = (max_coord[0]-min_coord[0], max_coord[1]-min_coord[1])
        brain_spatial_shape[SLICE_THICKNESS_DIM] = spatial_shape[SLICE_THICKNESS_DIM]

        max_coutout_size = brain_spatial_shape//self.brain_div_patch_size

        if (max_coutout_size[SLICE_THICKNESS_DIM] <= 0).any() or (max_coutout_size[SLICE_DIMENSIONS] <= 0).any(): # if the visible brain is too small, just do a random cutout
            return super().get_random_indices_from_shape(spatial_shape, patch_size)
        
        index_ini, index_fin = super().get_random_indices_from_shape(brain_spatial_shape, np.random.randint(0, max_coutout_size))
        index_ini = index_ini.astype(int)
        index_fin = index_fin.astype(int)

        index_ini[SLICE_DIMENSIONS] += np.array(min_coord)
        index_fin[SLICE_DIMENSIONS] += np.array(min_coord)

        
        # chance of completely random cutout, otherwise full cutout in slice thickness dimension and ensure that at least three of the four corners are within the brain
        if np.random.rand() > self.full_random_probability:
            index_ini[SLICE_THICKNESS_DIM] = 0
            index_fin[SLICE_THICKNESS_DIM] = spatial_shape[SLICE_THICKNESS_DIM]-1 #TODO: check for off by one errors            

            for _ in range(3):
                pts_in_brain = 0
                corner_points = np.array(list(itertools.product(*zip(index_ini[SLICE_DIMENSIONS],index_fin[SLICE_DIMENSIONS]))))


                for point in corner_points:
                    if brain_mask[0,MIDSLICE,point[0], point[1]]:
                        pts_in_brain += 1
                


                if pts_in_brain >= 3:
                    return index_ini, index_fin
                else: # if the cutout is not within the brain, try again, but constrain the cutout to the brain
                    index_ini, index_fin = super().get_random_indices_from_shape(brain_spatial_shape, np.random.randint(0, max_coutout_size))
                    index_ini[SLICE_THICKNESS_DIM] = 0
                    index_fin[SLICE_THICKNESS_DIM] = spatial_shape[SLICE_THICKNESS_DIM]-1 #TODO: check for off by one errors 

                    index_ini = index_ini.astype(int)
                    index_fin = index_fin.astype(int)
                    index_ini[SLICE_DIMENSIONS] += min_coord
                    index_fin[SLICE_DIMENSIONS] += min_coord
                    
            #print('WARNING: could not find a cutout that was within the brain')
            return index_ini, index_fin
        else:
            return index_ini, index_fin
        
        

# class RandGridDistortiond(monai.transforms.RandGridDistortiond):

#     ONE_D_KEYS = ['label', 'cutout_mask', 'weight', 'unmodified_center_slice']


#     @property
#     def p(self):
#         return self.prob
    
#     @p.setter
#     def p(self, value):
#         self.prob = value


#     def __call__(self, data):
#         """
#         Args:
#             spatial_size: spatial size of the grid.
#         """
#         monai_dict = {}
#         for key in self.keys:
#             if key in self.ONE_D_KEYS:
#                 #print(key, data[key].data.dtype)
#                 #print(data[key].data.shape)
#                 monai_dict[key] = data[key].data[:,3, None]

#                 # check if data is int
#                 #if data[key].data.dtype == torch.uint8:
#                 #    monai_dict[key] = data[key].data.to(torch.float16)
#                 #    print('casted', key, 'to float32')
#                 #else:
                
#             else:
#                 monai_dict[key] = data[key].data

#         out = super().__call__(monai_dict)

#         for key in self.keys:
#             if key in self.ONE_D_KEYS:
#                 data[key].set_data(torch.nn.functional.pad(out[key], pad=(0,0,0,0,3,3), value=0))
#                 #print(data[key].data.shape)
#             else:
#                 data[key].set_data(out[key])

#         return data




class RandGridDistortiond():

    def __init__(self, **kwargs):
        pass


    @property
    def p(self):
        return self.prob
    
    @p.setter
    def p(self, value):
        self.prob = value


    def __call__(self, data):
        raise NotImplementedError('use monai version')

        return data


class CutoutBRATSTumor(SmartRandomCutout):

    def __init__(self, tumor_mask_hdf5: str, num_cuts: int = 1, cutout_value: int=0, random=False, **kwargs):
        super().__init__(None, num_cuts, **kwargs)
        self.cutout_value = cutout_value
        self.random = random

        # load tumor masks
        self.tumor_masks = []
        
        #print('loading tumor masks for data augmentation')
        # start_t = time.time()
        #with h5py.File(tumor_mask_hdf5, 'r') as f:
        f = h5py.File(tumor_mask_hdf5, 'r')
        #for size in f.keys():
            #self.tumor_masks.extend(list(f[f'{size}']['mask_dataset']))
        sizes = list(f.keys())
        assert(len(sizes) == 1), 'expected only one size in tumor mask hdf5'
        size = sizes[0]
        self.tumor_masks = f[f'{size}']['mask_dataset']

        #print(f'loading tumor masks took {time.time()-start_t} seconds')
            
    def get_desired_tumor_center(self, subject: Subject, midslice=3) -> np.ndarray:
        """
        Get a random point within the brain mask as the center of the tumor

        :param subject: subject to get tumor center from
        :return: tumor center
        """

        brain_mask = self.get_brain_mask(subject)

        assert(brain_mask is not None), 'brain mask not found - this can happen if convex hull fails; use precomputed brain mask instead'

        # get random point in brain
        brain_points = brain_mask.nonzero()[:,[2,3]]
        tumor_center = brain_points[np.random.randint(brain_points.shape[0])]

        #assert(brain_mask[0, midslice, tumor_center[0], tumor_center[1]]), 'random point not in brain'

        return tumor_center
    
    @staticmethod
    def _set_random(
            image: TensorArray,
            mask: np.ndarray,
            #constant: float = 0
    ) -> TensorArray:
        #i_ini, j_ini, k_ini = index_ini
        #i_fin, j_fin, k_fin = index_fin
        #image[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = np.random.rand(image[:, i_ini:i_fin, j_ini:j_fin, k_ini:k_fin].shape)

        image[mask] = np.random.rand(image.shape[0])[mask]
        return image
    
    @staticmethod
    def _set_constant(
            image: TensorArray,
            mask: np.ndarray,
            constant: float = 0
    ) -> TensorArray:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        image[mask] = constant
        return image
    
    @staticmethod
    def _set_random(
            image: TensorArray,
            mask: np.ndarray,
    ) -> TensorArray:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        image[mask] = torch.rand(*image.shape)[mask]
        return image
    
    @staticmethod
    def _multiply(
            image: TensorArray,
            mask: np.ndarray,
            constant: float = 0
    ) -> TensorArray:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        image[mask] *= constant
        return image
    

    def get_random_tumor_mask(self, subject: Subject=None):
        # TODO: maybe reorder dimensions
        t_mask = self.tumor_masks[np.random.randint(len(self.tumor_masks))]
        while t_mask.sum() == 0:
            print('WARNING: tumor mask is empty, trying again')
            t_mask = self.tumor_masks[np.random.randint(len(self.tumor_masks))]
        return t_mask
    
    @staticmethod
    def _apply_random_affine(mask: np.ndarray, mask_center: Tuple) -> np.ndarray:
        """
        Apply random affine transformation to tumor mask
        :param mask: tumor mask
        :param subject: subject to get affine from
        :return: transformed tumor mask
        """
        rotation = (-180, 180)
        translation = (0,0)
        scaling = (0.5, 2)
        shearing = (-0.1, 0.1)

        # get random parameters
        rotation = np.random.randint(rotation[0], rotation[1])
        scaling = np.random.rand() * (scaling[1] - scaling[0]) + scaling[0]
        shearing = np.random.rand() * (shearing[1] - shearing[0]) + shearing[0]
        mask_center = np.array(mask_center) + np.random.randint(-5, 5, size=2) # add random offset to tumor center

        # get random affine transformation
        return torchvision.transforms.functional.affine(torch.from_numpy(mask), angle=rotation, translate=translation, scale=scaling, shear=shearing, fill=0, center=list(mask_center))
    
    @staticmethod
    def _get_mask_center(mask):

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        #return np.round(np.mean(np.array(mask.nonzero())[[1,2]], axis=1)).astype(int) # get center of tumor mask in slice

        # Find indices where we have mass
        _, x, y = torch.where(mask)
        # mass_x and mass_y are the list of x indices and y indices of mass pixels

        cent_x = torch.mean(x, dtype=float).round().int()
        cent_y = torch.mean(y, dtype=float).round().int()

        return cent_x, cent_y
        
        #return torch.median(mask.nonzero(), axis=0)[[0,1]].round().int() # get center of tumor mask in slice

    
    def apply_transform(self, subject: Subject) -> Subject:
        THICKSLICE_DIMENSION = 0
        
        orig_image = self.get_images_dict(subject)['image']

        MIDSLICE = orig_image.data.shape[THICKSLICE_DIMENSION]//2
        # if self.downweighting_factor > 0:
        #     wgt = subject['weight'].data.clone()
        
        midslice_no = orig_image.data.shape[THICKSLICE_DIMENSION]//2
        orig_slice= orig_image.data[midslice_no]

        tumor_mask = self.get_random_tumor_mask(subject=subject)
        # pad tumor mask to match image size
        if tumor_mask.shape[0] < orig_image.data.shape[2] or tumor_mask.shape[1] < orig_image.data.shape[3]:
            tumor_mask = np.pad(tumor_mask, (
                (0, orig_image.data.shape[2] - tumor_mask.shape[0]), 
                (0, orig_image.data.shape[3] - tumor_mask.shape[1]), 
                (0,0)), #, (0,0), 
                'constant', constant_values=0)
        elif tumor_mask.shape[0] > orig_image.data.shape[2] or tumor_mask.shape[1] > orig_image.data.shape[3]:
            tumor_mask = tumor_mask[:orig_image.data.shape[2], :orig_image.data.shape[3], :]
        elif tumor_mask.shape[0] == orig_image.data.shape[2] and tumor_mask.shape[1] == orig_image.data.shape[3]:
            pass
        else:
            raise ValueError('unexpected tumor mask shape')
        
        # bring thickness dimension to front
        tumor_mask = np.moveaxis(tumor_mask, -1, 0)

        tumor_mask = tumor_mask.astype(bool)

        tumor_mask_center_pre_affine = self._get_mask_center(tumor_mask)

        
        #plt.savefig('../../tmp/tumor_mask.png')

        # apply random affine transformation to tumor mask
        tumor_mask = self._apply_random_affine(tumor_mask, tumor_mask_center_pre_affine)

        tumor_mask_center = self._get_mask_center(tumor_mask)
        desired_center = self.get_desired_tumor_center(subject, midslice=MIDSLICE)
        

        # # plot tumor mask with center
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1,7, figsize=(20,5))
        # for i,ax in enumerate(axs):
        #     ax.imshow(tumor_mask[i].T)
        #     ax.scatter(tumor_mask_center[0], tumor_mask_center[1], c='r', label='tumor mask center')
        #     ax.scatter(desired_center[0], desired_center[1], c='b', label='desired center')
        #     ax.legend()
        # # plt.imshow(tumor_mask[MIDSLICE])
        # # plt.scatter(tumor_mask_center[0], tumor_mask_center[1], c='r', label='tumor mask center')
        # # plt.legend()

        

        to_translate =  desired_center - torch.tensor(tumor_mask_center)
        # translate tumor mask to desired center
        tumor_mask = torch.roll(tumor_mask, (to_translate[0], to_translate[1]), dims=(1,2)) 

        # # plot transformed tumor mask with desired center
        # plt.figure()
        # plt.imshow(tumor_mask[MIDSLICE].T)
    
        # plt.scatter(tumor_mask_center[0], tumor_mask_center[1], c='r', label='tumor mask center')
        # plt.scatter(desired_center[0], desired_center[1], c='b', label='desired center')
        # plt.legend()
        # #plt.savefig('../../tmp/tumor_mask_transformed.png')

        # plt.figure()
        # plt.imshow(self.get_brain_mask(subject)[0,MIDSLICE].T)
        # plt.scatter(desired_center[0], desired_center[1], c='b', label='desired center')
        # plt.legend()
        

        tumor_mask = tumor_mask[np.newaxis, ...] # add batch dimension

        # print('tumor mask shape:', tumor_mask.shape)
        # print('tumor mask center', tumor_mask_center)
        # print('desired center', desired_center)
        # print('to translate', to_translate)
        # print('sum tumor mask', tumor_mask.sum())

        

        #for name, image in self.get_images_dict(subject).items():

        #print('applying cutout for', name)
        
        # if name != 'image':
        #     assert(orig_slice.shape == image.shape), f'expected slice dimension, but got {image.shape}, for {name}'
        
        #print(name)
        
        img = subject['image'].data.clone()

        # for image consider thickslices
        #if name == 'image': # apply to slice if midslice is included in image (applies cutout to weights and segmentation) - experimental
        if self.random:
            img = self._set_random(img, mask=tumor_mask)
        else:
            img = self._set_constant(img, mask=tumor_mask, constant=0)  # type: ignore[arg-type]  # noqa: E501
            
        # elif self.downweighting_factor > 0 and name == 'weight':
        #     print('reducing weight for cutout in image', name)
        #     img = self._multiply(img, mask=tumor_mask[:, THICKSLICE_DIMENSION], constant=self.downweighting_factor)
        # else:
        #     raise ValueError(f'unknown image type {name}')
            #img = self._set_constant(img, mask=tumor_mask[:, THICKSLICE_DIMENSION], constant=0)  # type: ignore[arg-type]  # noqa: E501
        
        subject['image'].set_data(img)

        if self.downweighting_factor > 0:
            #print('reducing weight for cutout in image', name)
            wgt = self._multiply(subject['weight'].data.clone(), mask=tumor_mask, constant=self.downweighting_factor)
            subject['weight'].set_data(wgt)

        if not 'cutout_mask' in subject.keys():
            subject.add_image(tio.LabelMap(tensor=torch.zeros(self.get_images_dict(subject)['label'].data.size(), dtype=bool)), 'cutout_mask')
        subject['cutout_mask'] = self._set_constant(subject['cutout_mask'].data, mask=tumor_mask, constant=1)
        
        return subject
    


class CutoutBRATSTumorDeterministic(CutoutBRATSTumor):
         
    def get_random_tumor_mask(self, subject: Subject):
        idx = int.from_bytes(subject["subject_id"], byteorder="big") % len(self.tumor_masks)
        t_mask = self.tumor_masks[idx]
        return t_mask

    def get_tumor_mask(self, index: int):
        return self.tumor_masks[index]
    
    def get_desired_tumor_center(self, subject: Subject, midslice=3) -> np.ndarray:
        """
        Get a point within the brain mask as the center of the tumor

        :param subject: subject to get tumor center from
        :return: tumor center
        """

        brain_mask = self.get_brain_mask(subject)

        assert(brain_mask is not None), 'brain mask not found - this can happen if convex hull fails; use precomputed brain mask instead'

        # get random point in brain
        brain_points = brain_mask.nonzero()[:,[2,3]]
        tumor_center = brain_points[int.from_bytes(subject["subject_id"], byteorder="big") % brain_points.shape[0]]

        #assert(brain_mask[0, midslice, tumor_center[0], tumor_center[1]]), 'random point not in brain'

        return tumor_center

class CutoutTumorMask(IntensityTransform):

    def apply_transform(self, subject: Subject) -> Subject:

        if 'cutout_mask' in subject.keys() and subject['cutout_mask'].data.sum() > 0:
            #subject['image'].data[subject['cutout_mask'].data.bool()] = torch.rand(subject['image'].data.shape)[subject['cutout_mask'].data.bool()]
            subject['image'].data[subject['cutout_mask'].data.bool()] = torch.ones(subject['image'].data.shape)[subject['cutout_mask'].data.bool()]
            

        return subject
    
class CutoutTumorMaskInference(object):

    def __call__(self, subject: dict) -> dict:
        if 'cutout_mask' in subject.keys():
            #subject['image'][subject['cutout_mask'].bool()] = torch.rand(subject['image'].shape)[subject['cutout_mask'].bool()]
            subject['image'][subject['cutout_mask'].bool()] = torch.ones(subject['image'].shape)[subject['cutout_mask'].bool()]

        return subject

        


class CutoutHemisphere(IntensityTransform, RandomTransform):

    """
    Sets half of the image to zero - on the medial plane, with a small random offset
    
    TODO: determine medial plane - e.g. run make_upright on all training images, then and add to hdf5
    """

    def __init__(self, orientation: str, left: bool, weight_multiplier: int=0.5, **kwargs):
        super().__init__(**kwargs)
        self.left = left
        self.weight_reduction = weight_multiplier

        self.LR_axis, right_oriented = self._get_LR_axis(orientation)
        if not right_oriented:
            self.left = not self.left

        #if self.LR_axis == -1:
            
        #print(f'CutoutHemisphere: left = {self.left}, LR_axis = {self.LR_axis}, orientation = {orientation}')
        if self.LR_axis == 0: # slice thickness dimension, skip
            print('WARNING: disabling CutoutHemisphere transform, as it is not applicable to the slice thickness dimension')
            self.enabled = False
        else:
            self.enabled = True

        self.LR_axis += 1 # add batch dimension

    @staticmethod
    def _convert_orientation(orientation: str) -> str:
        if orientation == 'sagittal':
            orientation = 'RSA'
        elif orientation == 'axial':
            orientation = 'SAR'
        elif orientation == 'coronal':
            orientation = 'ARS'
        return orientation
    
    @staticmethod
    def _get_LR_axis(orientation: str) -> int:
        right_oriented = True
        orientation = CutoutHemisphere._convert_orientation(orientation) # handle FastSurfer slicing strings

        if orientation == 'RSA' or orientation == 'SRA' or orientation == 'ARS':
            LR_axis = orientation.index('R')
        else:
            print(f'WARNING - orientation not recognized, standard FastSurfer orientations are  RSA [saggital], SRA [axial], ARS [coronal] - instead got {orientation}')
            if 'R' in orientation and 'L' not in orientation:
                LR_axis = orientation.index('R')
            elif 'L' in orientation and 'R' not in orientation:
                right_oriented = False
                LR_axis = orientation.index('L')
            else:
                raise ValueError('orientation {orientation} not recognized')
        
        return LR_axis, right_oriented

    def apply_transform(self, subject: Subject) -> Subject:
        if self.enabled:
            image = subject['image']
            weight = subject['weight']
            sagittal_size = image.shape[self.LR_axis] # LR axis size

            if self.LR_axis != 0:
                brain_center = getBrainCenter(subject)
                mid_slice = brain_center[self.LR_axis - 1] # not actually the mid slice, but a crude approximation (not accounting for rotation)
            
            slc = [slice(None)] * len(image.data.shape)
            if self.left:
                slc[self.LR_axis] = slice(mid_slice, sagittal_size)
            else:
                slc[self.LR_axis] = slice(0, mid_slice)
            image.data[slc] = 0
            weight.data[slc] *= self.weight_reduction



            # slc = [slice(None)] * len(image.data.shape)
            # slc[1] = slice(image.data.shape[1]//2, image.data.shape[1])
            # image.data[slc] = torch.max(image.data) 

            # slc = [slice(None)] * len(image.data.shape)
            # slc[2] = slice(image.data.shape[2]//2, image.data.shape[2])
            # image.data[slc] = torch.max(image.data//2) # right

            # slc = [slice(None)] * len(image.data.shape)
            # slc[3] = slice(image.data.shape[3]//2, image.data.shape[3])
            # image.data[slc] = 0 # inferior
        return subject


class FlipLeftRight(RandomFlip):
    """
    Flip depending on orientation
    """
    def __init__(self, orientation, **kwargs):
        axis, _ = CutoutHemisphere._get_LR_axis(orientation)

        # NOTE: flip probability is per axis, since we only flip one axis this is the same as setting p (probability) to 1.0
        super().__init__(axis, flip_probability=1.0, **kwargs) 





class CutoutRandomHemisphere(CutoutHemisphere):

    """
    Sets half of the image to zero - on the medial plane, with a small random offset
    
    TODO: determine medial plane
    """

    def __init__(self, orientation, weight_multiplier=0.5, **kwargs):
        super().__init__(left=True, orientation=orientation, weight_multiplier=weight_multiplier, **kwargs)

    def __call__(self, subject: Subject) -> Subject:
        self.left = bool(torch.randint(2, size=(1,)).item())
        return super().__call__(subject)

    
        


class CutoutRandomTumor(RandomAugmentation):
    """
    Load tumor mask from brats dataset and generate cutout from tumor mask
    """

    def __init__(self, tumor_seg_paths, p=1.0, pre_load=True, img_size=(256,256,256)):
        self.probability = p
        self.img_size = img_size

        if pre_load:
            self.tumor_masks = []

            for path in tumor_seg_paths:

                tumor_mask = nib.load(path).get_fdata()

                #tumor_mask.data = tumor_mask.get_fdata()[tumor_mask.get_fdata() > 0] = 1
                #tumor_mask = conform(tumor_mask, order=0, resample_only=True).get_fdata() # NN interpolation & no rescaling keeps intesities
                tumor_mask[tumor_mask > 0] = 1
                tumor_mask = tumor_mask.astype(bool)
                tumor_mask = self._padding(tumor_mask, *img_size)

                self.tumor_masks.append(tumor_mask)
        else:
            raise NotImplementedError('only pre-loading is available for now')

    @staticmethod
    def _padding(array, xx, yy, zz):
        """
        :param array: numpy array
        :param xx: desired height
        :param yy: desirex width
        :return: padded array
        """
        h = array.shape[0]
        w = array.shape[1]
        d = array.shape[2]

        a = (xx - h) // 2
        aa = xx - a - h

        b = (yy - w) // 2
        bb = yy - b - w

        c = (zz - d) // 2
        cc = zz - c - d

        return np.pad(array, pad_width=((a, aa), (b, bb), (c, cc)), mode='constant')

    @staticmethod
    def random_small_offset(x, max=None):
        x = x + random.randint(0,4)
        x = 0 if x<0 else x
        if max is not None:
            x = max if x>max else x
        return x


    def bbox_3D(self, img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        # dilate by 5 mm
        rmin -= 5
        rmax += 5
        cmin -= 5
        cmax += 5
        zmin -= 5
        zmax += 5

        # add variance to tumor masks and check for out of bounds
        rmin = self.random_small_offset(rmin, max=self.img_size[0])
        rmax = self.random_small_offset(rmax, max=self.img_size[0])
        
        cmin = self.random_small_offset(cmin, max=self.img_size[1])
        cmax = self.random_small_offset(cmax, max=self.img_size[1])

        zmin = self.random_small_offset(zmin, max=self.img_size[2])
        zmax = self.random_small_offset(zmax, max=self.img_size[2])


        bounding_mask = np.zeros(self.img_size, dtype=bool)
        bounding_mask[rmin:rmax, cmin:cmax, zmin:zmax] = 1

        return bounding_mask


    def __call__(self, sample) -> dict:
        if self.decision():
            sample['image'][self.bbox_3D(random.choice(self.tumor_masks))] = 0 
        else:
            pass
        
        return {**sample}



# --------------------- general helper functions

def getConvexHull(subject: Subject):
    """
    Get the convex hull of the labels - approximation for brain mask
    """
    brain_mask = subject['label'].numpy()[:,3].squeeze() > 0
    return Delaunay(np.array(brain_mask.nonzero()).T), brain_mask

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


def getBrainCenter(subject: Subject):
    """
    get the center of the labels
    """
    return torch.round(torch.mean(subject['label'].data[:,3].nonzero(),axis=0, dtype=torch.float32)).int()

