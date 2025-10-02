# Copyright 2025 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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
from monai.transforms import MapTransform, RandomizableTransform


class CropAroundACPC(RandomizableTransform, MapTransform):
    """Crop image around anterior and posterior commissure points.

    A transform that crops the input image around the AC and PC points with
    optional padding and random translation.

    Parameters
    ----------
    keys : list[str]
        Keys of the data dictionary to apply the transform to
    allow_missing_keys : bool, optional
        Whether to allow missing keys in the data dictionary, by default False
    padding_mm : float, optional
        Padding around AC-PC region in millimeters, by default 10
    random_translate : float, optional
        Maximum random translation in voxels, by default 0

    Notes
    -----
    The transform expects the following keys in the data dictionary:
    - AC_center : np.ndarray
        Coordinates of anterior commissure
    - PC_center : np.ndarray
        Coordinates of posterior commissure
    - res : float
        Voxel resolution in mm
    """
    
    def __init__(self, keys: list[str], allow_missing_keys: bool = False, 
                 padding_mm: float = 10, random_translate: float = 0) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=1, do_transform=True)
        self.padding_mm = padding_mm
        self.random_translate = random_translate

    def __call__(self, data: dict) -> dict:
        """Apply the transform to the data.

        Parameters
        ----------
        data : dict
            Dictionary containing the data to transform

        Returns
        -------
        dict
            Transformed data dictionary
        """
        d = dict(data)

        if 'AC_center_original' not in d:
            d['AC_center_original'] = d['AC_center'].copy()
        if 'PC_center_original' not in d:
            d['PC_center_original'] = d['PC_center'].copy()

        if self.random_translate > 0:
            random_translate = np.random.randint(-self.random_translate, self.random_translate, size=2)
        else:
            random_translate = (0,0,0)
        
        for key in self.keys:
            if key not in d.keys() and self.allow_missing_keys:
                continue

            pc_center = d['PC_center']
            ac_center = d['AC_center']
            
            ac_pc_bottomleft = (np.min([ac_center[1], pc_center[1]]).astype(int), 
                               np.min([ac_center[2], pc_center[2]]).astype(int))
            ac_pc_topright = (np.max([ac_center[1], pc_center[1]]).astype(int), 
                             np.max([ac_center[2], pc_center[2]]).astype(int))

            voxel_padding = round(self.padding_mm / d['res'])

            crop_left = ac_pc_bottomleft[0]-int(voxel_padding*1.5)+random_translate[0]
            crop_right = ac_pc_topright[0]+voxel_padding//2+random_translate[0]
            crop_top = ac_pc_bottomleft[1]-voxel_padding+random_translate[1]
            crop_bottom = ac_pc_topright[1]+voxel_padding+random_translate[1]

            d['to_pad'] = crop_left, d[key].shape[2]-crop_right, crop_top, d[key].shape[3]-crop_bottom
            d[key] = d[key][:, :, crop_left:crop_right, crop_top:crop_bottom]

        return d


class CropAroundACPCtrack(CropAroundACPC):
    """Crop image around AC-PC points and update their coordinates.

    Extends CropAroundACPC to also adjust the AC and PC center coordinates
    after cropping to maintain their correct positions in the cropped image.

    Parameters
    ----------
    keys : list[str]
        Keys of the data dictionary to apply the transform to
    allow_missing_keys : bool, optional
        Whether to allow missing keys in the data dictionary, by default False
    padding_mm : float, optional
        Padding around AC-PC region in millimeters, by default 10
    random_translate : float, optional
        Maximum random translation in voxels, by default 0

    Notes
    -----
    The transform expects the following keys in the data dictionary:
    - AC_center : np.ndarray
        Coordinates of anterior commissure
    - PC_center : np.ndarray
        Coordinates of posterior commissure
    - AC_center_original : np.ndarray
        Original coordinates of anterior commissure
    - PC_center_original : np.ndarray
        Original coordinates of posterior commissure
    """

    def __call__(self, data: dict) -> dict:
        """Apply the transform to the data.

        Parameters
        ----------
        data : dict
            Dictionary containing the data to transform

        Returns
        -------
        dict
            Transformed data dictionary with updated AC and PC coordinates
        """

        
        # First call parent class to get cropped image
        d = super().__call__(data)
        
        # Get the crop coordinates that were used
        pad_left, pad_right, pad_top, pad_bottom = d['to_pad']

        # Adjust AC and PC center coordinates based on cropping
        if 'AC_center' in d:
            d['AC_center'][1] = d['AC_center_original'][1] - pad_left.item()
            d['AC_center'][2] = d['AC_center_original'][2] - pad_top.item()
            
        if 'PC_center' in d:
            d['PC_center'][1] = d['PC_center_original'][1] - pad_left.item() 
            d['PC_center'][2] = d['PC_center_original'][2] - pad_top.item()

        return d
    
