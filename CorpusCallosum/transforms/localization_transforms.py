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


class CropAroundACPCFixedSize(RandomizableTransform, MapTransform):
    """Crop image around AC-PC points with fixed size.

    A transform that crops the input image around the midpoint between
    AC and PC points with a fixed size window and optional random translation.

    Parameters
    ----------
    keys : list[str]
        Keys of the data dictionary to apply the transform to
    fixed_size : tuple[int, int]
        Fixed size of the crop window (width, height)
    allow_missing_keys : bool, optional
        Whether to allow missing keys in the data dictionary, by default False
    random_translate : float, optional
        Maximum random translation in voxels, by default 0

    Notes
    -----
    The transform expects the following keys in the data dictionary:
    - AC_center : np.ndarray
        Coordinates of anterior commissure
    - PC_center : np.ndarray
        Coordinates of posterior commissure
    - image : np.ndarray
        Input image to crop

    Raises
    ------
    ValueError
        If the crop boundaries extend outside the image dimensions
    """

    def __init__(self, keys: list[str], fixed_size: tuple[int, int], 
                 allow_missing_keys: bool = False, random_translate: float = 0) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self)
        self.random_translate = random_translate
        self.fixed_size = fixed_size

    def __call__(self, data: dict) -> dict:
        """Apply the transform to the data.

        Parameters
        ----------
        data : dict
            Dictionary containing the data to transform

        Returns
        -------
        dict
            Transformed data dictionary with cropped images and updated coordinates.
            Also includes crop boundary information:
            - crop_left : int
            - crop_right : int
            - crop_top : int
            - crop_bottom : int

        Raises
        ------
        ValueError
            If crop boundaries extend outside the image dimensions
        """
        d = dict(data)

        for key in self.keys:
            if key not in d.keys() and self.allow_missing_keys:
                continue

        # Get AC and PC centers
        pc_center = d['PC_center']
        ac_center = d['AC_center']
        
        # calculate center point between AC and PC
        center_point = ((ac_center + pc_center) / 2).astype(int)

        # Calculate voxel padding based on mm padding
        voxel_padding = np.asarray(self.fixed_size) // 2

        # Add random translation if specified
        if self.random_translate > 0:
            random_translate = np.random.randint(-self.random_translate, 
                                               self.random_translate, size=2)
        else:
            random_translate = np.asarray((0, 0))

        # Calculate crop boundaries with padding and random translation
        crops = center_point - voxel_padding + random_translate
        
        # Ensure crop boundaries are within image
        img_shape = np.asarray(d['image'].shape[2:])  # Get spatial dimensions
        crops = np.maximum(0, np.minimum(img_shape, crops + np.asarray(self.fixed_size)) - np.asarray(self.fixed_size))
        crop_left, crop_top = crops.tolist()
        crop_right, crop_bottom = (crops + np.asarray(self.fixed_size)).tolist()

        # raise error if crop boundaries are out of image
        if crop_left < 0 or crop_right > d['image'].shape[2] or crop_top < 0 or crop_bottom > d['image'].shape[3]:
            raise ValueError("Crop boundaries are out of image")

        # Apply crop to image
        for key in self.keys:
            if key not in d.keys() and self.allow_missing_keys:
                continue
                
            d[key] = d[key][:, :, crop_left:crop_right, crop_top:crop_bottom]

            # Update point coordinates relative to cropped image
            d['PC_center'][1:] = d['PC_center'][1:] - [crop_left, crop_top]
            d['AC_center'][1:] = d['AC_center'][1:] - [crop_left, crop_top]

        
        d['crop_left'] = crop_left
        d['crop_right'] = crop_right
        d['crop_top'] = crop_top
        d['crop_bottom'] = crop_bottom
        return d
