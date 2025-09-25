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
    """
    Crop around AC and PC with fixed size
    """

    def __init__(self, keys, fixed_size: tuple[int, int], allow_missing_keys: bool = False, 
                 random_translate: float = 0) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self)
        self.random_translate = random_translate
        self.fixed_size = fixed_size


    def __call__(self, data):
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
        voxel_padding_x = self.fixed_size[0] // 2
        voxel_padding_y = self.fixed_size[1] // 2

        # Add random translation if specified
        if self.random_translate > 0:
            random_translate = np.random.randint(-self.random_translate, 
                                               self.random_translate, size=2)
        else:
            random_translate = (0,0)



        # Calculate crop boundaries with padding and random translation
        crop_left = center_point[1] - voxel_padding_x + random_translate[0]
        crop_right = center_point[1] + voxel_padding_x + random_translate[0]
        crop_top = center_point[2] - voxel_padding_y + random_translate[1] 
        crop_bottom = center_point[2] + voxel_padding_y + random_translate[1]

        # Ensure crop boundaries are within image
        #img_shape = d['image'].shape[2:]  # Get spatial dimensions
        # crop_left = max(0, crop_left)
        # crop_right = min(img_shape[0], crop_right)
        # crop_top = max(0, crop_top)
        # crop_bottom = min(img_shape[1], crop_bottom)

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
