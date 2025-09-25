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
    """
    Crop around AC and PC
    """
    
    def __init__(self, keys, allow_missing_keys: bool = False, padding_mm: float = 10, 
                 random_translate: float = 0) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=1, do_transform=True)
        self.padding_mm = padding_mm
        self.random_translate = random_translate

    def __call__(self, data):
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
            
            # 'PC_center': array([  2., 139., 143.], dtype=float32), 'AC_center': array([  2., 128., 168.]
            
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


            #d[key] = d[key][:, d[key].shape[1]//2-voxel_padding:d[key].shape[2]//2+voxel_padding]

            #print('cropped', d[key].shape, 'for key', key)

        return d

class CropAroundACPCtrack(CropAroundACPC):
    """
    Same as crop around ACPC but also adjusts AC_center and PC_center accordingly
    
    
    """

    def __call__(self, data):

        
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
    
