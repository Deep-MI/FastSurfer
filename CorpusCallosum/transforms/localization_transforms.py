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

from logging import getLogger

import numpy as np
import torch
from monai.transforms import MapTransform, RandomizableTransform


class CropAroundACPCFixedSize(RandomizableTransform, MapTransform):
    """Crop image around AC-PC points with fixed size.

    A transform that crops the input image around the midpoint between
    AC and PC points with a fixed size window and optional random translation.

    Parameters
    ----------
    keys : list[str]
        Keys of the data dictionary to apply the transform to.
    fixed_size : tuple[int, int]
        Fixed size of the crop window (width, height).
    allow_missing_keys : bool, optional
        Whether to allow missing keys in the data dictionary, by default False.
    random_translate : int, default=0
        Maximum random translation in voxels.

    Raises
    ------
    ValueError
        If the crop boundaries extend outside the image dimensions.

    Notes
    -----
    The transform expects the following keys in the data dictionary:

    - AC_center : np.ndarray
        Coordinates of anterior commissure
    - PC_center : np.ndarray
        Coordinates of posterior commissure
    - image : np.ndarray
        Input image to crop

    """

    def __init__(
            self,
            keys: list[str],
            fixed_size: tuple[int, int],
            allow_missing_keys: bool = False,
            random_translate: int = 0,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self)
        self.random_translate = random_translate
        self.fixed_size = fixed_size

    def __call__(self, data: dict) -> dict:
        """Apply the 2D crop transform to the data.

        Parameters
        ----------
        data : dict
            Dictionary containing the data to transform AND keys AC_center and PC_center, each of shape (B, 2).

        Returns
        -------
        dict
            Transformed data dictionary with cropped images and updated coordinates.
            Also includes crop boundary information:
            - crop_left : list[int]
            - crop_right : list[int]
            - crop_top : list[int]
            - crop_bottom : list[int]

        Raises
        ------
        ValueError
            If crop boundaries extend outside the image dimensions
        """
        d = dict(data)

        expected_keys = {"PC_center", "AC_center"} | set(self.keys) if not self.allow_missing_keys else {}

        if expected_keys & set(d.keys()) != expected_keys:
            raise ValueError(f"The following keys are missing in the data dictionary: {expected_keys - set(d.keys())}!")

        if any(d[k].ndim != 2 or d[k].shape[1] != 2 for k in ["PC_center", "AC_center"]):
            raise ValueError("Shape of AC_center or PC_center incorrect, must be (B, 2)!")

        if any(d[k].ndim != 4 for k in self.keys if k in d.keys()):
            raise ValueError(f"At least one key of {self.keys} does not have a 4-dimensional tensor.")

        # calculate center point between AC and PC
        center_point = ((d['AC_center'] + d['PC_center']) / 2).astype(int)

        # Calculate voxel padding based on mm padding
        voxel_padding = np.asarray(self.fixed_size) // 2

        existing_keys = set(self.keys) & set(d.keys())
        if len(existing_keys) == 0:
            getLogger(__name__).warning(f"None of the keys in {self.keys} are present in the data dictionary!")
            return d

        first_key = tuple(existing_keys)[0]

        # Calculate crop boundaries with padding and random translation
        crops = center_point - voxel_padding

        # Add random translation if specified
        if self.random_translate > 0:
            crops += np.random.randint(
                -self.random_translate,
                self.random_translate + 1,
                size=(d[first_key].shape[0], 2),
            )

        # Ensure crop boundaries are within image
        img_shape = np.asarray(d[first_key].shape[2:])  # Get spatial dimensions
        if any(np.any(img_shape != d[k].shape[2:]) for k in self.keys if k in d.keys()):
            raise ValueError(f"At least one key of {self.keys} does not have the expected shape.")

        patch_size_with_batch_dim = np.asarray(self.fixed_size)[None]
        crops = np.maximum(0, np.minimum(img_shape, crops + patch_size_with_batch_dim) - patch_size_with_batch_dim)
        d["crop_left"], d["crop_top"] = crops.T.tolist()
        d["crop_right"], d["crop_bottom"] = (crops_end := crops + patch_size_with_batch_dim).T.tolist()

        # raise error if crop boundaries are out of image
        if np.any(crops < 0) or np.any(crops_end > np.asarray([d[first_key].shape[2:]])):
            raise ValueError("Crop boundaries are out of image")

        # Apply crop to image
        for key in self.keys:
            if key not in d.keys() and self.allow_missing_keys:
                continue
            arr = [v[:, cl:cr, ct:cb] for v, cl, ct, cr, cb in zip(d[key], *crops.T, *crops_end.T, strict=True)]
            d[key] = torch.stack(arr, dim=0) if torch.is_tensor(arr[0]) else np.stack(arr, axis=0)

        # Update point coordinates relative to cropped image
        d["PC_center"] = d["PC_center"] - crops
        d["AC_center"] = d["AC_center"] - crops
        return d
