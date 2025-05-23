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
from numbers import Number, Real
from typing import Any

import numpy as np
import numpy.typing as npt
import torch


##
# Transformations for evaluation
##
class ToTensorTest:
    """
    Convert np.ndarrays in sample to Tensors.

    Methods
    -------
    __call__
        Converts image.
    """

    def __call__(self, img: npt.NDArray) -> np.ndarray:
        """
        Convert the image to float within range [0, 1] and make it torch compatible.

        Parameters
        ----------
        img : npt.NDArray
            Image to be converted.

        Returns
        -------
        img : np.ndarray
            Conformed image.
        """
        img = img.astype(np.float32)

        # Normalize and clamp between 0 and 1
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        return img


class ZeroPad2DTest:
    """
    Pad the input with zeros to get output size.

    Attributes
    ----------
    output_size : Union[Number, Tuple[Number, Number]]
        Size of the output image either as Number or tuple of two Number.
    pos : str
        Position to put the input.

    Methods
    -------
    pad
        Pad zeroes of image.
    call
        Call _pad().
    """
    def __init__(
            self,
            output_size: Number | tuple[Number, Number],
            pos: str = 'top_left'
    ):
        """
        Construct object.

        Parameters
        ----------
        output_size : Union[Number, Tuple[Number, Number]]
            Size of the output image either as Number or tuple of two Number.
        pos : Union[Number, Tuple[Number, Number]]
            Position to put the input. Defaults to 'top_left'.
        """
        if isinstance(output_size, Number):
            output_size = (int(output_size),) * 2
        self.output_size = output_size
        self.pos = pos

    def _pad(self, image: npt.NDArray) -> np.ndarray:
        """
        Pad with zeros of the input image.

        Parameters
        ----------
        image : npt.NDArray
            The image to pad.

        Returns
        -------
        padded_img : np.ndarray
            Original image with padded zeros.
        """
        if len(image.shape) == 2:
            h, w = image.shape
            padded_img = np.zeros(self.output_size, dtype=image.dtype)
        else:
            h, w, c = image.shape
            padded_img = np.zeros(self.output_size + (c,), dtype=image.dtype)

        if self.pos == "top_left":
            padded_img[0:h, 0:w] = image

        return padded_img

    def __call__(self, img: npt.NDArray) -> np.ndarray:
        """
        Call the _pad() function.

        Parameters
        ----------
        img : npt.NDArray
            The image to pad.

        Returns
        -------
        img : np.ndarray
            Original image with padded zeros.
        """
        img = self._pad(img)

        return img


##
# Transformations for training
##
class ToTensor:
    """
    Convert ndarrays in sample to Tensors.
    
    Methods
    -------
    __call__
        Convert image.
    """

    def __call__(self, sample: npt.NDArray) -> dict[str, Any]:
        """
        Convert the image to float within range [0, 1] and make it torch compatible.

        Parameters
        ----------
        sample : npt.NDArray
            Sample image.

        Returns
        -------
        Dict[str, Any]
            Converted image.
        """
        img, label, weight, sf = (
            sample["img"],
            sample["label"],
            sample["weight"],
            sample["scale_factor"],
        )
        
        img = img.astype(np.float32)

        # Normalize image and clamp between 0 and 1
        img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        return {
            "img": torch.from_numpy(img),
            "label": torch.from_numpy(label),
            "weight": torch.from_numpy(weight),
            "scale_factor": torch.from_numpy(sf),
        }


class ZeroPad2D:
    """
    Pad the input with zeros to get output size.

    Attributes
    ----------
    output_size : Union[Number, Tuple[Number, Number]]
        Size of the output image either as Number or tuple of two Number.
    pos : str, Optional
        Position to put the input.

    Methods
    -------
    _pad
        Pads zeroes of image.
    __call__
        Cals _pad for sample.
    """
    def __init__(
            self,
            output_size: Number | tuple[Number, Number],
            pos: None | str = 'top_left'
    ):
        """
        Initialize position and output_size (as Tuple[float]).

        Parameters
        ----------
        output_size : Union[Number, Tuple[Number, Number]]
            Size of the output image either as Number or
            tuple of two Number.
        pos : str, Optional
            Position to put the input. Default = 'top_left'.
        """
        if isinstance(output_size, Number):
            output_size = (int(output_size),) * 2
        self.output_size = output_size
        self.pos = pos

    def _pad(self, image: npt.NDArray) -> np.ndarray:
        """
        Pad the input image with zeros.

        Parameters
        ----------
        image : npt.NDArray
            The image to pad.

        Returns
        -------
        padded_img : np.ndarray
            Original image with padded zeros.
        """
        if len(image.shape) == 2:
            h, w = image.shape
            padded_img = np.zeros(self.output_size, dtype=image.dtype)
        else:
            h, w, c = image.shape
            padded_img = np.zeros(self.output_size + (c,), dtype=image.dtype)

        if self.pos == "top_left":
            padded_img[0:h, 0:w] = image

        return padded_img

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Pad the image, label and weights.

        Parameters
        ----------
        sample : Dict[str, Any]
            Sample image.

        Returns
        -------
        Dict[str, Any]
            Dictionary including the padded image, label, weight and scale factor.
        """
        img, label, weight, sf = (
            sample["img"],
            sample["label"],
            sample["weight"],
            sample["scale_factor"],
        )
        
        img = self._pad(img)
        label = self._pad(label)
        weight = self._pad(weight)

        return {"img": img, "label": label, "weight": weight, "scale_factor": sf}


class AddGaussianNoise:
    """
    Add gaussian noise to sample.

    Attributes
    ----------
    std
        Standard deviation.
    mean
        Gaussian mean.

    Methods
    -------
    __call__
        Adds noise to scale factor.
    """
    def __init__(self, mean: Real = 0, std: Real = 0.1):
        """
        Construct object.

        Parameters
        ----------
        mean : Real
            Standard deviation. Default = 0.
        std : Real
            Gaussian mean. Default = 0.1.
        """
        self.std = std
        self.mean = mean

    def __call__(self, sample: dict[str, Real]) -> dict[str, Real]:
        """
        Add gaussian noise to scalefactor.

        Parameters
        ----------
        sample : Dict[str, Real]
            Sample data to add noise.

        Returns
        -------
        Dict[str, Real]
            Sample with noise.
        """
        img, label, weight, sf = (
            sample["img"],
            sample["label"],
            sample["weight"],
            sample["scale_factor"],
        )
        # change 1 to sf.size() for isotropic scale factors (now same noise change added to both dims)
        sf = sf + torch.randn(1) * self.std + self.mean
        return {"img": img, "label": label, "weight": weight, "scale_factor": sf}


class AugmentationPadImage:
    """
    Pad Image with either zero padding or reflection padding of img, label and weight.

    Attributes
    ----------
    pad_size_image : tuple
        The padding size for the image.
    pad_size_mask : tuple
        The padding size for the mask.
    pad_type : str
        The type of padding to be applied.

    Methods
    -------
     __call
        Add zeroes.
    """
    def __init__(
            self,
            pad_size: tuple[tuple[int, int],
            tuple[int, int]] = ((16, 16), (16, 16)),
            pad_type: str = "edge"
    ):
        """
        Construct object.

        Parameters
        ----------
        pad_size : tuple
            The padding size.
        pad_type : str
            The type of padding to be applied.
        """
        assert isinstance(pad_size, int | tuple)

        if isinstance(pad_size, int):

            # Do not pad along the channel dimension
            self.pad_size_image = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
            self.pad_size_mask = ((pad_size, pad_size), (pad_size, pad_size))

        else:
            self.pad_size = pad_size

        self.pad_type = pad_type

    def __call__(self, sample: dict[str, Number]):
        """
        Pad zeroes of sample image, label and weight.

        Attributes
        ----------
        sample : Dict[str, Number]
            Sample image and data.
        """
        img, label, weight, sf = (
            sample["img"],
            sample["label"],
            sample["weight"],
            sample["scale_factor"],
        )

        img = np.pad(img, self.pad_size_image, self.pad_type)
        label = np.pad(label, self.pad_size_mask, self.pad_type)
        weight = np.pad(weight, self.pad_size_mask, self.pad_type)

        return {"img": img, "label": label, "weight": weight, "scale_factor": sf}


class AugmentationRandomCrop:
    """
    Randomly Crop Image to given size.
    """

    def __init__(self, output_size: int | tuple, crop_type: str = 'Random'):
        """Construct object.

        Attributes
        ----------
        output_size
            Size of the output image either an integer or a tuple.
        crop_type
        The type of crop to be performed.
        """
        assert isinstance(output_size, int | tuple)

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

        else:
            self.output_size = output_size

        self.crop_type = crop_type

    def __call__(self, sample: dict[str, Number]) -> dict[str, Number]:
        """
        Crops the augmentation.

        Attributes
        ----------
        sample : Dict[str, Number]
            Sample image with data.

        Returns
        -------
        Dict[str, Number]
            Cropped sample image.
        """
        img, label, weight, sf = (
            sample["img"],
            sample["label"],
            sample["weight"],
            sample["scale_factor"],
        )

        h, w, _ = img.shape

        if self.crop_type == "Center":
            top = (h - self.output_size[0]) // 2
            left = (w - self.output_size[1]) // 2

        else:
            top = np.random.randint(0, h - self.output_size[0])
            left = np.random.randint(0, w - self.output_size[1])

        bottom = top + self.output_size[0]
        right = left + self.output_size[1]

        # print(img.shape)
        img = img[top:bottom, left:right, :]
        label = label[top:bottom, left:right]
        weight = weight[top:bottom, left:right]

        return {"img": img, "label": label, "weight": weight, "scale_factor": sf}
