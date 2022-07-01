
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
import typing as _T
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as _F
import numpy as np


class _ZoomNd(nn.Module):
    def __init__(self, target_shape: _T.Sequence[int], interpolation_mode: str = "nearest"):
        """
        Initialization of Zoom.

        Args:
            target_shape (sequence of ints): Target tensor size for after this module,
                                             not including batchsize and channels.
            interpolation_mode (str): interpolation mode as in `torch.nn.interpolate`
                                      (default: 'neareast')
        """
        super(_ZoomNd, self).__init__()
        self._target_shape = target_shape
        self._mode = interpolation_mode
        self._N = 0

    def forward(self, input_tensor: Tensor, scale_factors: _T.Union[_T.Sequence, float], rescale: bool = False) -> _T.Tuple[Tensor, _T.List[_T.Sequence[float]]]:
        """
        Zoom the the `input_tensor` with `scale_factors`. This is not an exact zoom, but rather an "approximate zoom".
        This is due to the fact that the backbone function only interpolates between integer-sized images and therefore
        the target shape must be rounded to the nearest integer

        Args:
            input_tensor: The tensor of shape (N, C, D_1, ...D_{dim}), where N is the batch size, C is the number of channels
                and D_1, ..., D_{dim} are the dimensions of the image.
            scale_factors: The factor, by which to zoom the image. Can be a torch.Tensor or an array_like (numpy.ndarray
                or a (cascaded) sequence of floats or ints) or a float. If it is a float, all axis and all images of the
                batch are treated the same (zoomed by the float). Else, it will be interpreted as a multi-dimensional
                image: The first dimension corresponds to and must be equal to the batch size of the image. The second
                dimension is optional and may contain different values for the _scale_limits factor per axis. In consequence,
                this dimension can have 1 or {dim} values.
        Returns:
            The zoomed tensor and the zoom factors that were actually used in the calculation for correct rescaling.

        Notes:
            If this Module is used to zoom images of different voxelsizes to the same voxelsize, then `scale_factor`
            should be equal to `target_voxelsize / source_voxelsize`.
        """
        if self._N == 0:
            raise RuntimeError("Direct instantiation of _InterpolateNd is not supported.")

        if input_tensor.dim() != 2 + self._N:
            raise ValueError("Expected {self._N+2}-dimensional input tensor, got {input.dim()}")

        scales = self._fix_scale_factors(scale_factors, input_tensor.shape[0])
        interp = []
        scales_out = []

        # Pytorch Tensor shape BxCxHxW --> loop over batches, interpolate single images, concatenate output at end
        for tensor, scale_f in zip(torch.split(input_tensor, 1, dim=0), scales):
            if rescale:
                if isinstance(scale_f, list):
                    scale_f = [1/sf for sf in scale_f]
                else:
                    scale_f = torch.div(1, scale_f)
            i, sf = self._interpolate(tensor, scale_f)
            interp.append(i)
            scales_out.append(sf)

        return torch.cat(interp, dim=0), scales_out

    def _fix_scale_factors(self, scale_factors: _T.Union[_T.Sequence, float], batch_size: int) -> _T.Sequence[_T.Sequence[float]]:
        """
        Checking and fixing the conformity of scale_factors.
        """
        # add same check for tensor
        if isinstance(scale_factors, np.ndarray):
            scale_factors = scale_factors.tolist()

        else:
            if hasattr(scale_factors, '__len__'):
                scale_factors = list(scale_factors)

            else:
                scale_factors = scale_factors

        if isinstance(scale_factors, list):

            if batch_size != len(scale_factors):
                raise ValueError("scale_factors is a Sequence, but not the same length as the batch-size.")

            # Loop over batches
            for i, sf in enumerate(scale_factors):
                sf = sf.tolist() if isinstance(sf, np.ndarray) else sf
                if isinstance(sf, _T.Sequence):
                    if len(sf) == 1 and self._N != 1:
                        scale_factors[i] = list(sf) * self._N

                    if len(sf) < self._N:
                        raise ValueError(f"scale_factors is a Sequence containing at least one Sequence of length "
                                         f"not equal to 1 or {self._N}.")
                else:
                    #scale_factors[i] = (sf,) * self._N # add tensor to sequence thingy
                    scale_factors[i] = (sf[0], sf[1])


        elif isinstance(scale_factors, float):
            scale_factors = ((scale_factors,) * self._N,) * batch_size

        else:
            raise ValueError("scale_factors is not the correct type, must be sequence of floats or float.")
        return scale_factors

    def _interpolate(self, *args) -> _T.Tuple[Tensor, _T.Sequence[float]]:
        raise NotImplementedError

    def _calculate_crop_pad(self, in_shape: _T.Sequence[int], scale_factor: _T.List[float], dim: int, alignment: str) -> _T.Tuple[slice, _T.List[float], _T.Union[bool, _T.Tuple[int, int]], int]:
        """
        Return start- and end- coordinate given sizes, the updated scale factor
        """
        this_in_shape = in_shape[dim + 2]
        source_size = self._target_shape[dim] * scale_factor[dim]

        # default no-cropping values for start and end
        start = 0
        end = this_in_shape

        # calculate out and rescale_factor
        if alignment == 'mid' and (this_in_shape - int(source_size)) % 2 != 0:
            out = int(source_size / 2) * 2

        else:
            out = int(source_size)

        rescale_factor = out / self._target_shape[dim]

        if out > this_in_shape:
            # set distribution of padding for before and after
            if alignment == 'from_end':
                padding = (1., 0.)

            elif alignment == 'mid':
                padding = (0.5, 0.5)

            else:
                padding = (0., 1.)

            interp_target_shape = int(this_in_shape / scale_factor[dim])

            if alignment == 'mid' and (interp_target_shape - self._target_shape[dim]) % 2 != 0:
                raise NotImplementedError("not tested yet!")
                interp_target_shape = interp_target_shape - 1

            rescale_factor = this_in_shape / interp_target_shape

            # update padding to match full_pad
            full_pad = self._target_shape[dim] - interp_target_shape
            padding = tuple(int(p * full_pad) for p in padding)

        else:
            # update cropping to fit alignment
            if alignment == 'mid':
                start = int((end - out) / 2)

            if alignment == 'from_end':
                start = end - out

            else:
                end = start + out

            # default values for padding and target_shape
            padding = False
            interp_target_shape = self._target_shape[dim]

        scale_factor[dim] = rescale_factor

        return slice(start, end), scale_factor, padding, interp_target_shape


class Zoom2d(_ZoomNd):
    """
    Performs a crop and interpolation on a Four-dimensional Tensor respecting batch and channel.
    """

    def __init__(self, target_shape: _T.Sequence[int], interpolation_mode: str = "nearest",
                 crop_position: str = "top_left"):
        """
        Initialization of Interpolation.

        Args:
            target_shape (len 2): Target tensor size for after this module, not including batchsize and channels.
            interpolation_mode: interpolation mode as in `torch.nn.interpolate` (default: 'neareast')
            crop_position: crop position to use from 'top_left', 'bottom_left', top_right', 'bottom_right',
                          'center' (default: 'top_left')
        """
        if not isinstance(target_shape, _T.Sequence) or len(target_shape) != 2:
            raise ValueError("target_shape must be a sequence of ints of length 2")

        if interpolation_mode not in ["nearest", "bilinear", "bicubic", "area"]:
            raise ValueError(f"invalid interpolation_mode, got {interpolation_mode}")

        if crop_position not in ['top_left', 'bottom_left', 'top_right', 'bottom_right', 'center']:
            raise ValueError(f"invalid crop_position, got {crop_position}")

        super(Zoom2d, self).__init__(target_shape, interpolation_mode)
        self._crop_position = crop_position
        self._N = 2

    def _interpolate(self, tensor: Tensor, scale_factor: _T.Sequence[float]) -> _T.Tuple[Tensor, _T.Sequence[float]]:
        """
        Crops, interpolates and pads the tensor according to the scale_factor. scale_factor must be 2-length
        sequence.

        :param tensor:
        :param scale_factor:
        :return:
        """
        if not isinstance(scale_factor, _T.Sequence) or len(scale_factor) != 2:
            raise ValueError("target_shape must be a sequence of ints of length 2")

        c = self._crop_position == "center"
        vertical_alignment, horizontal_alignment = ('from_start',) * 2

        if self._crop_position == "center":
            vertical_alignment, horizontal_alignment = ('mid',) * 2

        if self._crop_position.startswith("bottom"):
            vertical_alignment = 'from_end'

        if self._crop_position.endswith("right"):
            horizontal_alignment = 'from_end'

        scale_factor = list(scale_factor)
        top_bottom, scale_factor, pad_tb, shape_tb = self._calculate_crop_pad(tensor.shape, scale_factor, 0,
                                                                              vertical_alignment)
        left_right, scale_factor, pad_lr, shape_lr = self._calculate_crop_pad(tensor.shape, scale_factor, 1,
                                                                              horizontal_alignment)

        if isinstance(pad_tb, tuple) or isinstance(pad_lr, tuple):
            def _ensure_tuple(x: _T.Union[bool, _T.Tuple[int, int]]) -> _T.Tuple[int, int]:
                return x if isinstance(x, tuple) else (0, 0)

            padding = list(_ensure_tuple(pad_lr) + _ensure_tuple(pad_tb))

        else:
            padding = None
        interp = _F.interpolate(tensor[:, :, top_bottom, left_right], size=(shape_tb, shape_lr), mode=self._mode, align_corners=False)
        return (_F.pad(interp, padding) if padding is not None else interp), scale_factor


class Zoom3d(_ZoomNd):
    """
    Performs a crop and interpolation on a Five-dimensional Tensor respecting batch and channel.
    """
    def __init__(self, target_shape:_T.Sequence[int], interpolation_mode:str="nearest", crop_position:str="front_top_left"):
        """
        Initialization of Interpolation.
        Args:
            target_shape (len 2): Target tensor size for after this module,
                not including batchsize and channels.
            interpolation_mode: interpolation mode as in `torch.nn.interpolate`
                (default: 'neareast')
            crop_position: crop position to use from 'front_top_left', 'back_top_left',
                'front_bottom_left', 'back_bottom_left', 'front_top_right', 'back_top_right',
                'front_bottom_right', 'back_bottom_right', 'center' (default: 'front_top_left')
        """
        if not isinstance(target_shape, _T.Sequence) or len(target_shape) != 3:
            raise ValueError("target_shape must be a sequence of ints of length 3")

        if interpolation_mode not in ["nearest", "trilinear", "area"]:
            raise ValueError(f"invalid interpolation_mode, got {interpolation_mode}")

        if crop_position not in ['front_top_left', 'back_top_left',
                'front_bottom_left', 'back_bottom_left', 'front_top_right', 'back_top_right',
                'front_bottom_right', 'back_bottom_right', 'center']:
            raise ValueError(f"invalid crop_position, got {crop_position}")

        super(Zoom3d, self).__init__(target_shape, interpolation_mode)
        self._crop_position = crop_position
        self._N = 3

    def _interpolate(self, tensor:Tensor, scale_factor:_T.Sequence[int]):
        """
        Crops, interpolates and pads the tensor acccording to the scale_factor. scale_factor must be 2-length sequence.
        """
        if not isinstance(scale_factor, _T.Sequence) or len(scale_factor) != 3:
            raise ValueError("target_shape must be a sequence of floats of length 3")

        c = self._crop_position == "center"
        depth_alignment, vertical_alignment, horizontal_alignment = ('from_start',) *3

        if self._crop_position == "center":
            depth_alignment, vertical_alignment, horizontal_alignment = ('mid',) *3

        if self._crop_position.startswith("back"):
            depth_alignment = 'from_end'

        if "_bottom_" in self._crop_position:
            vertical_alignment = 'from_end'

        if self._crop_position.endswith("right"):
            horizontal_alignment = 'from_end'

        front_back, scale_factor, pad_fb, shape_fb = self._calculate_crop_size(tensor.shape, scale_factor, 0, depth_alignment)
        top_bottom, scale_factor, pad_tb, shape_tb = self._calculate_crop_pad(tensor.shape, scale_factor, 1, vertical_alignment)
        left_right, scale_factor, pad_lr, shape_lr = self._calculate_crop_pad(tensor.shape, scale_factor, 2, horizontal_alignment)
        needs_padding = pad_tb or pad_lr or pad_fb
        interp = _F.interpolate(tensor[:, :, front_back, top_bottom, left_right], size=(shape_fb, shape_tb, shape_lr), mode=self._mode) #, align_corners=False)

        return (_F.pad(interp, pad_lr + pad_tb + pad_fb) if needs_padding else interp), scale_factor

