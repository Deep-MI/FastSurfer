# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, TypeVar

import nibabel as nib
import numpy as np
import torch
from numpy import typing as npt

from FastSurferCNN.data_loader.conform import getscale, scalecrop

CLASS_NAMES = {
    "Background": 0,
    "Left_I_IV": 1,
    "Right_I_IV": 2,
    "Left_V": 3,
    "Right_V": 4,
    "Left_VI": 5,
    "Vermis_VI": 6,
    "Right_VI": 7,
    "Left_CrusI": 8,
    "Right_CrusI": 10,
    "Left_CrusII": 11,
    "Right_CrusII": 13,
    "Left_VIIb": 14,
    "Right_VIIb": 16,
    "Vermis_VII": 12,
    "Left_VIIIa": 17,
    "Right_VIIIa": 19,
    "Left_VIIIb": 20,
    "Right_VIIIb": 22,
    "Vermis_VIII": 18,
    "Left_IX": 23,
    "Vermis_IX": 24,
    "Right_IX": 25,
    "Left_X": 26,
    "Vermis_X": 27,
    "Right_X": 28,
    "Left_Corpus_Medullare": 37,
    "Right_Corpus_Medullare": 38,
}

# class names for network training and validation/testing
subseg_labels = {"cereb_subseg": np.array(list(CLASS_NAMES.values()))}

AT = TypeVar("AT", np.ndarray, torch.Tensor)


class LTADict(TypedDict):
    type: int
    nxforms: int
    mean: list[float]
    sigma: float
    lta: npt.NDArray[float]
    src_valid: int
    src_filename: str
    src_volume: list[int]
    src_voxelsize: list[float]
    src_xras: list[float]
    src_yras: list[float]
    src_zras: list[float]
    src_cras: list[float]
    dst_valid: int
    dst_filename: str
    dst_volume: list[int]
    dst_voxelsize: list[float]
    dst_xras: list[float]
    dst_yras: list[float]
    dst_zras: list[float]
    dst_cras: list[float]
    src: npt.NDArray[float]
    dst: npt.NDArray[float]


def define_size(mov_dim, ref_dim):
    new_dim = np.zeros(len(mov_dim), dtype=int)
    borders = np.zeros((len(mov_dim), 2), dtype=int)
    padd = [int(d // 2) for d in mov_dim]
    for i in range(len(mov_dim)):
        new_dim[i] = int(max(2 * mov_dim[i], 2 * ref_dim[i]))
        borders[i, 0] = int(new_dim[i] // 2) - padd[i]
        borders[i, 1] = borders[i, 0] + mov_dim[i]
    return list(new_dim), borders


def map_size(arr, base_shape, return_border=False):
    """Resize the image to base_shape."""
    if arr.shape == base_shape:
        if return_border:
            return arr, np.array(
                [[0, arr.shape[0]], [0, arr.shape[1]], [0, arr.shape[2]]]
            )
        else:
            return arr

    _selection = []
    _pad = []
    _unpad_borders = []

    for i, j in zip(arr.shape, base_shape, strict=False):
        delta = i - j
        left = delta // 2
        if delta > 0:  # crop
            _selection.append(slice(left, left + j))
            _pad.append((0, 0))
        else:
            _selection.append(slice(None, None))
            _pad.append((-left, left - delta))
        _unpad_borders.append([-left, -left + i])

    out = arr[tuple(_selection)]
    if np.any(np.asarray(_pad) != 0):
        out = np.pad(out, _pad)

    if return_border:
        return out, np.asarray(_unpad_borders)
    return out


def map_size_leg(arr, base_shape, return_border=False):
    if arr.shape == base_shape:
        if return_border:
            return arr, np.array(
                [[0, arr.shape[0]], [0, arr.shape[1]], [0, arr.shape[2]]]
            )
        else:
            return arr

    # print('Volume will be resized from %s to %s ' % (arr.shape, base_shape))
    new_shape, borders = define_size(np.array(arr.shape), np.array(base_shape))

    # print("new shape", new_shape)
    new_arr = np.zeros(new_shape, dtype=arr.dtype)
    final_arr = np.zeros(base_shape, dtype=arr.dtype)
    new_arr[
        borders[0, 0] : borders[0, 1],
        borders[1, 0] : borders[1, 1],
        borders[2, 0] : borders[2, 1],
    ] = arr[:]
    middle_point = [
        int(new_arr.shape[0] // 2),
        int(new_arr.shape[1] // 2),
        int(new_arr.shape[2] // 2),
    ]
    padd = [int(base_shape[0] / 2), int(base_shape[1] / 2), int(base_shape[2] / 2)]
    low_border = np.array((np.array(middle_point) - np.array(padd)), dtype=int)
    high_border = np.array(np.array(low_border) + np.array(base_shape), dtype=int)
    final_arr = new_arr[
        low_border[0] : high_border[0],
        low_border[1] : high_border[1],
        low_border[2] : high_border[2],
    ]

    if return_border:
        low_back_border = np.array(borders[:, 0]) - low_border
        high_back_border = low_back_border + np.array(arr.shape)
        back_borders = np.vstack((low_back_border, high_back_border)).T
        back_arr = final_arr[
            back_borders[0, 0] : back_borders[0, 1],
            back_borders[1, 0] : back_borders[1, 1],
            back_borders[2, 0] : back_borders[2, 1],
        ]

        assert np.all(
            back_arr == arr
        ), f"Borders for unpadding are not correct {back_borders}"

        return final_arr, back_borders

    return final_arr


def bounding_volume_offset(
    img: np.ndarray | Sequence[int],
    target_img_size: tuple[int, ...],
    image_shape: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Find the center of the non-zero values in img and returns offsets so this center is in the center of a bounding
    volume of size target_img_size."""
    if isinstance(img, np.ndarray):
        from FastSurferCNN.data_loader.data_utils import bbox_3d

        bbox = bbox_3d(np.not_equal(img, 0))
        bbox = bbox[::2] + bbox[1::2]
    else:
        bbox = img
    center = (
        (_max + _min) / 2
        for _min, _max in zip(bbox[: len(bbox) // 2], bbox[len(bbox) // 2 :], strict=False)
    )
    offset = tuple(
        max(0, int(round(c - ts / 2))) for c, ts in zip(center, target_img_size, strict=False)
    )
    img_shape = (
        image_shape
        if image_shape is not None
        else img.shape
        if hasattr(img, "shape")
        else None
    )
    if img_shape is not None:
        offset = tuple(
            min(max(0, o), imgs - ts)
            for o, ts, imgs in zip(offset, target_img_size, img_shape, strict=False)
        )
        if any(o < 0 for o in offset):
            raise RuntimeError(
                f"Insufficient image size {img_shape} for target image size {target_img_size}"
            )
    return offset


def bounding_volume(img, target_img_size):
    binary_mask = img != 0

    x_axis = np.any(binary_mask, axis=(1, 2))
    y_axis = np.any(binary_mask, axis=(0, 2))
    z_axis = np.any(binary_mask, axis=(0, 1))

    h_min, h_max = np.where(x_axis)[0][[0, -1]]
    w_min, w_max = np.where(y_axis)[0][[0, -1]]
    d_min, d_max = np.where(z_axis)[0][[0, -1]]

    h, w, d = img.shape

    dx = target_img_size[0] - (h_max - h_min)
    dx = (dx // 2, dx - dx // 2)
    dy = target_img_size[1] - (w_max - w_min)
    dy = (dy // 2, dy - dy // 2)
    dz = target_img_size[2] - (d_max - d_min)
    dz = (dz // 2, dz - dz // 2)

    h_min = np.maximum(0, h_min - dx[0])
    h_max = np.minimum(h, h_max + dx[1])

    w_min = np.maximum(0, w_min - dy[0])
    w_max = np.minimum(w, w_max + dy[1])

    d_min = np.maximum(0, d_min - dz[0])
    d_max = np.minimum(d, d_max + dz[1])

    roi = (slice(h_min, h_max), slice(w_min, w_max), slice(d_min, d_max))
    return roi


def rescale_image(img_data):
    # Conform intensities
    src_min, scale = getscale(img_data, 0, 255)
    mapped_data = img_data
    if not img_data.dtype == np.dtype(np.uint8):
        if np.max(img_data) > 255:
            mapped_data = scalecrop(img_data, 0, 255, src_min, scale)

    new_data = np.uint8(np.rint(mapped_data))
    return new_data


def load_reorient(img_filename: str) -> nib.analyze.SpatialImage:
    img_file = nib.load(img_filename)
    canonical_img = nib.as_closest_canonical(img_file)
    return canonical_img


def load_reorient_lia(img_filename: str) -> nib.analyze.SpatialImage:
    return load_reorient(img_filename).as_reoriented([[1, -1], [0, -1], [2, 1]])


def load_reorient_rescale_image(img_filename):
    img_file = load_reorient(img_filename)
    img_data = img_file.get_fdata()
    new_data = rescale_image(img_data)
    return new_data, img_file


def load_and_map_size(filename, size, dtype):
    img_file = load_reorient(filename)
    img_data = np.asarray(img_file.get_fdata(), dtype=dtype)
    img_data, back_borders = map_size(img_data, size, return_border=True)
    return img_data, img_file, back_borders


def filter_blank_slices_thick(data_dict, threshold=10):
    """
    Function to filter blank slices from the volume using the label volume
    :param dict data_dict: dictionary containing all volumes need to be filtered
    :return:
    """
    # Get indices of all slices with more than threshold labels/pixels

    select_slices = np.sum(data_dict["cereb_subseg"], axis=(0, 1)) > threshold
    for name, vol in data_dict.items():
        if name == "orig" or name == "native_img":
            data_dict[name] = vol[:, :, select_slices, :]
        else:
            data_dict[name] = vol[:, :, select_slices]
    return data_dict


def map_label2subseg(mapped_subseg, label_type="cereb_subseg"):
    """
    Function to perform look-up table mapping from label space to subseg space
    """
    labels = subseg_labels[label_type]
    subseg = np.zeros_like(mapped_subseg)
    h, w, d = subseg.shape
    subseg = labels[mapped_subseg.ravel()]

    return subseg.reshape((h, w, d))


# subseg: segmentation of subfield
def map_subseg2label(subseg, label_type="cereb_subseg"):
    h, w, d = subseg.shape
    lbls = subseg_labels[label_type]
    lut_subseg = np.zeros(max(lbls) + 1, dtype="int")
    for idx, value in enumerate(lbls):
        lut_subseg[value] = idx

    mapped_subseg = lut_subseg.ravel()[subseg.ravel()]
    mapped_subseg = mapped_subseg.reshape((h, w, d))
    return mapped_subseg


def apply_warp_field(dform_field, img, interpol_order=3):
    from scipy.ndimage import map_coordinates

    dxyz = np.squeeze(dform_field).transpose(3, 0, 1, 2)
    x, y, z = np.meshgrid(
        np.arange(img.shape[0]),
        np.arange(img.shape[1]),
        np.arange(img.shape[2]),
        indexing="ij",
    )

    indices = x + dxyz[0], y + dxyz[1], z + dxyz[2]
    indices = np.stack(indices, 0)
    deformed_img = map_coordinates(img, indices, order=interpol_order)
    return deformed_img


def read_lta(file: Path | str) -> LTADict:
    """Read the LTA info."""
    import re
    from functools import partial

    import numpy as np
    parameter_pattern = re.compile("^\s*([^=]+)\s*=\s*([^#]*)\s*(#.*)")
    vol_info_pattern = re.compile("^(.*) volume info$")
    shape_pattern = re.compile("^(\s*\d+)+$")
    matrix_pattern = re.compile("^(-?\d+\.\S+\s+)+$")

    _Type = TypeVar("_Type", bound=type)

    def _vector(_a: str, dtype: type[_Type] = float, count: int = -1) -> list[_Type]:
        return np.fromstring(_a, dtype=dtype, count=count, sep=" ").tolist()

    parameters = {
        "type": int,
        "nxforms": int,
        "mean": partial(_vector, dtype=float, count=3),
        "sigma": float,
        "subject": str,
        "fscale": float,
    }
    vol_info_par = {
        "valid": int,
        "filename": str,
        "volume": partial(_vector, dtype=int, count=3),
        "voxelsize": partial(_vector, dtype=float, count=3),
        **{f"{c}ras": partial(_vector, dtype=float) for c in "xyzc"}
    }

    with open(file) as f:
        lines = f.readlines()

    items = []
    shape_lines = []
    matrix_lines = []
    section = ""
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        if hits := parameter_pattern.match(line):
            name = hits.group(1)
            if section and name in vol_info_par:
                items.append((f"{section}_{name}", vol_info_par[name](hits.group(2))))
            elif name in parameters:
                section = ""
                items.append((name, parameters[name](hits.group(2))))
            else:
                raise NotImplementedError(f"Unrecognized type string in lta-file "
                                          f"{file}:{i+1}: '{name}'")
        elif hits := vol_info_pattern.match(line):
            section = hits.group(1)
            # not a parameter line
        elif shape_pattern.search(line):
            shape_lines.append(np.fromstring(line, dtype=int, count=-1, sep=" "))
        elif matrix_pattern.search(line):
            matrix_lines.append(np.fromstring(line, dtype=float, count=-1, sep=" "))

    shape_lines = list(map(tuple, shape_lines))
    lta = dict(items)
    if lta["nxforms"] != len(shape_lines):
        raise OSError("Inconsistent lta format: nxforms inconsistent with shapes.")
    if len(shape_lines) > 1 and np.any(np.not_equal([shape_lines[0]], shape_lines[1:])):
        raise OSError(f"Inconsistent lta format: shapes inconsistent {shape_lines}")
    lta_matrix = np.asarray(matrix_lines).reshape((-1,) + shape_lines[0].shape)
    lta["lta"] = lta_matrix
    return lta


def load_talairach_coordinates(tala_path, img_shape, vox2ras):
    tala_lta = read_lta(tala_path)
    # create image grid p
    x, y, z = np.meshgrid(
        np.arange(img_shape[0]),
        np.arange(img_shape[1]),
        np.arange(img_shape[2]),
        indexing="ij",
    )
    p = np.array([x.flatten(), y.flatten(), z.flatten()]).transpose()
    p1 = np.concatenate((p, np.ones((p.shape[0], 1))), axis=1)

    assert tala_lta["type"] == 1, "talairach not in ras2ras"  # ras2ras
    m = np.matmul(tala_lta["lta"][0, 0], vox2ras)

    tala_coordinates = np.matmul(m, p1.transpose()).transpose()
    tala_coordinates = tala_coordinates[:, :-1]
    tala_coordinates = tala_coordinates.reshape(*img_shape, 3).astype(np.float16)
    return tala_coordinates


def normalize_array(arr):
    min = arr.min()
    max = arr.max()

    arr = arr - min
    if min == max:
        return arr
    arr = arr / (max - min)
    return arr


def _crop_transform_make_indices(image_shape, offsets, target_shape):
    """
    Create the indexing tuple and return padding tuples for the last N dimensions.

    Parameters
    ----------
    image_shape : np.ndarray
        The shape of the image from which a region is to be cropped.
    offsets : Sequence[int]
        Exact location within the image from which the cropping should start.
    target_shape : Sequence[int], optional
        The desired shape of the cropped region.

    Returns
    -------
    paddings: list of 2-tuples of paddings or None
        A list of per-axis tuples of the padding to apply to the slice to get the target_shape.
    indices : tuple of indices
        A tuple of per-axis indices to index in the data to get the target_shape.
    """
    if len(offsets) != len(target_shape):
        raise ValueError(
            f"offsets {offsets} and target shape {target_shape} must be same length."
        )
    if len(offsets) > len(image_shape):
        raise ValueError("offsets too long for image")
    batch_dims = len(image_shape) - len(offsets)
    indices = [slice(None)] * batch_dims
    paddings = []
    any_pad = False
    for offset, t_shape, i_shape in zip(
        offsets, target_shape, image_shape[batch_dims:], strict=False
    ):
        crop_end = min(offset + t_shape, i_shape)
        indices.append(slice(max(0, offset), crop_end))
        pads = (max(0, -offset), max(0, offset + t_shape - crop_end))
        paddings.append(pads)
        any_pad = any_pad or any(p != 0 for p in pads)

    return paddings if any_pad else None, tuple(indices)


def _crop_transform_pad_fn(image, pad_tuples, pad):
    """
    Generate a parameterized pad function.

    Parameters
    ----------
    image : np.ndarray, torch.Tensor
        Input image.
    pad_tuples : List[Tuple[int, int]]
        List of padding tuples for each axis.

    Returns
    -------
    partial
        A partial function to pad the image.
    """
    if all(p1 == 0 and p2 == 0 for p1, p2 in pad_tuples):
        return None

    kwargs = {"mode": "constant"}
    if isinstance(pad, str):
        kwargs["mode"] = pad
    elif isinstance(image, np.ndarray):
        kwargs["constant_values"] = pad
    else:  # Tensor
        kwargs["value"] = pad

    from functools import partial

    if isinstance(image, np.ndarray):
        return partial(
            np.pad,
            pad_width=[(0, 0)] * (image.ndim - len(pad_tuples)) + pad_tuples,
            **kwargs,
        )
    else:  # Tensor
        from itertools import chain

        return partial(
            torch.nn.functional.pad,
            pad=list(chain.from_iterable(reversed(pad_tuples))),
            **kwargs,
        )


def crop_transform(
        image: AT,
        offsets: Sequence[int] | None = None,
        target_shape: Sequence[int] | None = None,
        out: AT | None = None,
        pad: int = 0,
) -> AT:
    """
    Perform a crop transform of the last N dimensions on the image data.
    Cropping does not interpolate the image, but "just removes" border pixels/voxels.
    Negative offsets lead to padding.

    Parameters
    ----------
    image : np.ndarray, torch.Tensor
        Image of size [..., D_1, D_2, ..., D_N], where D_1, D_2, ..., D_N are the N
        image dimensions.
    offsets : Sequence[int], optional
        Offset of the cropped region for the last N dimensions (default: center crop
        with less crop/pad towards index 0).
    target_shape : Sequence[int], optional
        If defined, target_shape specifies the target shape of the "cropped region",
        else the crop will be centered cropping offset[dim] voxels on each side (then
        the shape is derived by subtracting 2x the dimension-specific offset).
        target_shape should have the same number of elements as offsets.
        May be implicitly defined by out.
    out : np.ndarray, torch.Tensor, optional
        Array to store the cropped image in (optional), can be a view on image for
        memory-efficiency.
    pad :  int, str, default=0/zero-pad
        Padding strategy to use when padding is required, if int, pad with that value.

    Returns
    -------
    out : np.ndarray, torch.Tensor
        The image (stack) cropped in the last N dimensions by offsets to the shape
        target_shape, or if target_shape is not given image.shape[i+2] - 2*offset[i].

    Raises
    ------
    ValueError
        If neither offsets nor target_shape nor out are defined.
    ValueError  
        If out is not target_shape.
    TypeError
        If the type of image is not an np.ndarray or a torch.Tensor.
    RuntimeError 
        If the dimensionality of image, out, offset or target_shape is invalid or
        inconsistent.

    See Also
    --------
    numpy.pad
        For additional information refer to numpy.pad function.

    Notes
    -----
    Either offsets, target_shape or out must be defined.
    """
    if target_shape is None and out is not None:
        target_shape = out.shape

    # check the type of offsets
    if offsets is None:
        if target_shape is None:
            raise ValueError("Either target_shape or offsets must be defined!")
        _target_shape = image.shape[: -len(target_shape)] + tuple(target_shape)
        offsets = tuple(int((i - t) / 2) for t, i in zip(_target_shape, image.shape, strict=False))
        len_off = len(offsets)
    else:
        len_off = len(offsets)
        if target_shape is None:
            _target_shape = image.shape[:-len_off] + tuple(
                i - 2 * o for i, o in zip(image.shape[-len_off:], offsets, strict=False)
            )
        elif len_off != len(target_shape):
            raise ValueError(
                "Incompatible offset and target_shape dimensionality (at least once)."
            )
        else:
            _target_shape = tuple(
                i if t == -1 else t
                for i, t in zip(image.shape[-len_off:], target_shape, strict=False)
            )
            _target_shape = image.shape[:-len_off] + _target_shape

    if len_off > image.ndim:
        raise RuntimeError("shape of offsets is larger than dim of image allows.")

    pad_tuples, indices = _crop_transform_make_indices(
        image.shape, offsets, _target_shape
    )
    if out is None:
        if pad_tuples is None:
            return image[indices]
        else:
            pad_fn = _crop_transform_pad_fn(image, pad_tuples, pad)
            return pad_fn(image[indices])
    else:
        if pad_tuples is None:
            out[:] = image[indices]
        else:
            pad_fn = _crop_transform_pad_fn(image, pad_tuples, pad)
            out[:] = pad_fn(image[indices])

    return out
