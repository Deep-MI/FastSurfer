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
import logging
from typing import Optional, Type, Tuple, Union, overload, Dict, Literal
import argparse
import sys

import numpy as np
import nibabel as nib

from FastSurferCNN.utils.arg_types import (vox_size as __vox_size, target_dtype as __target_dtype,
                                           conform_to_one_mm as __conform_to_one_mm, VoxSizeOption)

HELPTEXT = """
Script to conform an MRI brain image to UCHAR, RAS orientation, and 1mm or minimal isotropic voxels
USAGE:
conform.py  -i <input> -o <output> <options>
OR
conform.py  -i <input> --check_only <options>
Dependencies:
    Python 3.8
    Numpy
    http://www.numpy.org
    Nibabel to read and write FreeSurfer data
    http://nipy.org/nibabel/
Original Author: Martin Reuter
Date: Jul-09-2019
"""

h_input = 'path to input image'
h_output = 'path to ouput image'
h_order = 'order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)'


def options_parse():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(usage=HELPTEXT)
    parser.add_argument('--version', action='version',
                        version='$Id: conform.py,v 1.0 2019/07/19 10:52:08 mreuter Exp $')
    parser.add_argument('--input', '-i', dest='input', help=h_input)
    parser.add_argument('--output', '-o', dest='output', help=h_output)
    parser.add_argument('--order', dest='order', help=h_order, type=int, default=1)
    parser.add_argument('--check_only', dest='check_only', default=False, action='store_true',
                        help='If True, only checks if the input image is conformed, and does not return an output.')
    parser.add_argument('--seg_input', dest='seg_input', default=False, action='store_true',
                        help='Specifies whether the input is a seg image. If true, the check for conformance '
                             'disregards the uint8 dtype criteria. Use --dtype any for equivalent results. '
                             '--seg_input overwrites --dtype arguments.')
    parser.add_argument('--vox_size', dest='vox_size', default=1., type=__vox_size,
                        help="Specifies the target voxel size to conform to. Also allows 'min' for conforming to the "
                             "minimum voxel size, otherwise similar to mri_convert's --conform_size <size> "
                             "(default: 1, conform to 1mm).")
    parser.add_argument('--conform_min', dest='conform_min', default=False, action='store_true',
                        help='Specifies whether the input is or should be conformed to the '
                             'minimal voxel size (used for high-res processing) - overwrites --vox_size.')
    advanced = parser.add_argument_group("Advanced options")
    advanced.add_argument('--conform_to_1mm_threshold', type=__conform_to_one_mm,
                          help="Advanced option to change the threshold beyond which images are conformed to 1"
                               "(default: infinity, all images are conformed to their minimum voxel size).",
                          )
    parser.add_argument('--dtype', dest="dtype", default="uint8", type=__target_dtype,
                        help="Specifies the target data type of the target image or 'any' (default: 'uint8', "
                             "as in FreeSurfer)")
    parser.add_argument('--verbose', dest='verbose', default=False, action='store_true',
                        help='If verbose, more specific messages are printed')
    args = parser.parse_args()
    if args.input is None:
        sys.exit('ERROR: Please specify input image')
    if not args.check_only and args.output is None:
        sys.exit('ERROR: Please specify output image')
    if args.check_only and args.output is not None:
        sys.exit('ERROR: You passed in check_only. Please do not also specify output image')
    if args.seg_input and args.dtype not in ["uint8", "any"]:
        print("WARNING: --seg_input overwrites the --dtype arguments.")
    return args


def map_image(img: nib.analyze.SpatialImage, out_affine: np.ndarray, out_shape: np.ndarray,
              ras2ras: Optional[np.ndarray] = None,
              order: int = 1, dtype: Optional[Type] = None) -> np.ndarray:
    """
    Function to map image to new voxel space (RAS orientation)

    Args:
        img: the src 3D image with data and affine set
        out_affine: trg image affine
        out_shape: the trg shape information
        ras2ras: an additional mapping that should be applied (default=id to just reslice)
        order: order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)
        dtype: target dtype of the resulting image (relevant for reorientation, default=same as img)

    Returns:
        mapped image data array
    """
    from scipy.ndimage import affine_transform
    from numpy.linalg import inv

    if ras2ras is None:
        ras2ras = np.eye(4)

    # compute vox2vox from src to trg
    vox2vox = inv(out_affine) @ ras2ras @ img.affine

    # here we apply the inverse vox2vox (to pull back the src info to the target image)
    image_data = np.asanyarray(img.dataobj)
    # convert frames to single image
    if len(image_data.shape) > 3:
        if any(s != 1 for s in image_data.shape[3:]):
            raise ValueError(f'Multiple input frames {tuple(image_data.shape)} not supported!')
        image_data = np.squeeze(image_data, axis=tuple(range(3, len(image_data.shape))))

    if dtype is not None:
        image_data = image_data.astype(dtype)

    new_data = affine_transform(image_data, inv(vox2vox), output_shape=out_shape, order=order)
    return new_data


def getscale(data: np.ndarray, dst_min: float, dst_max: float,
             f_low: float = 0.0, f_high: float = 0.999) -> Tuple[float, float]:
    """
    Function to get offset and scale of image intensities to robustly rescale to range dst_min..dst_max.
    Equivalent to how mri_convert conforms images.

    Args:
        data: image data (intensity values)
        dst_min: future minimal intensity value
        dst_max: future maximal intensity value
        f_low: robust cropping at low end (0.0 no cropping, default)
        f_high: robust cropping at higher end (0.999 crop one thousandth of high intensity voxels, default)

    Returns:
        a tuple of the (adjusted) offset and the scale factor
    """
    # get min and max from source
    src_min = np.min(data)
    src_max = np.max(data)

    if src_min < 0.0:
        print("WARNING: Input image has value(s) below 0.0 !")

    print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    # compute non-zeros and total vox num
    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    # compute histogram
    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    # compute cummulative sum
    cs = np.concatenate(([0], np.cumsum(hist)))

    # get lower limit
    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    # get upper limit
    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print('ERROR: rescale upper bound not found')

    src_max = idx * bin_size + src_min

    # scale
    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))

    return src_min, scale


def scalecrop(data: np.ndarray, dst_min: float, dst_max: float, src_min: float, scale: float) -> np.ndarray:
    """
    Function to crop the intensity ranges to specific min and max values

    Args:
        data: Image data (intensity values)
        dst_min: future minimal intensity value
        dst_max: future maximal intensity value
        src_min: minimal value to consider from source (crops below)
        scale: scale value by which source will be shifted

    Returns:
        scaled image data
    """
    data_new = dst_min + scale * (data - src_min)

    # clip
    data_new = np.clip(data_new, dst_min, dst_max)
    print("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))

    return data_new


def rescale(data: np.ndarray, dst_min: float, dst_max: float,
             f_low: float = 0.0, f_high: float = 0.999) -> np.ndarray:
    """
    Function to rescale image intensity values (0-255).

    Args:
        data: image data (intensity values)
        dst_min: future minimal intensity value
        dst_max: future maximal intensity value
        f_low: robust cropping at low end (0.0 no cropping, default)
        f_high: robust cropping at higher end (0.999 crop one thousandth of high intensity voxels, default)

    Returns:
        scaled image data
    """
    src_min, scale = getscale(data, dst_min, dst_max, f_low, f_high)
    data_new = scalecrop(data, dst_min, dst_max, src_min, scale)
    return data_new


def find_min_size(img: nib.analyze.SpatialImage, max_size: float = 1) -> float:
    """
    Function to find minimal voxel size <= 1mm.

    Args:
        img: loaded source image
        max_size: maximal voxel size in mm (default: 1.0)

    Returns:
        The rounded minimal voxel size
    """
    # find minimal voxel side length
    sizes = np.array(img.header.get_zooms()[:3])
    min_vox_size = np.round(np.min(sizes)*10000) / 10000
    # set to max_size mm if larger than that (usually 1mm)
    return min(min_vox_size, max_size)


def find_img_size_by_fov(img: nib.analyze.SpatialImage, vox_size: float, min_dim: int = 256) -> int:
    """
    Function to find the cube dimension (>= 256) to cover the field of view of img. If vox_size is one, the img_size
    MUST always be min_dim (the FreeSurfer standard).

    Args:
        img: loaded source image
        vox_size: the target voxel size in mm
        min_dim: minimal image dimension in voxels (default 256)

    Returns:
        The number of voxels needed to cover field of view.
    """
    if vox_size == 1.:
        return min_dim
    # else (other voxel sizes may use different sizes)

    # compute field of view dimensions in mm
    sizes = np.array(img.header.get_zooms()[:3])
    max_fov = np.max(sizes * np.array(img.shape[:3]))
    # compute number of voxels needed to cover field of view
    conform_dim = int(np.ceil(int(max_fov/vox_size * 10000) / 10000))
    # use cube with min_dim (default 256) in each direction as minimum
    return max(min_dim, conform_dim)


def conform(img: nib.analyze.SpatialImage,
            order: int = 1, conform_vox_size: VoxSizeOption = 1., dtype: Optional[Type] = None,
            conform_to_1mm_threshold: Optional[float] = None) -> nib.MGHImage:
    f"""
    Python version of mri_convert -c, which by default turns image intensity values
    into UCHAR, reslices images to standard position, fills up slices to standard
    256x256x256 format and enforces 1mm or minimum isotropic voxel sizes.

    Notes:
        Unlike mri_convert -c, we first interpolate (float image), and then rescale
        to uchar. mri_convert is doing it the other way around. However, we compute
        the scale factor from the input to increase similarity.

    Args:
        img: loaded source image
        order: interpolation order (0=nearest,1=linear(default),2=quadratic,3=cubic)
        conform_vox_size: conform image the image to voxel size 1. (default), a
            specific smaller voxel size (0-1, for high-res), or automatically
            determine the 'minimum voxel size' from the image (value 'min').
            This assumes the smallest of the three voxel sizes.
        dtype: the dtype to enforce in the image (default: UCHAR, as mri_convert -c)
        conform_to_1mm_threshold: the threshold above which the image is conformed to 1mm 
            (default: ignore). 

    Returns:
         conformed image
    """
    from nibabel.freesurfer.mghformat import MGHHeader

    conformed_vox_size, conformed_img_size = get_conformed_vox_img_size(img, conform_vox_size,
                                                                        conform_to_1mm_threshold=conform_to_1mm_threshold)

    h1 = MGHHeader.from_header(img.header)  # may copy some parameters if input was MGH format

    h1.set_data_shape([conformed_img_size, conformed_img_size, conformed_img_size, 1])
    h1.set_zooms([conformed_vox_size, conformed_vox_size, conformed_vox_size])  # --> h1['delta']
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = conformed_img_size * conformed_vox_size
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]

    # Here, we are explicitly using MGHHeader.get_affine() to construct the affine as
    # MdcD = np.asarray(h1['Mdc']).T * h1['delta']
    # vol_center = MdcD.dot(hdr['dims'][:3]) / 2
    # affine = from_matvec(MdcD, h1['Pxyz_c'] - vol_center)
    affine = h1.get_affine()

    # from_header does not compute Pxyz_c (and probably others) when importing from nii
    # Pxyz is the center of the image in world coords

    # target scalar type and dtype
    sctype = np.uint8 if dtype is None else np.obj2sctype(dtype, default=np.uint8)
    target_dtype = np.dtype(sctype)

    src_min, scale = 0, 1.
    # get scale for conversion on original input before mapping to be more similar to mri_convert
    if img.get_data_dtype() != np.dtype(np.uint8) or img.get_data_dtype() != target_dtype:
        src_min, scale = getscale(np.asanyarray(img.dataobj), 0, 255)

    kwargs = {"dtype": "float"} if sctype != np.uint else {}
    mapped_data = map_image(img, affine, h1.get_data_shape(), order=order, **kwargs)

    if img.get_data_dtype() != np.dtype(np.uint8) or (img.get_data_dtype() != target_dtype and scale != 1.):
        scaled_data = scalecrop(mapped_data, 0, 255, src_min, scale)
        # map zero in input to zero in ouput (usually background)
        scaled_data[mapped_data == 0] = 0
        mapped_data = scaled_data

    mapped_data = sctype(np.rint(mapped_data) if target_dtype == np.dtype(np.uint8) else mapped_data)
    new_img = nib.MGHImage(mapped_data, affine, h1)

    # make sure we store uchar
    try:
        new_img.set_data_dtype(target_dtype)
    except nib.freesurfer.mghformat.MGHError as e:
        if "not recognized" in e.args[0]:
            codes = set(k.name for k in nib.freesurfer.mghformat.data_type_codes.code.keys() if isinstance(k, np.dtype))
            print(f'The data type "{options.dtype}" is not recognized for MGH images, switching '
                  f'to "{new_img.get_data_dtype()}" (supported: {tuple(codes)}).')

    return new_img


def is_conform(img: nib.analyze.SpatialImage,
               conform_vox_size: VoxSizeOption = 1., eps: float = 1e-06, check_dtype: bool = True,
               dtype: Optional[Type] = None, verbose: bool = True,
               conform_to_1mm_threshold: Optional[float] = None) -> bool:
    f"""
    Function to check if an image is already conformed or not (Dimensions: 256x256x256,
    Voxel size: 1x1x1, LIA orientation, and data type UCHAR).

    Args:
        img: Loaded source image
        conform_vox_size: which voxel size to conform to. Can either be a float between 0.0 and
            1.0 or 'min' check, whether the image is conformed to the minimal voxels size, i.e.
            conforming to smaller, but isotropic voxel sizes for high-res (default: 1.0).
        eps: allowed deviation from zero for LIA orientation check (default: 1e-06).
            Small inaccuracies can occur through the inversion operation. Already conformed
            images are thus sometimes not correctly recognized. The epsilon accounts for
            these small shifts.
        check_dtype: specifies whether the UCHAR dtype condition is checked for;
            this is not done when the input is a segmentation (default: True).
        dtype: specifies the intended target dtype (default: uint8 = UCHAR)
        verbose: if True, details of which conformance conditions are violated (if any)
            are displayed (default: True).
        conform_to_1mm_threshold: the threshold above which the image is conformed to 1mm 
            (default: ignore). 

    Returns:
        whether the image is already conformed.
    """

    conformed_vox_size, conformed_img_size = get_conformed_vox_img_size(img, conform_vox_size,
                                                                        conform_to_1mm_threshold=conform_to_1mm_threshold)

    ishape = img.shape
    # check 3d
    if len(ishape) > 3 and ishape[3] != 1:
        raise ValueError(f'ERROR: Multiple input frames ({img.shape[3]}) not supported!')

    criteria = {}
    # check dimensions
    criteria['Dimensions {0}x{0}x{0}'.format(conformed_img_size)] = all(s == conformed_img_size for s in ishape[:3])

    # check voxel size
    izoom = np.array(img.header.get_zooms())
    is_correct_vox_size = np.max(np.abs(izoom - conformed_vox_size) < eps)
    criteria['Voxel Size {0}x{0}x{0}'.format(conformed_vox_size)] = is_correct_vox_size

    # check orientation LIA
    LIA_affine = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    iaffine = img.affine[0:3, 0:3] - LIA_affine * (conformed_vox_size if is_correct_vox_size else izoom)
    criteria['Orientation LIA'] = (np.max(np.abs(iaffine)) <= eps)

    # check dtype uchar
    if check_dtype:
        if dtype is None or (isinstance(dtype, str) and dtype.lower() == "uchar"):
            dtype = 'uint8'
        else:  # assume obj
            dtype = np.dtype(np.obj2sctype(dtype)).name
        criteria[f'Dtype {dtype}'] = (img.get_data_dtype() == dtype)

    _is_conform = all(criteria.values())
    #result = (_is_conform, criteria) if return_criteria else _is_conform

    if verbose:
        if not _is_conform:
            print('The input image is not conformed.')

        conform_str = "conformed" if conform_vox_size == 1.0 else f"{conform_vox_size}-conformed"
        print(f'A {conform_str} image must satisfy the following criteria:')
        for condition, value in criteria.items():
            print(' - {:<30} {}'.format(condition + ':', value))
    return _is_conform


def get_conformed_vox_img_size(img: nib.analyze.SpatialImage, conform_vox_size: VoxSizeOption,
                               conform_to_1mm_threshold: Optional[float] = None) -> Tuple[float, int]:
    """Extract the voxel size and the image size."""
    # this is similar to mri_convert --conform_min
    if isinstance(conform_vox_size, str) and conform_vox_size.lower() in ["min", "auto"]:
        conformed_vox_size = find_min_size(img)
        if conform_to_1mm_threshold is not None and conformed_vox_size > conform_to_1mm_threshold:
            conformed_vox_size = 1.
    # this is similar to mri_convert --conform_size <float>
    elif isinstance(conform_vox_size, float) and 0. < conform_vox_size <= 1.0:
        conformed_vox_size = conform_vox_size
    else:
        raise ValueError("Invalid value for conform_vox_size passed.")
    conformed_img_size = find_img_size_by_fov(img, conformed_vox_size)
    return conformed_vox_size, conformed_img_size


def check_affine_in_nifti(img: Union[nib.Nifti1Image, nib.Nifti2Image], logger: Optional[logging.Logger] = None) -> bool:
    """
    Function to check the affine in nifti Image. Sets affine with qform, if it exists
    and differs from sform. If qform does not exist, voxel sizes between header
    information and information in affine are compared. In case these do not match,
    the function returns False (otherwise True).

    Args:
        img: loaded nifti-image
        logger: Logger object or None (default) to log or print an info message to
            stdout (for None)

    Returns:
        True, if: affine was reset to qform voxel sizes in affine are equivalent to
            voxel sizes in header
        False, if: voxel sizes in affine and header differ
    """
    check = True
    message = ""

    if img.header['qform_code'] != 0 and np.max(np.abs(img.get_sform() - img.get_qform())) > 0.001:
        message = "#############################################################" \
                  "\nWARNING: qform and sform transform are not identical!\n sform-transform:\n{}\n " \
                  "qform-transform:\n{}\n" \
                  "You might want to check your Nifti-header for inconsistencies!" \
                  "\n!!! Affine from qform transform will now be used !!!\n" \
                  "#############################################################".format(img.header.get_sform(),
                                                                                         img.header.get_qform())
        # Set sform with qform affine and update best affine in header
        img.set_sform(img.get_qform())
        img.update_header()

    else:
        # Check if affine correctly includes voxel information and print Warning/Exit otherwise
        vox_size_head = img.header.get_zooms()
        aff = img.affine
        xsize = np.sqrt(aff[0][0] * aff[0][0] + aff[1][0] * aff[1][0] + aff[2][0] * aff[2][0])
        ysize = np.sqrt(aff[0][1] * aff[0][1] + aff[1][1] * aff[1][1] + aff[2][1] * aff[2][1])
        zsize = np.sqrt(aff[0][2] * aff[0][2] + aff[1][2] * aff[1][2] + aff[2][2] * aff[2][2])

        if (abs(xsize - vox_size_head[0]) > .001) or (abs(ysize - vox_size_head[1]) > .001) or (
                abs(zsize - vox_size_head[2]) > 0.001):
            message = f"#############################################################\n" \
                      f"ERROR: Invalid Nifti-header! Affine matrix is inconsistent with Voxel sizes. " \
                      f"\nVoxel size (from header) vs. Voxel size in affine: " \
                      f"{tuple(vox_size_head[:3])}, {(xsize, ysize, zsize)}\n" \
                      f"Input Affine----------------\n{aff}\n" \
                      f"#############################################################"
            check = False

    if logger is not None:
        logger.info(message)

    else:
        print(message)

    return check


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()

    print(f"Reading input: {options.input} ...")
    image = nib.load(options.input)

    if len(image.shape) > 3 and image.shape[3] != 1:
        sys.exit(f'ERROR: Multiple input frames ({image.shape[3]}) not supported!')

    target_dtype = "uint8" if options.seg_input else options.dtype
    opt_kwargs = {}
    check_dtype = target_dtype != "any"
    if check_dtype:
        opt_kwargs["dtype"] = target_dtype

    if hasattr(options, "conform_to_1mm_threshold"):
        opt_kwargs["conform_to_1mm_threshold"] = options.conform_to_1mm_threshold

    _vox_size = 'min' if options.conform_min else options.vox_size
    try:
        image_is_conformed = is_conform(image,
                                        conform_vox_size=_vox_size, check_dtype=check_dtype, verbose=options.verbose,
                                        **opt_kwargs)
    except ValueError as e:
        sys.exit(e.args[0])

    if image_is_conformed:
        print(f"Input {options.input} is already conformed! Exiting.\n")
        sys.exit(0)
    else:
        # Note: if check_only, a non-conforming image leads to an error code, this result is needed in recon_surf.sh
        if options.check_only:
            print("check_only flag provided. Exiting without conforming input image.\n")
            sys.exit(1)

    # If image is nifti image
    if options.input[-7:] == ".nii.gz" or options.input[-4:] == ".nii":

        if not check_affine_in_nifti(image):
            sys.exit("ERROR: inconsistency in nifti-header. Exiting now.\n")

    try:
        new_image = conform(image, order=options.order, conform_vox_size=_vox_size, dtype=options.dtype)
    except ValueError as e:
        sys.exit(e.args[0])
    print(f"Writing conformed image: {options.output}")

    nib.save(new_image, options.output)

    sys.exit(0)
