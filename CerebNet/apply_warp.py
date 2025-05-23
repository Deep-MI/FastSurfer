#!/bin/python

# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
import argparse
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
from numpy import typing as npt

from CerebNet.datasets import utils


def save_nii_image(
        img_data: npt.ArrayLike,
        save_path: Path | str,
        header: nib.Nifti1Header | nib.Nifti2Header,
        affine: npt.NDArray[float],
):
    """
    Save an image data array as a NIfTI file.

    Parameters
    ----------
    img_data : ndarray
        The image data to be saved.
    save_path : Path, str
        The path (including file name) where the image will be saved.
    header : nibabel.Nifti1Header, nibabel.Nifti2Header
        The header information for the NIfTI file.
    affine : ndarray
        The affine matrix for the NIfTI file.
    """
    if not isinstance(header, nib.Nifti1Header):
        header = nib.Nifti1Header.from_header(header)
    img_out = nib.Nifti1Image(img_data, header=header, affine=affine)
    print(f"Saving {save_path}")
    nib.save(img_out, save_path)


def main(
        img_path: Path | str,
        lbl_path: Path | str,
        warp_path: Path | str,
        result_path: Path | str,
        patch_size,
):

    """
    Load, warp, crop, and save both an image and its corresponding label based on a given warp field.

    Parameters
    ----------
    img_path : Path, str
        Path to the T1-weighted MRI image to be warped.
    lbl_path : Path, str
        Path to the label image corresponding to the T1 image, to be warped similarly.
    warp_path : Path, str
        Path to the warp field file used to warp the images.
    result_path : Path, str
        Directory path where the warped and cropped images will be saved.
    patch_size : tuple of int
        The dimensions (height, width, depth) cropped images after warping.
    """

    img, img_file = utils.load_reorient_rescale_image(img_path)

    lbl_file = nib.load(lbl_path)
    # if not isinstance(lbl_file, nib.analyze.SpatialImage):
    if not isinstance(lbl_file, nib.Nifti1Image | nib.Nifti2Image):
        raise ValueError(f"{lbl_file} is not a valid file format!")
    lbl_header = cast(nib.Nifti1Header | nib.Nifti2Header, lbl_file.header)
    label = np.asarray(lbl_file.dataobj, dtype=np.int16)

    warp_file = nib.load(warp_path)
    if not isinstance(warp_file, nib.analyze.SpatialImage):
        raise ValueError(f"{warp_file} is not a valid file format!")
    warp_field = np.asarray(warp_file.dataobj, dtype=float)
    img = utils.map_size(img, base_shape=warp_field.shape[:3])
    label = utils.map_size(label, base_shape=warp_field.shape[:3])
    warped_img = utils.apply_warp_field(warp_field, img, interpol_order=3)
    warped_lbl = utils.apply_warp_field(warp_field, label, interpol_order=0)
    utils.map_subseg2label(warped_lbl, label_type='cereb_subseg')
    roi = utils.bounding_volume(label, patch_size)

    img = utils.map_size(warped_img[roi], patch_size)
    label = utils.map_size(warped_lbl[roi], patch_size)

    img_file.header['dim'][1:4] = patch_size
    img_file.set_data_dtype(img.dtype)
    lbl_header['dim'][1:4] = patch_size
    save_nii_image(img,
                   Path(result_path) / "T1_warped_cropped.nii.gz",
                   header=img_file.header,
                   affine=img_file.affine)
    save_nii_image(label,
                   Path(result_path) / "label_warped_cropped.nii.gz",
                   header=lbl_header,
                   affine=lbl_file.affine)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path",
                        help="path to T1 image",
                        type=Path)
    parser.add_argument("--lbl_path",
                        help="path to label image",
                        type=Path)
    parser.add_argument("--result_path",
                        help="folder to store the results",
                        type=Path)

    parser.add_argument("--warp_filename",
                        help="Warp field file",
                        default='1Warp.nii.gz',
                        type=str)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    warp_path = Path(args.result_path) / args.warp_filename
    main(
        args.img_path,
        args.lbl_path,
        warp_path=warp_path,
        result_path=args.result_path,
        patch_size=(128, 128, 128),
    )
