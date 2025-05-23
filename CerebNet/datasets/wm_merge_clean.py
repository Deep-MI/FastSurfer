from collections.abc import Iterable
from numbers import Number

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
from os.path import join
from typing import TypeVar

import nibabel as nib
import numpy as np
from numpy import typing as npt
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation

NT = TypeVar("NT", bound=Number)


def locating_unknowns(gm_binary, wm_mask):
    """
    Find labels with missing labels, i.e. find holes.
    """
    selem = ndimage.generate_binary_structure(3, 3)
    wm_binary = np.array(wm_mask, dtype=np.bool)
    # gm_binary = (segmap != 0) ^ wm_binary
    wm_boundary = binary_dilation(wm_binary, selem) ^ wm_binary
    # wm_boundary = (binary_dilation(wm_boundary, selem))
    gm_boundary = binary_dilation(gm_binary, selem) ^ gm_binary
    boundary_holes = np.logical_and(wm_boundary, gm_boundary)
    return boundary_holes


def drop_disconnected_component(
    img_data: npt.NDArray[NT], classes: Iterable[NT]) -> npt.NDArray[NT]:
    """
    Dropping the smaller disconnected component of each label.
    """
    kept_components_mask = np.zeros_like(img_data)
    holes_mask = np.zeros_like(img_data)
    for c in classes:
        current_class = img_data == c
        label_image = label(current_class, connectivity=3)
        regions = regionprops(label_image)

        sorted_reg = sorted(regions, key=lambda region: region.area, reverse=True)
        largest_component = sorted_reg[0]
        bin_mask = label_image == largest_component.label
        kept_components_mask[bin_mask] = 1
        mask = np.logical_and(label_image != largest_component.label, label_image != 0)
        holes_mask[mask] = 1

    return kept_components_mask, holes_mask


def filling_unknown_labels(segmap, unknown_mask, candidate_lbls):
    """
    For each unknown voxel in unknown_mask, find and fill it with a candidate.
    """
    h, w, d = segmap.shape
    blur_vals = np.ndarray((h, w, d, 0), dtype=np.float)
    for lbl in candidate_lbls:
        seg_blur = ndimage.distance_transform_edt(segmap != lbl).astype(np.int)
        blur_vals = np.append(blur_vals, np.expand_dims(seg_blur, axis=3), axis=3)

    img_unknown_boundary = np.argmin(blur_vals, axis=3)
    array = candidate_lbls[img_unknown_boundary.ravel()]
    img_unknown_boundary = np.reshape(array, (h, w, d))
    filled_seg = segmap.copy()
    filled_seg[unknown_mask == 1] = img_unknown_boundary[unknown_mask == 1]
    return filled_seg


def cereb_subseg_lateral_mask(cereb_subseg):
    """
    Create mask for left and right cerebellar gray matter.
    """
    left_gm_idxs = np.array([1, 3, 5, 8, 11, 14, 17, 20, 23, 26])
    right_gm_idxs = np.array([2, 4, 7, 10, 13, 16, 19, 22, 25, 28])

    left_mask = np.ones_like(cereb_subseg, dtype=np.bool)
    for idx in left_gm_idxs:
        left_mask = np.logical_and(left_mask, cereb_subseg == idx)

    right_mask = np.ones_like(cereb_subseg, dtype=np.bool)
    for idx in right_gm_idxs:
        right_mask = np.logical_and(right_mask, cereb_subseg == idx)

    return left_mask, right_mask


def sphere(radius):
    """
    Create a spherical binary mask.
    """
    shape = (2 * radius + 1,) * 3
    struct = np.zeros(shape)
    x, y, z = np.indices(shape)
    mask = (x - radius) ** 2 + (y - radius) ** 2 + (z - radius) ** 2 <= radius**2
    struct[mask] = 1
    return struct.astype(np.bool)


def add_cereb_wm(cereb_subseg, aseg, manual_cereb):
    """
    Adding cerebellar wm from FreeSurfer and filling the gaps
    if cereb_subseg is dzne_manual we also update FreeSurfer cereb wm accordingly

    :param cereb_subseg:
    :param aseg:
    :param manual_cereb:
    :return:
    """
    # to capture small holes
    # struc = ndimage.generate_binary_structure(3, 1)

    l_wm_fs = aseg == 7
    r_wm_fs = aseg == 46
    gm_fs = np.logical_or(aseg == 47, aseg == 8)

    gm_mask = gm_fs != 0
    if manual_cereb:
        gm_mask = cereb_subseg != 0
        print("1.Intersection of manual labels and WM from fs")
    else:
        print("1.Intersection of SUIT and WM from FS")

    intersect_seg = np.zeros_like(cereb_subseg)
    intersect_seg[gm_mask] = cereb_subseg[gm_mask]
    intersect_seg[intersect_seg == 29] = 36
    intersect_seg[intersect_seg == 30] = 36
    intersect_seg[intersect_seg == 31] = 36
    intersect_seg[intersect_seg == 32] = 36
    intersect_seg[intersect_seg == 33] = 36
    intersect_seg[intersect_seg == 34] = 36
    intersect_seg[l_wm_fs != 0] = 37
    intersect_seg[r_wm_fs != 0] = 38

    # print("2.Filling WM holes")
    wm_mask = np.logical_or(intersect_seg == 37, intersect_seg == 38)

    # filled_bin_wm_mask = ndimage.binary_fill_holes(wm_mask, structure=struc)
    # wm_holes_mask = np.logical_xor(filled_bin_wm_mask, wm_mask)
    # wm_holes_mask = np.logical_or(wm_holes_mask, intersect_seg == 36)
    # intersect_seg = filling_unknown_labels(intersect_seg, wm_holes_mask, candidate_lbls=np.array([34, 35]))

    print("2.Locating unknown labels touching the wm")
    boundary_holes = locating_unknowns(gm_mask, wm_mask)
    holes_mask = np.logical_or(boundary_holes, intersect_seg == 9)
    holes_mask = np.logical_or(holes_mask, intersect_seg == 36)

    if manual_cereb:
        # mapping unknown regions to WM
        candidate_lbls = np.array([37, 38])
    else:
        candidate_lbls = np.array(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                37,
                38,
            ]
        )

    print("3.Filling holes touching WM")
    filled_seg = filling_unknown_labels(intersect_seg, holes_mask, candidate_lbls)

    classes = np.unique(filled_seg)
    kept_components_mask, discon_holes_mask = drop_disconnected_component(
        img_data=filled_seg, classes=classes
    )
    dropped_comp_img = np.zeros_like(filled_seg)
    dropped_comp_img[kept_components_mask == 1] = filled_seg[kept_components_mask == 1]

    candidate_lbls = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            37,
            38,
        ]
    )

    print("4.Filling any remaining holes")
    #filled_bin_mask = ndimage.binary_fill_holes(dropped_comp_img != 0, structure=struc)
    remaining_holes_mask = (
        dropped_comp_img != 0
    )  # np.logical_xor(filled_bin_mask, dropped_comp_img != 0)
    out_img = filling_unknown_labels(
        dropped_comp_img, remaining_holes_mask, candidate_lbls
    )

    if manual_cereb:
        # put back cerebellar gray matter lobules to overwrite FreeSurfer WM
        out_img[gm_mask] = cereb_subseg[gm_mask]
        # updating aseg cerebellar wm and gm
        aseg[gm_fs] = 0
        aseg[out_img == 37] = 7
        aseg[out_img == 38] = 46
        new_wm_mask = np.logical_or(out_img == 37, out_img == 38)
        new_gm_mask = np.logical_and(out_img != 0, ~new_wm_mask)
        aseg[new_gm_mask] = 8
    else:
        out_img[l_wm_fs] = 37
        out_img[r_wm_fs] = 38

    return out_img, aseg


def correct_cereb_brainstem(cereb_subseg, brainstem, manual_cereb):
    """
    Correct brainstem or cereb_subseg according to the 
    other (select which to correct by manual_cereb).
    """
    if manual_cereb:
        print("Correcting brainstem according to cerebellum dzne_manual subseg.")
        # mapping the overlapping part to dzne_manual labels
        brainstem[cereb_subseg != 0] = 0
    else:
        print("Correcting cereb subseg according to brainstem.")
        cereb_subseg[brainstem != 0] = 0

    return cereb_subseg, brainstem


def save_mgh_image(img_data, save_path, header, affine):
    """
    Save data as mgh image.
    """
    mgh_out = nib.MGHImage(img_data, header=header, affine=affine)
    print(f"Saving {save_path}")
    nib.save(mgh_out, save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject_path",
        help="path of subject folder with 'mri' folder",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--aseg_filename",
        help="name of modified aseg file, "
        "default: aparc.DKTatlas+aseg.bs_corr.filled.mgz",
        default="aparc.DKTatlas+aseg.bs_corr.filled.mgz",
        type=str,
    )
    parser.add_argument(
        "--cereb_filename",
        help="filename of cerebellum sub-segmentation"
        "default: suit_fs/suit_cereb_labels.nii",
        default="suit_fs/suit_cereb_labels.nii",
        type=str,
    )
    parser.add_argument(
        "--out_cereb_name",
        help="name of modified aseg file, "
        "default: cleaned_suit_fs_bs_corr.mgz"
        " or cleaned_manual_fs_bs_corr.mgz",
        default="cleaned_suit_fs_bs_corr.mgz",
        type=str,
    )
    parser.add_argument(
        "--manual_cereb",
        help="dzne_manual labels for cerebellum sub-regions",
        default=False,
        type=bool,
    )

    args = parser.parse_args()
    if args.manual_cereb:
        print("Processing dzne_manual labels.")

    aseg_file = nib.load(join(args.subject_path, args.aseg_filename))
    cereb_subseg_file = nib.load(join(args.subject_path, args.cereb_filename))

    aseg = np.array(aseg_file.get_fdata(), dtype=np.int16)

    cereb_subseg = np.array(cereb_subseg_file.get_fdata(), dtype=np.int16)

    print("**Adding cerebellum white matter to cereb subseg and fill gaps.")
    cereb_subseg, aseg = add_cereb_wm(cereb_subseg, aseg, args.manual_cereb)

    # # relabeling left/right CM (34, 35) to 37, 38
    # cereb_subseg[cereb_subseg == 34] = 37
    # cereb_subseg[cereb_subseg == 35] = 38

    # merge 12.Vermis_CrusII, 15.Vermis_VIIb into 12. Vermis_VII
    cereb_subseg[cereb_subseg == 15] = 12

    # merge 18.Vermis_VIIIa, 21. Vermis_VIIIb into 18. Vermis_VIII
    cereb_subseg[cereb_subseg == 21] = 18

    cereb_path = join(args.subject_path, args.out_cereb_name)
    save_mgh_image(
        cereb_subseg,
        cereb_path,
        header=cereb_subseg_file.header,
        affine=cereb_subseg_file.affine,
    )
