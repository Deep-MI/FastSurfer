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
from os.path import join, isfile
from functools import partial

import numpy as np

from CerebNet.data_loader.data_utils import get_plane_transform
from CerebNet.datasets import utils


class SubjectLoader:
    """
    Subject loader class.
    """
    def __init__(self, cfg, aux_subjects_files=None):
        self.cfg = cfg
        self.patch_size = cfg.PATCH_SIZE
        # dictionary of subject ids to a list of warp dirs
        self.aux_subjects_path = (
            {} if aux_subjects_files is None else aux_subjects_files
        )
        self.label_mapping = partial(utils.map_subseg2label, label_type="cereb_subseg")

    def _process_segm_volumes(
        self,
        seg_map,
        label_map_func,
        plane_transform=None,):
        """
        Process segmentation volumes.

        Parameters
        ----------
        seg_map : np.ndarray
            The segmentation map to be processed.
        label_map_func : function
            A function to map labels in the segmentation map.
        plane_transform : function, optional
            A function to transform the segmentation map in plane. Defaults to None.

        Returns
        -------
        np.ndarray
            The processed segmentation map.
        """
        mapped_seg = label_map_func(seg_map)
        if plane_transform is not None:
            mapped_seg = plane_transform(mapped_seg)
        return mapped_seg

    def _load_volumes(self, subject_path, store_talairach=False):
        """
        [MISSING].
        """
        orig_path = join(subject_path, self.cfg.IMAGE_NAME)
        subseg_path = join(subject_path, self.cfg.CEREB_SUBSEG_NAME)

        img_meta_data = {}
        orig, _ = utils.load_reorient_rescale_image(orig_path)
        print("Orig image {}".format(orig_path))

        print("Loading from {}".format(subseg_path))
        subseg_file = utils.load_reorient(subseg_path)
        cereb_subseg = np.asarray(subseg_file.get_fdata(), dtype=np.int16)
        img_meta_data["affine"] = subseg_file.affine
        img_meta_data["header"] = subseg_file.header

        if store_talairach:
            tala_path = join(subject_path, "transforms/talairach.xfm.lta")
            if isfile(tala_path):
                talairach_coordinates = utils.load_talairach_coordinates(
                    tala_path, subseg_file.shape, subseg_file.affine
                )
                img_meta_data["talairach_coordinates"] = talairach_coordinates

        return orig, cereb_subseg, img_meta_data

    def load_test_subject(self, current_subject):
        """
        Loading subject for eval
        :param current_subject:
        :return:
        """
        processed_data = {"image": {}, "label": {}}
        subject_path = join(self.cfg.DATA_DIR, current_subject)
        orig, cereb_subseg, meta_data = self._load_volumes(subject_path)
        roi = utils.bounding_volume(cereb_subseg, self.patch_size)
        orig, unpad_border = utils.map_size(
            orig[roi], self.patch_size, return_border=True
        )
        subseg_cereb = utils.map_size(cereb_subseg[roi], self.patch_size)
        meta_data["unpad_border"] = unpad_border
        meta_data["bounding_vol"] = roi
        # subseg_cereb = self._process_segm_volumes(cereb_subseg, self.label_mapping)
        processed_data["label"] = subseg_cereb
        orig = orig[None, ...]
        for plane in ["axial", "coronal", "sagittal"]:
            plane_transform = get_plane_transform(plane, self.cfg.PRIMARY_SLICE_DIR)
            orig_in_plane = plane_transform(orig)
            orig_in_plane = np.transpose(orig_in_plane, (0, 3, 1, 2))
            thick_slices = utils.get_thick_slices(orig_in_plane, self.cfg.THICKNESS)
            processed_data["image"][plane] = np.squeeze(thick_slices)

        processed_data["meta"] = meta_data

        return processed_data

    def _get_roi_extracted_data(self, img, label, talairach):
        """
        Finding the bounding volume and returning extracted img and label
        according to roi
        Args:
            img:
            label:

        Returns:
            img and label resized according to roi and patch size
        """
        roi = utils.bounding_volume(label, self.patch_size)
        img = utils.map_size(img[roi], self.patch_size)
        label = utils.map_size(label[roi], self.patch_size)
        if talairach is not None:
            pad_width = []
            talairach = talairach[roi]
            talairach = utils.normalize_array(talairach)
            if talairach.shape[:-1] != self.patch_size:
                for i, size in enumerate(self.patch_size):
                    p = size - talairach.shape[i]
                    low = p // 2
                    high = p - low
                    pad_width.append((low, high))

                pad_width.append((0, 0))
                talairach = np.pad(talairach, pad_width, mode="edge")

        return img, label, talairach

    def _load_auxiliary_data(self, aux_subjects_path):
        """
        Loading auxiliary data create by registration of original images
        Args:
            subjects_path: list of full path to auxiliary data

        Returns:
            dictionary with list of warped images and labels
        """
        aux_data = {"auxiliary_img": [], "auxiliary_lbl": []}
        for t1_path, lbl_path in aux_subjects_path:
            aux_data["auxiliary_img"].append(
                np.array(utils.load_reorient(t1_path).get_fdata())
            )
            lbl_data = np.array(
                utils.load_reorient(lbl_path).get_fdata(), dtype=np.int16
            )
            lbl_data = self.label_mapping(lbl_data)
            aux_data["auxiliary_lbl"].append(lbl_data)

        if len(aux_data["auxiliary_img"]) > 0:
            aux_data["auxiliary_img"] = np.stack(aux_data["auxiliary_img"], axis=0)
            aux_data["auxiliary_lbl"] = np.stack(aux_data["auxiliary_lbl"], axis=0)
        return aux_data

    def load_subject(self, current_subject, store_talairach=False, load_aux_data=False):
        """
        Loads and processes the subject and returns data in a dictionary.

        Parameters
        ----------
        current_subject : [MISSING]
            Subject ID.
        store_talairach : bool, optional
            Whether to store Talairach coordinates. Defaults to False.
        load_aux_data : bool, optional
            Whether to load auxiliary data. Defaults to False.

        Returns
        -------
        dict
            Dictionary of processed data.
        """
        in_data = {}
        subject_path = join(self.cfg.DATA_DIR, current_subject)
        orig, cereb_subseg, meta_data = self._load_volumes(
            subject_path, store_talairach
        )
        aux_data = {}
        if load_aux_data:
            aux_data = self._load_auxiliary_data(
                self.aux_subjects_path[current_subject]
            )

        orig, cereb_subseg, talairach = self._get_roi_extracted_data(
            orig, cereb_subseg, meta_data.get("talairach_coordinates")
        )
        in_data["img"] = orig[None, ...]

        if talairach is not None:
            in_data["talairach"] = talairach[None, ...]
        in_data.update(aux_data)
        subseg_cereb = self._process_segm_volumes(cereb_subseg, self.label_mapping)
        in_data["label"] = subseg_cereb[None, ...]
        return in_data
