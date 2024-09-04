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
import time
import warnings
from collections import defaultdict
from os.path import isfile, join

import h5py
import numpy as np

from CerebNet.datasets.load_data import SubjectLoader


# Class to create hdf5-file
class CerebNetDataset:
    """
    Class to load all images in a directory into a hdf5-file.
    """

    def __init__(self, cfg):

        self.cfg = cfg
        self.size = cfg.PATCH_SIZE
        self.load_aux_data = cfg.LOAD_AUXILIARY_DATA
        aux_subjects_files = {}
        if self.load_aux_data:
            aux_subjects_files = self._read_warp_dict()
        self.subj_loader = SubjectLoader(cfg, aux_subjects_files)

    def _save_hdf5_file(self, datasets, dataset_name):
        # Write the hdf5 file
        with h5py.File(dataset_name, "w") as hf:
            for name, vol in datasets.items():
                if name != "subject":
                    hf.create_dataset(f"{name}", data=vol, compression="gzip")
                else:
                    dt = h5py.special_dtype(vlen=str)
                    hf.create_dataset(
                        f"{name}", data=datasets[name], dtype=dt, compression="gzip"
                    )

    def _read_warp_dict(self):
        """
        It loads txt/csv file with a list of fixed to moving subjects as follows:
            [fixed_subj_id], [moving_subj1_id], [moving_subj2_id], [moving_subj3_id]
            .
            .

        Returns:
            dictionary of moving ids to a list of fixed subject ids
        """
        subj2warps = defaultdict(list)
        all_imgs = []
        with open(join(self.cfg.REG_DATA_DIR, self.cfg.REG_DATA_CSV)) as f:
            for line in f.readlines():
                line = line.strip()
                ids = line.split(",")
                for i in ids[1:]:
                    img_path = join(
                        self.cfg.REG_DATA_DIR,
                        f"{i}_to_{ids[0]}",
                        self.cfg.AUXILIARY_IMAGE,
                    )
                    lbl_path = join(
                        self.cfg.REG_DATA_DIR,
                        f"{i}_to_{ids[0]}",
                        self.cfg.AUXILIARY_LABEL,
                    )
                    if isfile(img_path) and isfile(lbl_path):
                        all_imgs.append(img_path)
                        subj2warps[i].append((img_path, lbl_path))
                    else:
                        warnings.warn(f"Warp field at {img_path} not found.", stacklevel=2)
        return subj2warps

    def create_hdf5_dataset(
        self, subjects_list, dataset_name, store_talairach=False, load_aux_data=False
    ):
        """
        Function to store all images in a given directory (or pattern) in a hdf5-file.
        :return:
        """
        start_d = time.time()
        datasets = {}
        # Prepare arrays to hold the data
        datasets["img"] = np.ndarray(
            shape=(0, self.size[0], self.size[1], self.size[2]), dtype=np.uint8
        )
        datasets["label"] = np.ndarray(
            shape=(0, self.size[0], self.size[1], self.size[2]), dtype=np.uint8
        )
        datasets["subject"] = []

        if store_talairach:
            datasets["talairach"] = np.ndarray(
                shape=(0, self.size[0], self.size[1], self.size[2], 3), dtype=np.float16
            )

        if load_aux_data:
            datasets["auxiliary_img"] = np.ndarray(
                shape=(0, self.size[0], self.size[1], self.size[2]), dtype=np.uint8
            )
            datasets["auxiliary_lbl"] = np.ndarray(
                shape=(0, self.size[0], self.size[1], self.size[2]), dtype=np.uint8
            )

        # Loop over all subjects and load orig, aseg and create the weights
        for idx, current_subject in enumerate(subjects_list):
            # try:
            start = time.time()
            print(
                f"Volume Nr: {idx + 1}/{len(subjects_list)} Processing MRI Data from {current_subject}"
            )

            in_data = self.subj_loader.load_subject(
                current_subject,
                store_talairach=store_talairach,
                load_aux_data=load_aux_data,
            )

            # Append finally processed images to arrays
            for name, vol in in_data.items():
                if len(vol) > 0:
                    datasets[name] = np.append(datasets[name], vol, axis=0)

            sub_name = current_subject.split("/")[-1]
            datasets["subject"].append(sub_name.encode("ascii", "ignore"))

            end = time.time() - start
            print("Number of Cerebellum classes", len(np.unique(in_data["label"])))
            print(
                f"Volume: {idx + 1} Finished Data Reading and Appending in {end:.3f} seconds."
            )

            # except Exception as e:
            #     print("Volume: {} Failed Reading Data. Error: {}".format(idx, e))
            #     continue

        self._save_hdf5_file(datasets, dataset_name)
        end_d = time.time() - start_d
        print(f"Successfully written {dataset_name} in {end_d:.3f} seconds.")
