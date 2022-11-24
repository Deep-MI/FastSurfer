
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
import time
import glob
from os.path import join
from collections import defaultdict

import numpy as np
import nibabel as nib
import h5py

from FastSurferCNN.data_loader.data_utils import (transform_axial, transform_sagittal, map_aparc_aseg2label,
                                                  create_weight_mask, get_thick_slices, filter_blank_slices_thick,
                                                  read_classes_from_lut, get_labels_from_lut, unify_lateralized_labels)
from FastSurferCNN.utils import logging

LOGGER = logging.getLogger(__name__)


class H5pyDataset:

    def __init__(self, params, processing="aparc"):

        self.dataset_name = params["dataset_name"]
        self.data_path = params["data_path"]
        self.slice_thickness = params["thickness"]
        self.orig_name = params["image_name"]
        self.aparc_name = params["gt_name"]
        self.aparc_nocc = params["gt_nocc"]
        self.processing = processing

        self.available_sizes = params["sizes"]
        self.max_weight = params["max_weight"]
        self.edge_weight = params["edge_weight"]
        self.hires_weight = params["hires_weight"]
        self.gradient = params["gradient"]
        self.gm_mask = params["gm_mask"]

        self.lut = read_classes_from_lut(params["lut"])
        self.labels, self.labels_sag = get_labels_from_lut(self.lut, params["sag-mask"])
        self.lateralization = unify_lateralized_labels(self.lut, params["combi"])

        if params["csv_file"] is not None:
            with open(params["csv_file"], "r") as s_dirs:
                self.subject_dirs = [line.strip() for line in s_dirs.readlines()]

        else:
            self.search_pattern = join(self.data_path, params["pattern"])
            self.subject_dirs = glob.glob(self.search_pattern)

        self.data_set_size = len(self.subject_dirs)

    def _load_volumes(self, subject_path):
        # Load the orig and extract voxel spacing informatino (x, y, and z dim)
        LOGGER.info('Processing intensity image {} and ground truth segmentation {}'.format(self.orig_name, self.aparc_name))
        orig = nib.load(join(subject_path, self.orig_name))
        # Load the segmentation ground truth
        aseg = np.asarray(nib.load(join(subject_path, self.aparc_name)).get_fdata(), dtype=np.int16)

        zoom = orig.header.get_zooms()
        orig = np.asarray(orig.get_fdata(), dtype=np.uint8)

        if self.aparc_nocc is not None:
            aseg_nocc = nib.load(join(subject_path, self.aparc_nocc))
            aseg_nocc = np.asarray(aseg_nocc.get_fdata(), dtype=np.int16)
        else:
            aseg_nocc = None

        return orig, aseg, aseg_nocc, zoom

    def transform(self, plane, imgs, zoom):

        for i in range(len(imgs)):
            if plane == "sagittal":
                imgs[i] = transform_sagittal(imgs[i])
                zoom = zoom[::-1][:2]
            elif plane == "axial":
                imgs[i] = transform_axial(imgs[i])
                zoom = zoom[1:]
            else:
                zoom = zoom[:2]
        return imgs, zoom

    def _pad_image(self, img, max_out):
        # Get correct size = max along shape
        h, w, d = img.shape
        LOGGER.info("Padding image from {0} to {1}x{1}x{1}".format(img.shape, max_out))
        padded_img = np.zeros((max_out, max_out, max_out), dtype=img.dtype)
        padded_img[0: h, 0: w, 0:d] = img
        return padded_img

    def create_hdf5_dataset(self, plane='axial'):
        data_per_size = defaultdict(lambda: defaultdict(list))
        start_d = time.time()

        for idx, current_subject in enumerate(self.subject_dirs):

            try:
                start = time.time()

                LOGGER.info("Volume Nr: {} Processing MRI Data from {}/{}".format(idx+1, current_subject, self.orig_name))

                orig, aseg, aseg_nocc, zoom = self._load_volumes(current_subject)
                size, _, _ = orig.shape

                mapped_aseg, mapped_aseg_sag = map_aparc_aseg2label(aseg, self.labels, self.labels_sag,
                                                                    self.lateralization, aseg_nocc,
                                                                    processing=self.processing)

                if plane == 'sagittal':
                    mapped_aseg = mapped_aseg_sag
                    weights = create_weight_mask(mapped_aseg, max_weight=self.max_weight, ctx_thresh=19,
                                                 max_edge_weight=self.edge_weight, max_hires_weight=self.hires_weight,
                                                 cortex_mask=self.gm_mask, gradient=self.gradient)

                else:
                    weights = create_weight_mask(mapped_aseg, max_weight=self.max_weight, ctx_thresh=33,
                                                 max_edge_weight=self.edge_weight, max_hires_weight=self.hires_weight,
                                                 cortex_mask=self.gm_mask, gradient=self.gradient)

                print("Created weights with max_w {}, gradient {},"
                      " edge_w {}, hires_w {}, gm_mask {}".format(self.max_weight, self.gradient, self.edge_weight,
                                                                  self.hires_weight, self.gm_mask))

                # transform volumes to correct shape
                [orig, mapped_aseg, weights], zoom = self.transform(plane, [orig, mapped_aseg, weights], zoom)

                # Create Thick Slices, filter out blanks
                orig_thick = get_thick_slices(orig, self.slice_thickness)

                orig, mapped_aseg, weights = filter_blank_slices_thick(orig_thick, mapped_aseg, weights)

                num_batch = orig.shape[2]
                orig = np.transpose(orig, (2, 0, 1, 3))
                mapped_aseg = np.transpose(mapped_aseg, (2, 0, 1))
                weights = np.transpose(weights, (2, 0, 1))

                data_per_size[f'{size}']['orig'].extend(orig)
                data_per_size[f'{size}']['aseg'].extend(mapped_aseg)
                data_per_size[f'{size}']['weight'].extend(weights)
                data_per_size[f'{size}']['zoom'].extend((zoom,) * num_batch)
                sub_name = current_subject.split("/")[-1]
                data_per_size[f'{size}']['subject'].append(sub_name.encode("ascii", "ignore"))

            except Exception as e:
                LOGGER.info("Volume: {} Failed Reading Data. Error: {}".format(idx, e))
                continue

        for key, data_dict in data_per_size.items():
            data_per_size[key]['orig'] = np.asarray(data_dict['orig'], dtype=np.uint8)
            data_per_size[key]['aseg'] = np.asarray(data_dict['aseg'], dtype=np.uint8)
            data_per_size[key]['weight'] = np.asarray(data_dict['weight'], dtype=np.float)

        with h5py.File(self.dataset_name, "w") as hf:
            dt = h5py.special_dtype(vlen=str)
            for key, data_dict in data_per_size.items():
                group = hf.create_group(f"{key}")
                group.create_dataset("orig_dataset", data=data_dict['orig'])
                group.create_dataset("aseg_dataset", data=data_dict['aseg'])
                group.create_dataset("weight_dataset", data=data_dict['weight'])
                group.create_dataset("zoom_dataset", data=data_dict['zoom'])
                group.create_dataset("subject", data=data_dict['subject'], dtype=dt)

        end_d = time.time() - start_d
        LOGGER.info("Successfully written {} in {:.3f} seconds.".format(self.dataset_name, end_d))


if __name__ == '__main__':
    import argparse

    # Training settings
    parser = argparse.ArgumentParser(description='HDF5-Creation')

    parser.add_argument('--hdf5_name', type=str, default="../data/hdf5_set/Multires_coronal.hdf5",
                        help='path and name of hdf5-data_loader (default: ../data/hdf5_set/Multires_coronal.hdf5)')
    parser.add_argument('--plane', type=str, default="axial", choices=["axial", "coronal", "sagittal"],
                        help="Which plane to put into file (axial (default), coronal or sagittal)")
    parser.add_argument('--data_dir', type=str, default="/data", help="Directory with images to load")
    parser.add_argument('--thickness', type=int, default=3, help="Number of pre- and succeeding slices (default: 3)")
    parser.add_argument('--csv_file', type=str, default=None, help="Csv-file listing subjects to include in file")
    parser.add_argument('--pattern', type=str, help="Pattern to match files in directory.")
    parser.add_argument('--image_name', type=str, default="mri/orig.mgz",
                        help="Default name of original images. FreeSurfer orig.mgz is default (mri/orig.mgz)")
    parser.add_argument('--gt_name', type=str, default="mri/aparc.DKTatlas+aseg.mgz",
                        help="Default name for ground truth segmentations. Default: mri/aparc.DKTatlas+aseg.mgz."
                             " If Corpus Callosum segmentation is already removed, do not set gt_nocc."
                             " (e.g. for our internal training set mri/aparc.DKTatlas+aseg.filled.mgz exists already"
                             " and should be used here instead of mri/aparc.DKTatlas+aseg.mgz). ")
    parser.add_argument('--gt_nocc', type=str, default=None,
                        help="Segmentation without corpus callosum (used to mask this segmentation in ground truth)."
                             " If the used segmentation was already processed, do not set this argument."
                             " For a normal FreeSurfer input, use mri/aseg.auto_noCCseg.mgz.")
    parser.add_argument('--lut', type=str, default='./config/FastSurfer_ColorLUT.tsv',
                        help="FreeSurfer-style Color Lookup Table with labels to use in final prediction. "
                             "Has to have columns: ID	LabelName	R	G	B	A"
                             "Default: ./config/FastSurfer_ColorLUT.tsv.")
    parser.add_argument('--combi', action='append', default=["Left-", "Right-"],
                        help="Suffixes of labels names to combine. Default: Left- and Right-.")
    parser.add_argument('--sag_mask', default=("Left-", "ctx-rh"),
                        help="Suffixes of labels names to mask for final sagittal labels. Default: Left- and ctx-rh.")
    parser.add_argument('--max_w', type=int, default=5,
                        help="Overall max weight for any voxel in weight mask. Default=5")
    parser.add_argument('--edge_w', type=int, default=5, help="Weight for edges in weight mask. Default=5")
    parser.add_argument('--hires_w', type=int, default=None,
                        help="Weight for hires elements (sulci, WM strands, cortex border) in weight mask. Default=None")
    parser.add_argument('--no_grad', action='store_true', default=False,
                        help="Turn on to only use median weight frequency (no gradient)")
    parser.add_argument('--gm', action="store_true", default=False,
                        help="Turn on to add cortex mask for hires-processing.")
    parser.add_argument('--processing', type=str, default="aparc", choices=["aparc", "aseg", "none"],
                        help="Use aseg, aparc or no specific mapping processing")
    parser.add_argument('--sizes', nargs='+', type=int, default=256, help="Sizes of images in the dataset. Default: 256")


    args = parser.parse_args()

    dataset_params = {"dataset_name": args.hdf5_name, "data_path": args.data_dir, "thickness": args.thickness,
                      "csv_file": args.csv_file, "pattern": args.pattern, "image_name": args.image_name,
                      "gt_name": args.gt_name, "gt_nocc": args.gt_nocc, "sizes": args.sizes,
                      "max_weight": args.max_w, "edge_weight": args.edge_w,
                      "lut": args.lut, "combi": args.combi, "sag-mask": args.sag_mask,
                      "hires_weight": args.hires_w, "gm_mask": args.gm, "gradient": not args.no_grad}

    dataset_generator = H5pyDataset(params=dataset_params, processing=args.processing)
    dataset_generator.create_hdf5_dataset(plane=args.plane)

