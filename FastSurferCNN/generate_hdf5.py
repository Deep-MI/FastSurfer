
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
import h5py
import numpy as np
import nibabel as nib

from os.path import join
from data_loader.load_neuroimaging_data import map_aparc_aseg2label, create_weight_mask, transform_sagittal, \
                                               transform_axial, get_thick_slices, filter_blank_slices_thick


# Class to create hdf5-file
class PopulationDataset:
    """
    Class to load all images in a directory into a hdf5-file.
    """

    def __init__(self, params):

        self.height = params["height"]
        self.width = params["width"]
        self.dataset_name = params["dataset_name"]
        self.data_path = params["data_path"]
        self.slice_thickness = params["thickness"]
        self.orig_name = params["image_name"]
        self.aparc_name = params["gt_name"]
        self.aparc_nocc = params["gt_nocc"]

        if params["csv_file"] is not None:
            with open(params["csv_file"], "r") as s_dirs:
                self.subject_dirs = [line.strip() for line in s_dirs.readlines()]

        else:
            self.search_pattern = join(self.data_path, params["pattern"])
            self.subject_dirs = glob.glob(self.search_pattern)

        self.data_set_size = len(self.subject_dirs)

    def create_hdf5_dataset(self, plane='axial', is_small=False):
        """
        Function to store all images in a given directory (or pattern) in a hdf5-file.
        :param str plane: which plane is processed (coronal, axial or saggital)
        :param bool is_small: small hdf5-file for pretraining?
        :return: None
        """
        start_d = time.time()

        # Prepare arrays to hold the data
        orig_dataset = np.ndarray(shape=(self.height, self.width, 0, 2 * self.slice_thickness + 1), dtype=np.float32)
        aseg_dataset = np.ndarray(shape=(self.height, self.width, 0), dtype=np.uint8)
        weight_dataset = np.ndarray(shape=(self.height, self.width, 0), dtype=np.float32)
        subjects = []

        # Loop over all subjects and load orig, aseg and create the weights
        for idx, current_subject in enumerate(self.subject_dirs):

            try:
                start = time.time()

                print("Volume Nr: {} Processing MRI Data from {}/{}".format(idx, current_subject, self.orig_name))

                # Load orig and aseg
                orig = nib.load(join(current_subject, self.orig_name))
                orig = np.asanyarray(orig.dataobj)

                aseg = nib.load(join(current_subject, self.aparc_name))

                print('Processing ground truth segmentation {}'.format(self.aparc_name))
                aseg = np.asanyarray(aseg.dataobj)

                if self.aparc_nocc is not None:
                    aseg_nocc = nib.load(join(current_subject, self.aparc_nocc))
                    aseg_nocc = np.asanyarray(aseg_nocc.dataobj)

                else:
                    aseg_nocc = None

                # Map aseg to label space and create weight masks
                if plane == 'sagittal':
                    _, mapped_aseg = map_aparc_aseg2label(aseg, aseg_nocc)
                    weights = create_weight_mask(mapped_aseg)
                    orig = transform_sagittal(orig)
                    mapped_aseg = transform_sagittal(mapped_aseg)
                    weights = transform_sagittal(weights)

                else:
                    mapped_aseg, _ = map_aparc_aseg2label(aseg, aseg_nocc)
                    weights = create_weight_mask(mapped_aseg)

                # Transform Data as needed (swap axis for axial view)
                if plane == 'axial':
                    orig = transform_axial(orig)
                    mapped_aseg = transform_axial(mapped_aseg)
                    weights = transform_axial(weights)

                # Create Thick Slices, filter out blanks
                orig_thick = get_thick_slices(orig, self.slice_thickness)
                orig, mapped_aseg, weights = filter_blank_slices_thick(orig_thick, mapped_aseg, weights)

                # Append finally processed images to arrays
                orig_dataset = np.append(orig_dataset, orig, axis=2)
                aseg_dataset = np.append(aseg_dataset, mapped_aseg, axis=2)
                weight_dataset = np.append(weight_dataset, weights, axis=2)

                sub_name = current_subject.split("/")[-1]
                subjects.append(sub_name.encode("ascii", "ignore"))

                end = time.time() - start

                print("Volume: {} Finished Data Reading and Appending in {:.3f} seconds.".format(idx, end))

                if is_small and idx == 2:
                    break

            except Exception as e:
                print("Volume: {} Failed Reading Data. Error: {}".format(idx, e))
                continue

        # Transpose to N, H, W, C and expand_dims for image

        orig_dataset = np.transpose(orig_dataset, (2, 0, 1, 3))
        aseg_dataset = np.transpose(aseg_dataset, (2, 0, 1))
        weight_dataset = np.transpose(weight_dataset, (2, 0, 1))

        # Write the hdf5 file
        with h5py.File(self.dataset_name, "w") as hf:
            hf.create_dataset('orig_dataset', data=orig_dataset, compression='gzip')
            hf.create_dataset('aseg_dataset', data=aseg_dataset, compression='gzip')
            hf.create_dataset('weight_dataset', data=weight_dataset, compression='gzip')

            dt = h5py.special_dtype(vlen=str)
            hf.create_dataset("subject", data=subjects, dtype=dt, compression="gzip")

        end_d = time.time() - start_d
        print("Successfully written {} in {:.3f} seconds.".format(self.dataset_name, end_d))


if __name__ == "__main__":

    import argparse

    # Training settings
    parser = argparse.ArgumentParser(description='HDF5-Creation')

    parser.add_argument('--hdf5_name', type=str, default="testsuite_2.hdf5",
                        help='path and name of hdf5-dataset (default: testsuite_2.hdf5)')
    parser.add_argument('--plane', type=str, default="axial", choices=["axial", "coronal", "sagittal"],
                        help="Which plane to put into file (axial (default), coronal or sagittal)")
    parser.add_argument('--height', type=int, default=256, help='Height of Image (Default 256)')
    parser.add_argument('--width', type=int, default=256, help='Width of Image (Default 256)')
    parser.add_argument('--data_dir', type=str, default="/testsuite", help="Directory with images to load")
    parser.add_argument('--thickness', type=int, default=3, help="Number of pre- and succeeding slices (default: 3)")
    parser.add_argument('--csv_file', type=str, default=None, help="Csv-file listing subjects to include in file")
    parser.add_argument('--pattern', type=str, default="*", help="Pattern to match files in directory.")
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

    args = parser.parse_args()

    network_params = {"dataset_name": args.hdf5_name, "height": args.height, "width": args.width,
                      "data_path": args.data_dir, "thickness": args.thickness, "csv_file": args.csv_file,
                      "pattern": args.pattern, "image_name": args.image_name,
                      "gt_name": args.gt_name, "gt_nocc": args.gt_nocc}

    stratified = PopulationDataset(network_params)
    stratified.create_hdf5_dataset(plane=args.plane)
