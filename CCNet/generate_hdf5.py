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
from ast import Tuple
from calendar import c
import os.path
from re import T, X
# IMPORTS
import time
import glob
from os.path import join
from collections import defaultdict
import json
import csv
from typing import Optional, Union

import numpy as np
import nibabel as nib
import h5py
from sympy import true

from FastSurferCNN.data_loader.data_utils import (transform_axial, transform_sagittal, map_aparc_aseg2label,
                                                  create_weight_mask, get_thick_slices, filter_blank_slices_thick,
                                                  read_classes_from_lut, get_labels_from_lut, unify_lateralized_labels,
                                                  map_incrementing, subset_volume_plane)
from FastSurferCNN.utils import logging
from CCNet.utils.misc import calculate_centers_of_comissures


LOGGER = logging.getLogger(__name__)


class H5pyDataset:

    def __init__(self, params, processing="aparc"):

        self.debug = params["debug"]
        self.dataset_name = params["dataset_name"]
        self.data_path = params["data_path"]
        self.slice_thickness = params["thickness"]
        self.orig_name = params["image_name"]
        self.aparc_name = params["gt_name"]
        self.aparc_nocc = params["gt_nocc"]
        self.aux_data = params["aux_data"]
        self.processing = processing

        self.max_weight = params["max_weight"]
        self.edge_weight = params["edge_weight"]
        self.hires_weight = params["hires_weight"]
        self.gradient = params["gradient"]
        self.gm_mask = params["gm_mask"]
        self.pad = params["pad"]
        self.crop = params["crop"]

        self.lut = read_classes_from_lut(params["lut"])
        self.labels, self.labels_sag = get_labels_from_lut(self.lut, params["sag-mask"])
        self.lateralization = unify_lateralized_labels(self.lut, params["combi"])

        self.centers = {} # RAS coordinates of the center of the commisures [PC, AC]

        if params["center"] is not None:
            with open(params["center"], 'r') as center_csv_file:
                rows = list(csv.reader(center_csv_file, delimiter=","))
                for row in rows[1:]:
                    pc_center = row[1:4]
                    ac_center = row[4:7]
                    self.centers[row[0]] = [pc_center, ac_center]



        # init files paths
        self.subject_dirs = []
        self.gt_files = []
        self.gt_nocc_files = None

        

        if params["csv_file"] is not None:
            # read from csv file
            with open(params["csv_file"], "r") as s_dirs:
                complete_list = list(csv.reader(s_dirs, delimiter=","))

                # csv file contains subjects
                self.subject_dirs = [subject[0] for subject in complete_list]

                if (len(complete_list[0]) >= 2) and (complete_list[0][1] is not None):
                    # csv file contains gt in second column
                    self.gt_files = [subject[1] for subject in complete_list]

                if (len(complete_list[0]) >= 3) and (complete_list[0][2] is not None):
                    # csv file contains gt_nocc in 3rd column
                    self.gt_nocc_files = [subject[2] for subject in complete_list]

        elif (self.data_path is not None) and (params["pattern"] is not None):
            self.search_pattern = join(self.data_path, "/", params["pattern"])
            self.s = glob.glob(self.search_pattern)
        else:
            raise ValueError("No valid subject path list could be created")

        # Set subject file names
        if os.path.isfile(self.subject_dirs[0]):
            self.subject_files = self.subject_dirs
            self.subject_dirs = [os.path.dirname(subject) for subject in self.subject_dirs]
        elif self.orig_name is not None:
            self.subject_files = [join(subject, self.orig_name) for subject in self.subject_dirs]
        else:
            raise ValueError("No valid subject file list could be created")

        # Set gt file name
        if self.gt_files is None:
            self.gt_files = self.subject_dirs

        if os.path.isdir(self.gt_files[0]):
            if self.aparc_name is not None:
                self.gt_files = [join(subject, self.aparc_name) for subject in self.gt_files]
            else:
                raise ValueError("No valid gt file list could be created")

        #if self.gt_nocc_files is not []:
        #    if (os.path.isdir(self.gt_nocc_files[0]) and (self.aparc_nocc is not None)):
        #        self.gt_nocc_files = [join(subject, self.aparc_nocc) for subject in self.subject_dirs]

        self.data_set_size = len(self.subject_dirs)

        for subject in self.subject_files:
            assert os.path.isfile(subject), f"{subject} is not a file"
        for subject in self.gt_files:
            assert os.path.isfile(subject), f"{subject} is not a file"
        
    
    def _load_volumes(self, gt_file, orig_file, gt_nocc_file=None, subject_path=None):
        # Load the orig and extract voxel spacing information (x, y, and z dim)
        LOGGER.info('Processing intensity image {} and ground truth segmentation {}'.format(self.orig_name, gt_file))

        # Load the orignal image and zoom
        orig = nib.load(orig_file)
        zoom = orig.header.get_zooms()
        orig = np.asarray(orig.get_fdata(), dtype=np.uint8)

        # Load the segmentation ground truth
        aseg = np.asarray(nib.load(gt_file).get_fdata(), dtype=np.uint16)

        if gt_nocc_file is not None:
            aseg_nocc = nib.load(gt_nocc_file)
            aseg_nocc = np.asarray(aseg_nocc.get_fdata(), dtype=np.uint16)
            assert (aseg_nocc.shape == aseg.shape), "Aseg and aseg_nocc must have the same shape"
        else:
            aseg_nocc = None

        # LUT for aux data:
        # 0: unknown
        # 1: brain
        # 2: midplane
        # 3: anomaly

        if self.aux_data is not None:
            if 'brainmask' in self.aux_data.keys():
                aux_data_brainmask = np.asarray(nib.load(join(subject_path, self.aux_data['brainmask'])).get_fdata())
                aux_data_brainmask = (aux_data_brainmask > 0).astype(np.uint8)

                # assert(np.unique(aux_data).shape[0] == 2), "Auxiliary data must be binary"
                assert (
                            aux_data_brainmask.shape == orig.shape), "Auxiliary data must have the same shape as the original image"
            else:
                aux_data_brainmask = np.zeros_like(orig, dtype=np.uint8)

            if aseg_nocc is not None:
                try:
                    aseg_cc = nib.load(join(subject_path, 'mri/aparc+aseg.mgz')).get_fdata()
                except:
                    aseg_cc = nib.load(gt_file).get_fdata()
                cc_center_mask = aseg_cc == 253  # get midplane from CC mask
                assert (cc_center_mask.any()), "Fornix mask is empty"
                cc_coordinates = np.where(cc_center_mask)
                midplane = np.zeros_like(orig, dtype=bool)
                LR_coordinate = int(np.median(cc_coordinates[0]))
                # print("midplane at ", LR_coordinate)
                midplane[LR_coordinate, :, :] = True

                if 'brainmask' in self.aux_data.keys():
                    midplane = midplane & (aux_data_brainmask == 1)
                    aux_data_brainmask[midplane] = 2
                    aux_data_brainmask = aux_data_brainmask
                else:
                    aux_data_brainmask[midplane] = 2

            if 'anomaly' in self.aux_data.keys():
                aux_data_anomaly = np.asarray(nib.load(join(subject_path, self.aux_data['anomaly'])).get_fdata(),
                                              dtype=bool)
                aux_data_brainmask[aux_data_anomaly] = 3

            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(20, 8))
            # plt.subplot(1, 3, 1)
            # plt.imshow(orig[LR_coordinate, :, :], cmap='gray')
            # plt.subplot(1, 3, 2)
            # plt.title("aux_data")
            # plt.imshow(aux_data[LR_coordinate, :, :], cmap='gray')
            # plt.subplot(1, 3, 3)
            # plt.title("aux_data")
            # plt.imshow(aux_data[:, 128, :], cmap='Pastel1')
            # plt.savefig("aux_data.png", dpi=300)

            # nib.save(nib.Nifti1Image(aux_data, np.eye(4)), "aux_data.nii.gz")
        else:
            aux_data_brainmask = None
        return orig, aseg, aseg_nocc, aux_data_brainmask, zoom

    def transform(self, plane, imgs, zoom):
        """
        Change axis of all volumes so that the slice thickness dimension is the first dimension
        and select the zooms in 2d accordingly
        """
        if plane == "sagittal":
            for i in range(len(imgs)):
                imgs[i] = transform_sagittal(imgs[i])
            zoom = [zoom[i] for i in [1, 0]]  # zoom[::-1][:2]     21
        elif plane == "axial":
            for i in range(len(imgs)):
                imgs[i] = transform_axial(imgs[i])
            zoom = [zoom[i] for i in [2, 0]]  # zoom[1:]           12
        elif plane == 'coronal':  # no image axis changes for coronal plane
            zoom = [zoom[i] for i in [1, 2]]  # zooms = zoom[:2]   01
        else:
            raise ValueError("Plane {} not supported".format(plane))
        return imgs, zoom

    def _pad_image(self, img, max_out):
        # Get correct size = max along shape
        h, w, d = img.shape
        LOGGER.info("Padding image from {0} to {1}x{1}x{1}".format(img.shape, max_out))
        padded_img = np.zeros((max_out, max_out, max_out), dtype=img.dtype)
        padded_img[0: h, 0: w, 0:d] = img
        return padded_img

    
    
    def calculate_center_volume(self, orig_file, centers : np.ndarray, l : int = 5, sig : float = 3., offset : int = 0):
        """Calculates a gaussian kernel around the given centers and adds them to a volume
        
        Arguments:
            orig_file (str): path to original image
            centers (np.ndarray): array of RAS coordinates of the centers (In the same space as orig file)
            l (int): size of the kernel (default: {5})
            sig (float): sigma of the kernel (default: {1.})
            
        Returns:
            np.ndarray: volume with gaussian kernels around the centers
        
        """
        assert l % 2 == 1, "l must be odd"
        orig = nib.load(orig_file)
        
        # create empty volume
        center_volume = np.zeros_like(orig.get_fdata(), dtype=np.float32)

        centers_vox = np.array(centers, dtype=np.float32)

        # add gkern around centers
        for center in centers_vox:
            center = np.rint(center).astype(int)
            kernel = self.gkern(l=l, sig=sig)

            # Calculate the start and end indices for adding the kernel to the center_volume
            start_idx = np.maximum(center - (l - 1) // 2, 0).astype(int)
            end_idx = np.minimum(center + (l - 1) // 2 + 1, center_volume.shape).astype(int)

            # Calculate the start and end indices for the kernel
            kernel_start_idx = np.maximum((l - 1) // 2 - center, 0).astype(int)
            kernel_end_idx = np.minimum((l - 1) // 2 + np.array(center_volume.shape) - center, l).astype(int)

            # check if centers are overlapping
            assert np.all(center_volume[start_idx[0]:end_idx[0], 
                                         start_idx[1]:end_idx[1], 
                                         start_idx[2]:end_idx[2]] == 0), "Centers are overlapping"

            # Add the weighted kernel to the center_volume
            center_volume[start_idx[0]:end_idx[0], 
                          start_idx[1]:end_idx[1], 
                          start_idx[2]:end_idx[2]] += kernel[
                              kernel_start_idx[0]:kernel_end_idx[0], 
                              kernel_start_idx[1]:kernel_end_idx[1],
                              kernel_start_idx[2]:kernel_end_idx[2]]

        tmp = calculate_centers_of_comissures(center_volume)
        return center_volume
    

    def gkern(self, l=5, sig=1.):
        """\
        creates gaussian kernel with side length `l` and a sigma of `sig`
        """
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)

        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)

        kernel = kernel.T[:,:,None]*kernel.T[:,None]

        return kernel / np.max(kernel)


    def create_hdf5_dataset(self, plane='axial'):
        data_per_size = defaultdict(lambda: defaultdict(list))
        start_d = time.time()
        failed = 0

        for idx in range(len(self.subject_dirs)):
            if true:
                current_path = self.subject_dirs[idx]
                current_orig = self.subject_files[idx]
                current_gt = self.gt_files[idx]

                if current_path.endswith('/'):
                    sub_name = current_path.split("/")[-2]
                else:
                    sub_name = current_path.split("/")[-1]

                if self.gt_nocc_files != None:
                    current_gt_nocc = self.gt_nocc_files[idx]
                else:
                    current_gt_nocc = None

                LOGGER.info("Volume Nr: {} Processing MRI Data from {}".format(idx + 1, current_path))

                orig, aseg, aseg_nocc, aux_data, zoom = self._load_volumes(gt_file=current_gt,
                                                                           orig_file=current_orig,
                                                                           gt_nocc_file=current_gt_nocc,
                                                                           subject_path=current_path)

                # Create ACPC volumes
                if self.centers != {}:
                    center_volume = self.calculate_center_volume(current_orig, self.centers[sub_name])
                    #center_volume = self.calculate_center_volume2(current_orig, self.centers[sub_name])


                if self.crop: 
                    axis = 0 #sagittal
                    mid_plane = 128 #mri_cc default
                    volume_thickness = np.ceil((5/zoom[axis]))
                    if volume_thickness % 2 == 0:
                        volume_thickness += 1

                    if plane=="sagittal" and not self.pad:
                        orig = subset_volume_plane(orig, plane=mid_plane, thickness=volume_thickness+args.thickness*2, axis=axis)
                    else:
                        orig = subset_volume_plane(orig, plane=mid_plane, thickness=volume_thickness, axis=axis)


                    aseg = subset_volume_plane(aseg, plane=mid_plane, thickness=volume_thickness, axis=axis)
                    if aux_data is not None:
                        aux_data = subset_volume_plane(aux_data, plane=mid_plane, thickness=volume_thickness, axis=axis)
                    if aseg_nocc is not None:
                        aseg_nocc = subset_volume_plane(aseg_nocc, plane=mid_plane, thickness=volume_thickness, axis=axis)
                    if self.centers != {}:
                        center_volume = subset_volume_plane(center_volume, plane=mid_plane, thickness=volume_thickness, axis=axis)

                

                mapped_aseg, mapped_aseg_sag = map_aparc_aseg2label(aseg, self.labels, self.labels_sag,
                                                                    self.lateralization, aseg_nocc,
                                                                    processing=self.processing)

                if plane == 'sagittal':
                    mapped_aseg = mapped_aseg_sag

                mapped_aseg = map_incrementing(mapped_aseg.copy(), lut=self.lut)
                weights = create_weight_mask(mapped_aseg.copy(), max_weight=self.max_weight, ctx_thresh=19 if plane == 'sagittal' else 33,
                                                max_edge_weight=self.edge_weight, max_hires_weight=self.hires_weight,
                                                cortex_mask=self.gm_mask, gradient=self.gradient)


                LOGGER.info("Created weights with max_w {}, gradient {},"
                        " edge_w {}, hires_w {}, gm_mask {}".format(self.max_weight, self.gradient, self.edge_weight,
                                                                    self.hires_weight, self.gm_mask))

                # transform volumes to correct shape
                if aux_data is not None:
                    [orig, mapped_aseg, weights, aux_data], zoom = self.transform(plane, [orig, mapped_aseg, weights,
                                                                                          aux_data], zoom)
                elif (self.centers != {}):
                    [orig, mapped_aseg, weights, center_volume], zoom = self.transform(plane, [orig, mapped_aseg, weights, center_volume], zoom)
                else:
                    [orig, mapped_aseg, weights], zoom = self.transform(plane, [orig, mapped_aseg, weights], zoom)

                assert (len(zoom) == 2), "Zoom should be 2D"


                # Create Thick Slices, filter out blanks
                orig_thick = get_thick_slices(orig, self.slice_thickness, pad=self.pad)

                if aux_data is not None:
                    aux_data = get_thick_slices(aux_data, self.slice_thickness, pad = self.pad) if aux_data is not None else None
                    orig, mapped_aseg, weights, aux_data = filter_blank_slices_thick(
                        [orig_thick, mapped_aseg, weights, aux_data], label_vol=mapped_aseg)
                elif (self.centers != {}):
                    orig, mapped_aseg, weights, center_volume = filter_blank_slices_thick(
                        [orig_thick, mapped_aseg, weights, center_volume], label_vol=mapped_aseg)
                else:
                    orig, mapped_aseg, weights = filter_blank_slices_thick([orig_thick, mapped_aseg, weights],
                                                                           label_vol=mapped_aseg)

                num_batch = orig.shape[2]
                orig = np.transpose(orig, (2, 0, 1,
                                           3))  # shape: (plane1, plane2, no_thick_slices, slice_thickness) -> (no_thick_slices, plane1, plane2, slice_thickness)
                mapped_aseg = np.transpose(mapped_aseg, (2, 0, 1))  # put no_thick_slices as first dimension
                weights = np.transpose(weights, (2, 0, 1))  # put no_thick_slices as first dimension
                if aux_data is not None:
                    aux_data = np.transpose(aux_data, (2, 0, 1, 3))  # put no_thick_slices as first dimension
                
                if self.centers != {}:
                    center_volume = np.transpose(center_volume, (2, 0, 1))  # put no_thick_slices as first dimension

                assert (orig.shape[0] == mapped_aseg.shape[0] == weights.shape[0]), "Number of slices does not match"
                assert (orig.shape[1] == mapped_aseg.shape[1] == weights.shape[1]), "Number of rows does not match"
                assert (orig.shape[2] == mapped_aseg.shape[2] == weights.shape[2]), "Number of columns does not match"

                if orig.shape[1] == orig.shape[2]:
                    size = orig.shape[1]
                else:
                    LOGGER.info("Image is not isotropic; using both dimenstions as key")
                    size = f"{orig.shape[1]}-{orig.shape[2]}"

                assert not (np.sum(mapped_aseg, axis=(1, 2)) == 0).any(), "Empty labels in mapped_aseg"

                data_per_size[f'{size}']['orig'].extend(orig)  # add slices to list
                data_per_size[f'{size}']['aseg'].extend(mapped_aseg)

                data_per_size[f'{size}']['weight'].extend(weights)
                if aux_data is not None:
                    data_per_size[f'{size}']['aux_data'].extend(aux_data)
                data_per_size[f'{size}']['zoom'].extend((zoom,) * num_batch)

                data_per_size[f'{size}']['subject'].extend([sub_name.encode("ascii", "ignore")] * len(orig))  # add subject name to each slice

                if self.centers != {}:
                #    assert not (np.sum(center_volume, axis=(1, 2)) == 0).all(), "Empty center_volume"
                    data_per_size[f'{size}']['center_volume'].extend(center_volume)

                if self.debug and idx == 20:
                    break

            #except AssertionError as e:
            #    LOGGER.warning("Volume: {} Failed Reading Data. Error: {}".format(idx, e))
            #    failed += 1
            #    continue

        for key, data_dict in data_per_size.items():
            data_per_size[key]['orig'] = np.asarray(data_dict['orig'], dtype=np.uint8)
            data_per_size[key]['aseg'] = np.asarray(data_dict['aseg'], dtype=np.uint8)
            data_per_size[key]['weight'] = np.asarray(data_dict['weight'], dtype=float)
            
            if 'center_volume' in data_dict.keys():
                data_per_size[key]['center_volume'] = np.asarray(data_dict['center_volume'], dtype=float)

            assert (data_dict['orig'].shape[0] == data_dict['aseg'].shape[0] == data_dict['weight'].shape[0] == len(
                data_dict['zoom'])
                    ), 'Data does have not the same length, but orig:{} aseg:{} weight:{} zoom:{}'.format(
                data_dict['orig'].shape[0], data_dict['aseg'].shape[0], data_dict['weight'].shape[0],
                data_dict['zoom'].shape[0])
            assert (len(data_dict['subject']) == data_dict['orig'].shape[
                0]), f'Subject and data do not have the same length, but subject:{data_dict["subject"].shape[0]} data:{data_dict["orig"].shape[0]}'

        with h5py.File(self.dataset_name, "w") as hf:
            dt = h5py.special_dtype(vlen=str)
            for key, data_dict in data_per_size.items():
                group = hf.create_group(f"{key}")
                group.create_dataset("orig_dataset", data=data_dict['orig'], chunks=True)
                group.create_dataset("aseg_dataset", data=data_dict['aseg'])
                group.create_dataset("weight_dataset", data=data_dict['weight'])
                if 'aux_data' in data_dict.keys():
                    group.create_dataset("aux_data", data=data_dict['aux_data'])
                if 'center_volume' in data_dict.keys():
                    group.create_dataset("center_dataset", data=data_dict['center_volume'])
                group.create_dataset("zoom_dataset", data=data_dict['zoom'])
                group.create_dataset("subject", data=data_dict['subject'], dtype=dt)

        end_d = time.time() - start_d
        LOGGER.info("Successfully written {} in {:.3f} seconds.".format(self.dataset_name, end_d))
        if failed > 0:
            LOGGER.warning("Failed to read {} volumes.".format(failed))
        else:
            LOGGER.info("No failures while reading volumes.")


class H5pyDatasetMasks(H5pyDataset):

    def __init__(self, params):
        self.debug = params["debug"]
        self.dataset_name = params["dataset_name"]
        self.data_path = params["data_path"]
        self.slice_thickness = params["thickness"]
        self.mask_name = params["image_name"]

        # self.lateralization = unify_lateralized_labels(self.lut, params["combi"])

        if params["csv_file"] is not None:
            with open(params["csv_file"], "r") as s_dirs:
                self.subject_dirs = [line.strip() for line in s_dirs.readlines()]

        else:
            self.search_pattern = join(self.data_path, params["pattern"])
            self.subject_dirs = glob.glob(self.search_pattern)

        self.data_set_size = len(self.subject_dirs)

        self.aux_data = None
        self.aseg_nocc = None

    def _load_volumes(self, subject_path):
        # return super()._load_volumes(subject_path)
        # Load the orig and extract voxel spacing information (x, y, and z dim)
        LOGGER.info(f'Processing mask image {self.mask_name}')
        mask = nib.load(join(subject_path, self.mask_name) if self.mask_name != "" else subject_path)

        zoom = mask.header.get_zooms()
        mask = np.asarray(mask.get_fdata(), dtype=bool)

        return mask, zoom

    def create_hdf5_dataset(self, plane='all'):
        data_per_size = defaultdict(lambda: defaultdict(list))
        start_d = time.time()
        failed = 0

        for idx, current_subject in enumerate(self.subject_dirs):

            try:
                LOGGER.info(
                    "Volume Nr: {} Processing MRI Data from {}/{}".format(idx + 1, current_subject, self.mask_name))

                mask, zoom = self._load_volumes(current_subject)

                if not (mask.shape[0] == mask.shape[1] == mask.shape[2]):
                    # LOGGER.warning(f"Image is not isotropic, but has size {mask.shape}")
                    # np.pad(mask, ((0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
                    mask = self._pad_image(mask, max_out=max(mask.shape))

                assert (zoom[0] == zoom[1] == zoom[2]), f"Zoom should be isotropic, but is {zoom}"

                size = mask.shape[0]

                # assert(mask.shape[1] == mask.shape[2] == size), f"Image is not isotropic, but has size {mask.shape}"

                # transform volumes to correct shape
                if plane == 'all':
                    [mask_axial], _ = self.transform('axial', [mask], zoom)
                    [mask_coronal], _ = self.transform('coronal', [mask], zoom)
                    [mask_sagittal], zoom = self.transform('sagittal', [mask], zoom)
                    mask = np.concatenate([mask_axial, mask_coronal, mask_sagittal], axis=2)
                else:
                    [mask], zoom = self.transform(plane, [mask], zoom)

                assert (len(zoom) == 2), "Zoom should be 2D"

                # Create Thick Slices, filter out blanks
                mask_thick = get_thick_slices(mask, self.slice_thickness, pad=self.pad)

                # filter blank slices
                select_slices = (np.sum(mask_thick,
                                        axis=(0, 1, 3)) > 10)  # select thick-slices with more than 10 voxels of mask
                mask_thick = mask_thick[:, :, select_slices, :]

                num_batch = mask_thick.shape[2]
                mask = np.transpose(mask_thick, (2, 0, 1,
                                                 3))  # shape: (plane1, plane2, no_thick_slices, slice_thickness) -> (no_thick_slices, plane1, plane2, slice_thickness)

                data_per_size[f'{size}']['mask'].extend(mask)  # add slices to list
                data_per_size[f'{size}']['zoom'].extend((zoom,) * num_batch)
                sub_name = current_subject.split("/")[-1]
                data_per_size[f'{size}']['subject'].extend(
                    [sub_name.encode("ascii", "ignore")] * len(mask))  # add subject name to each slice

                if self.debug and idx == 20:
                    break

            except Exception as e:
                LOGGER.info("Volume: {} Failed Reading Data. Error: {}".format(idx, e))
                failed += 1
                continue

        for key, data_dict in data_per_size.items():
            data_per_size[key]['mask'] = np.asarray(data_dict['mask'], dtype=bool)
            
            assert (len(data_dict['subject']) == data_dict['mask'].shape[0] == len(data_dict['zoom'])), \
                f'Subject and data do not have the same length, but subject:{data_dict["subject"].shape[0]} data:{data_dict["mask"].shape[0]}, zoom:{len(data_dict["zoom"])}'

        with h5py.File(self.dataset_name, "w") as hf:
            dt = h5py.special_dtype(vlen=str)
            for key, data_dict in data_per_size.items():
                group = hf.create_group(f"{key}")
                group.create_dataset("mask_dataset", data=data_dict['mask'])
                group.create_dataset("zoom_dataset", data=data_dict['zoom'])
                group.create_dataset("subject", data=data_dict['subject'], dtype=dt)

        end_d = time.time() - start_d
        LOGGER.info("Successfully written {} in {:.3f} seconds.".format(self.dataset_name, end_d))
        if failed > 0:
            LOGGER.warning("Failed to read {} volumes.".format(failed))
        else:
            LOGGER.info("No failures while reading volumes.")


if __name__ == '__main__':
    import argparse

    # Training settings
    parser = argparse.ArgumentParser(description='HDF5-Creation')

    parser.add_argument('--hdf5_name', type=str, default="../data/hdf5_set/Multires_coronal.hdf5",
                        help='path and name of hdf5-data_loader (default: ../data/hdf5_set/Multires_coronal.hdf5)')
    parser.add_argument('--plane', type=str, default="axial", choices=["axial", "coronal", "sagittal", "all"],
                        help="Which plane to put into file (axial (default), coronal or sagittal)")
    parser.add_argument('--data_dir', type=str, default="/data", help="Directory with images to load")
    parser.add_argument('--thickness', type=int, default=3, help="Number of pre- and succeeding slices (default: 3)")
    parser.add_argument('--csv_file', type=str, default=None, help="Csv-file listing subjects to include in file")
    parser.add_argument('--pattern', type=str, help="Pattern to match files in directory.")
    parser.add_argument('--image_name', type=str,
                        help="Default name of original images. FreeSurfer orig.mgz is default (mri/orig.mgz)")
    parser.add_argument('--gt_name', type=str, default=None,
                        help="Default name for ground truth segmentations. Default: mri/aparc.DKTatlas+aseg.mgz."
                             " If Corpus Callosum segmentation is already removed, do not set gt_nocc."
                             " (e.g. for our internal training set mri/aparc.DKTatlas+aseg.filled.mgz exists already"
                             " and should be used here instead of mri/aparc.DKTatlas+aseg.mgz). ")
    parser.add_argument('--gt_nocc', type=str, default=None,
                        help="Segmentation without corpus callosum (used to mask this segmentation in ground truth)."
                             " If the used segmentation was already processed, do not set this argument."
                             " For a normal FreeSurfer input, use mri/aseg.auto_noCCseg.mgz.")
    parser.add_argument('--aux_data', type=str, default=None
                        , help="Auxiliary data to load (e.g. mri/brainmask.mgz).")
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
    parser.add_argument('--mask_dataset', action="store_true", default=False,
                        help="Turn on to create a dataset with masks instead of images - useful to create masks for data augmentation")
    parser.add_argument('--debug', action="store_true", default=False,
                        help="Only process 20 subjects for debugging")
    parser.add_argument('--center', type=str, default=None, help="CSV file with RAS coordinates of the center of the commisures [PC, AC]")
    parser.add_argument('--pad', action="store_true", default=False, help="Pad images at the edge of the volume for thick slices")
    parser.add_argument('--crop', action="store_true", default=False, help="Crop images to the center of the volume for thick slices")

    args = parser.parse_args()

    if args.aux_data is not None:
        args.aux_data = json.loads(args.aux_data)

    dataset_params = {"dataset_name": args.hdf5_name, "data_path": args.data_dir, "thickness": args.thickness,
                      "csv_file": args.csv_file, "pattern": args.pattern, "image_name": args.image_name,
                      "gt_name": args.gt_name, "gt_nocc": args.gt_nocc,
                      "max_weight": args.max_w, "edge_weight": args.edge_w,
                      "lut": args.lut, "combi": args.combi, "sag-mask": args.sag_mask,
                      "hires_weight": args.hires_w, "gm_mask": args.gm, "gradient": not args.no_grad,
                      "aux_data": args.aux_data, "debug": args.debug, "center": args.center, "pad": args.pad, "crop": args.crop}

    logging.setup_logging()

    if not args.mask_dataset:
        dataset_generator = H5pyDataset(params=dataset_params, processing=args.processing)
        dataset_generator.create_hdf5_dataset(plane=args.plane)
    else:
        dataset_generator = H5pyDatasetMasks(params=dataset_params)
        dataset_generator.create_hdf5_dataset()
