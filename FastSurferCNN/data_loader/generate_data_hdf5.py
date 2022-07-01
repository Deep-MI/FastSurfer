
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
from os import makedirs
from os.path import join
from collections import defaultdict
import numpy as np
import nibabel as nib
import h5py
import pandas as pd

from skimage.transform import rescale

from data_utils import transform_axial, transform_sagittal, \
    map_aparc_aseg2label, create_weight_mask,\
    get_thick_slices, filter_blank_slices_thick, bounding_box_slices, bounding_box_crop, bounding_box_pad


class H5pyDataset:

    def __init__(self, params, interpol=False, aparc=True, hippo=False):

        self.dataset_name = params["dataset_name"]
        self.data_path = params["data_path"]
        self.slice_thickness = params["thickness"]
        self.orig_name = params["image_name"]
        self.aparc_name = params["gt_name"]
        self.aparc_nocc = params["gt_nocc"]
        self.aparc = aparc
        self.hippo = hippo
        self.suffix = ""
        self.prefix = ".mgz"

        self.available_sizes = params["sizes"]
        self.crop_size = params["crop_size"]
        self.interpol = interpol
        self.max_weight = params["max_weight"]
        self.edge_weight = params["edge_weight"]
        self.hires_weight = params["hires_weight"]
        self.smooth_weight = params["smooth_weights"]
        self.gradient = params["gradient"]
        self.gm_mask = params["gm_mask"]

        if params["csv_file"] is not None:
            with open(params["csv_file"], "r") as s_dirs:
                self.subject_dirs = [line.strip() for line in s_dirs.readlines()]

        else:
            self.search_pattern = join(self.data_path, params["pattern"])
            self.subject_dirs = glob.glob(self.search_pattern)

        self.data_set_size = len(self.subject_dirs)


    def _load_volumes(self, subject_path, plane):
        # Load the orig and extract voxel spacing informatino (x, y, and z dim)
        print('Processing ground truth segmentation {}'.format(self.aparc_name))
        print("Subject pat", subject_path[1:len("mindboggle") + 1])
        if subject_path[1:len("mindboggle")+1] == "mindboggle":
            orig = nib.load(join(subject_path, "mindboggle_orig"+ self.suffix + ".mgz"))
            aseg = np.asarray(nib.load(join(subject_path, "mindboggle_gt" + self.suffix +".mgz")).get_fdata(), dtype=np.int16)

            aseg[aseg > 2000] = 42
            aseg[aseg > 1000] = 3
        else:
            orig = nib.load(join(subject_path, self.orig_name[:-len(self.prefix)] + self.suffix + self.orig_name[-len(self.prefix):]))
            # Load the segmentation ground truth
            aseg = np.asarray(nib.load(join(subject_path, self.aparc_name[:-len(self.prefix)] + self.suffix + self.aparc_name[-len(self.prefix):])).get_fdata(), dtype=np.int16)

        zoom = orig.header.get_zooms()
        orig = np.asarray(orig.get_fdata(), dtype=np.uint8)
        maxo = max(orig.shape)
        if orig.shape != (maxo, maxo, maxo):
            orig = self._pad_image(orig, maxo)
            aseg = self._pad_image(aseg, maxo)

        if self.aparc_nocc is not None and subject_path[1:len("mindboggle")+1] != "mindboggle":
            aseg_nocc = nib.load(join(subject_path, self.aparc_nocc[:-len(self.prefix)] + self.suffix + self.aparc_nocc[-len(self.prefix):]))
            aseg_nocc = np.asarray(aseg_nocc.get_fdata(), dtype=np.int16)
            if orig.shape != (maxo, maxo, maxo):
                aseg_nocc = self._pad_image(aseg_nocc, maxo)

        else:
            aseg_nocc = None

        if self.crop_size is not None and orig.shape[0] != self.crop_size:
            print("Cropping images from {0} to {1}x{1}x{1}".format(orig.shape, self.crop_size))
            # Slice images to unified size (e.g. 256x256x256)
            bb_box_slices_start, bb_box_slices_stop = bounding_box_slices(aseg, size_half=self.crop_size // 2)
            orig, aseg, aseg_nocc = bounding_box_crop(orig, aseg, bb_box_slices_start, bb_box_slices_stop,
                                                      weight_im=aseg_nocc)
            print("finished cropping")

        if plane == 'sagittal':
            #orig = transform_sagittal(orig)
            #aseg = transform_sagittal(aseg)
            #aseg_nocc = transform_sagittal(aseg_nocc)
            zoom = zoom[::-1][:2]

        elif plane == 'axial':
            #orig = transform_axial(orig)
            #aseg = transform_axial(aseg)
            #aseg_nocc = transform_axial(aseg_nocc)
            zoom = zoom[1:]

        else:
            zoom = zoom[:2]

        if self.interpol:
            sf = 0.5 if zoom == (1.0, 1.0) else 0.8
            # interpol image to resoultion of choice
            orig = rescale(orig, sf, anti_aliasing=False, order=1, mode="constant", preserve_range=True)
            aseg = rescale(aseg, sf, anti_aliasing=False, order=0, mode="constant", preserve_range=True).astype(int)
            zoom = tuple(zoom_x / sf for zoom_x in zoom)
        return orig, aseg, aseg_nocc, zoom


    def transform(self, plane, imgs):

        for i in range(len(imgs)):
            if plane == "sagittal":
                imgs[i] = transform_sagittal(imgs[i])
            elif plane == "axial":
                imgs[i] = transform_axial(imgs[i])
            else:
                pass
        return imgs

    def _pad_image(self, img, max_out):
        # Get correct size = max along shape
        h, w, d = img.shape
        print("Padding image from {0} to {1}x{1}x{1}".format(img.shape, max_out))
        padded_img = np.zeros((max_out, max_out, max_out), dtype=img.dtype)
        padded_img[0: h, 0: w, 0:d] = img
        return padded_img

    def create_hdf5_dataset(self, plane='axial', meta="", diagd={'Adult': "", 'Children': ""}, voi="Age_Bin", orig_res="Adult"):
        data_per_size = defaultdict(lambda: defaultdict(list))
        start_d = time.time()
        # Load meta information
        print("Metas", meta)
        if meta != "":
            separator = "," if meta[-3] == "c" else "\t"
            metaf = pd.read_csv(meta, sep=separator)
            print("Meta", metaf.columns)

        for idx, current_subject in enumerate(self.subject_dirs):
            # get diagnosis for current subject
            if meta != "":
                try:
                    diag = metaf.loc[metaf["ID"] == current_subject.split("/")[-1], voi].iloc[0]
                except IndexError:
                    diag = metaf.loc[metaf["ID"] == current_subject.split("/")[-2], voi].iloc[0]
                self.suffix = diagd[diag.lstrip()]
                if self.suffix == "_XXmm":
                    continue
                if self.suffix == "_10mm" and diag.lstrip() == orig_res:  # use original image (only if DiagD = HC or Bottom40 or Adult!
                    self.suffix = ""
            else:
                self.suffix = ""

            try:
                start = time.time()

                print("Volume Nr: {} Processing MRI Data from {}/{}".format(idx+1, current_subject, self.orig_name))

                orig, aseg, aseg_nocc, zoom = self._load_volumes(current_subject, plane)
                size, _, _ = orig.shape
                #assert size in self.available_sizes, f"The input size {size} is not expected, available size are {self.available_sizes}"
                mapped_aseg, mapped_aseg_sag = map_aparc_aseg2label(aseg, aseg_nocc, aparc=self.aparc, hippo=self.hippo)

                if plane == 'sagittal':
                    mapped_aseg = mapped_aseg_sag
                    weights = create_weight_mask(mapped_aseg, max_weight=self.max_weight, ctx_thresh=19,
                                                 max_edge_weight=self.edge_weight, max_hires_weight=self.hires_weight,
                                                 mean_filter=self.smooth_weight, cortex_mask=self.gm_mask,
                                                 gradient=self.gradient)

                else:
                    weights = create_weight_mask(mapped_aseg, max_weight=self.max_weight, ctx_thresh=33,
                                                 max_edge_weight=self.edge_weight, max_hires_weight=self.hires_weight,
                                                 mean_filter=self.smooth_weight, cortex_mask=self.gm_mask,
                                                 gradient=self.gradient)

                print("Created weights with max_w {}, gradient {},"
                      " edge_w {}, hires_w {}, gm_mask {}, mean filter {}".format(self.max_weight, self.gradient, self.edge_weight,
                                                                      self.hires_weight, self.gm_mask, self.smooth_weight))

                # transform volumes to correct shape
                orig, mapped_aseg, weights = self.transform(plane, [orig, mapped_aseg, weights])

                # Create Thick Slices, filter out blanks
                orig_thick = get_thick_slices(orig, self.slice_thickness)

                orig, mapped_aseg, weights = filter_blank_slices_thick(orig_thick, mapped_aseg, weights, threshold=2)

                num_batch = orig.shape[2]
                orig = np.transpose(orig, (2, 0, 1, 3))
                print(orig.shape)
                mapped_aseg = np.transpose(mapped_aseg, (2, 0, 1))
                weights = np.transpose(weights, (2, 0, 1))

                data_per_size[f'{size}']['orig'].extend(orig)
                data_per_size[f'{size}']['aseg'].extend(mapped_aseg)
                data_per_size[f'{size}']['weight'].extend(weights)
                data_per_size[f'{size}']['zoom'].extend((zoom,) * num_batch)
                sub_name = current_subject.split("/")[-1]
                data_per_size[f'{size}']['subject'].append(sub_name.encode("ascii", "ignore"))

            #else:
            except Exception as e:
                print("Volume: {} Failed Reading Data. Error: {}".format(idx, e))
                continue

        for key, data_dict in data_per_size.items():
            data_per_size[key]['orig'] = np.asarray(data_dict['orig'], dtype=np.uint8)
            data_per_size[key]['aseg'] = np.asarray(data_dict['aseg'], dtype=np.uint8)
            data_per_size[key]['weight'] = np.asarray(data_dict['weight'], dtype=np.float)

        with h5py.File(self.dataset_name, "w") as hf:
            dt = h5py.special_dtype(vlen=str)
            for key, data_dict in data_per_size.items():
                group = hf.create_group(f"{key}")
                group.create_dataset("orig_dataset", data=data_dict['orig'])#, compression='lzf')
                group.create_dataset("aseg_dataset", data=data_dict['aseg'])#, compression='lzf')
                group.create_dataset("weight_dataset", data=data_dict['weight'])#, compression='lzf')
                group.create_dataset("zoom_dataset", data=data_dict['zoom'])
                group.create_dataset("subject", data=data_dict['subject'], dtype=dt)#, compression='lzf')

        end_d = time.time() - start_d
        print("Successfully written {} in {:.3f} seconds.".format(self.dataset_name, end_d))


if __name__ == '__main__':
    import argparse

    # Training settings
    parser = argparse.ArgumentParser(description='HDF5-Creation')

    parser.add_argument('--hdf5_name', type=str, default="../data/Multires_coronal.hdf5",
                        help='path and name of hdf5-data_loader (default: ../data/Multires_coronal.hdf5)')
    parser.add_argument('--plane', type=str, default="axial", choices=["axial", "coronal", "sagittal"],
                        help="Which plane to put into file (axial (default), coronal or sagittal)")
    parser.add_argument('--data_dir', type=str, default="/testsuite", help="Directory with images to load")
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
    parser.add_argument('--crop_size', type=int, default=None,
                        help="Unified size images should be cropped to (e.g. 256). Default=None (=no cropping)")
    parser.add_argument('--max_w', type=int, default=5,
                        help="Overall max weight for any voxel in weight mask. Default=5")
    parser.add_argument('--edge_w', type=int, default=5, help="Weight for edges in weight mask. Default=5")
    parser.add_argument('--hires_w', type=int, default=None,
                        help="Weight for hires elements (sulci, WM strands, cortex border) in weight mask. Default=None")
    parser.add_argument('--smooth', action='store_true', default=False,
                        help="Turn on to smooth final weight mask with mean filter")
    parser.add_argument('--no_grad', action='store_true', default=False,
                        help="Turn on to only use median weight frequency (no gradient)")
    parser.add_argument('--gm', action="store_true", default=False,
                        help="Turn on to add cortex mask for hires-processing.")
    parser.add_argument('--interpol', action="store_true", default=False, help="Interpolate image to lower resolution.")
    parser.add_argument('--aseg', action="store_true", default=False, help="Use aseg instead of aparc prediction")
    parser.add_argument('--hippo', action="store_true", default=False, help="Use hippo instead of aparc prediction")
    parser.add_argument('--ad_suffix', type=str, default="", help="suffix for AD/Top40 cases.")
    parser.add_argument('--hc_suffix', type=str, default="", help="suffix for HC/Bottom40 cases.")
    parser.add_argument('--meta', type=str, default="", help="Meta file with AD(Top)/HC(Bottom) and ID information.")
    parser.add_argument('--voi', type=str, default="Age_Bin",
                        help="Variable of interest to merge meta file on for suffix selection. Default=Age_Bin")
    parser.add_argument('--or', dest='orig_res', type=str, default="Adult",
                        help="Variable in voi which has original resolution. Default=Adult")
    parser.add_argument('--ir', dest='int_res', type=str, default="Children",
                        help="Variable in voi which has lower/differing resolution. Default=Children")
    parser.add_argument('--sizes', nargs='+', type=int, default=256)


    args = parser.parse_args()

    dataset_params = {"dataset_name": args.hdf5_name, "data_path": args.data_dir, "thickness": args.thickness,
                      "csv_file": args.csv_file, "pattern": args.pattern, "image_name": args.image_name,
                      "gt_name": args.gt_name, "gt_nocc": args.gt_nocc, "sizes": args.sizes,
                      "crop_size": args.crop_size, "max_weight": args.max_w, "edge_weight": args.edge_w,
                      "hires_weight": args.hires_w, "smooth_weights": args.smooth, "gm_mask": args.gm,
                      "gradient": not args.no_grad}
    aparc_flag = not args.aseg
    dataset_generator = H5pyDataset(params=dataset_params, interpol=args.interpol, aparc=aparc_flag, hippo=args.hippo)
    dataset_generator.create_hdf5_dataset(plane=args.plane, meta=args.meta,
                                   diagd={args.int_res: args.ad_suffix, args.orig_res: args.hc_suffix},
                                          voi=args.voi, orig_res=args.orig_res)

