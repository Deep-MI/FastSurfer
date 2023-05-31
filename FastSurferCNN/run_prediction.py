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
import sys
import os
import copy
import argparse
from typing import Tuple, Union, Literal, Dict, Any, Optional, Iterator
from concurrent.futures import Executor

import numpy as np
import torch
import nibabel as nib

from FastSurferCNN.inference import Inference
from FastSurferCNN.utils import logging, parser_defaults
from FastSurferCNN.utils.checkpoint import get_checkpoints, VINN_AXI, VINN_COR, VINN_SAG
from FastSurferCNN.utils.load_config import load_config
from FastSurferCNN.utils.common import (
    find_device,
    assert_no_root,
    handle_cuda_memory_exception,
    SubjectList,
    SubjectDirectory,
    NoParallelExecutor,
    pipeline,
)
from FastSurferCNN.data_loader import data_utils as du, conform as conf
from FastSurferCNN.quick_qc import check_volume
import FastSurferCNN.reduce_to_aseg as rta

##
# Global Variables
##

LOGGER = logging.getLogger(__name__)


##
# Processing
##
def set_up_cfgs(cfg, args):
    cfg = load_config(cfg)
    cfg.OUT_LOG_DIR = args.out_dir if args.out_dir is not None else cfg.LOG_DIR
    cfg.OUT_LOG_NAME = "fastsurfer"
    cfg.TEST.BATCH_SIZE = args.batch_size

    cfg.MODEL.OUT_TENSOR_WIDTH = cfg.DATA.PADDED_SIZE
    cfg.MODEL.OUT_TENSOR_HEIGHT = cfg.DATA.PADDED_SIZE
    return cfg


def args2cfg(args: argparse.Namespace):
    """
    Extract the configuration objects from the arguments.
    """
    cfg_cor = set_up_cfgs(args.cfg_cor, args) if args.cfg_cor is not None else None
    cfg_sag = set_up_cfgs(args.cfg_sag, args) if args.cfg_sag is not None else None
    cfg_ax = set_up_cfgs(args.cfg_ax, args) if args.cfg_ax is not None else None
    cfg_fin = (
        cfg_cor if cfg_cor is not None else cfg_sag if cfg_sag is not None else cfg_ax
    )
    return cfg_fin, cfg_cor, cfg_sag, cfg_ax


def removesuffix(string: str, suffix: str) -> str:
    """Similar to string.removesuffix in PY3.9+, removes a suffix from a string."""
    import sys

    if sys.version_info.minor >= 9:
        # removesuffix is a Python3.9 feature
        return string.removesuffix(suffix)
    else:
        return (
            string[: -len(suffix)]
            if len(suffix) > 0 and string.endswith(suffix)
            else string
        )


##
# Input array preparation
##


class RunModelOnData:
    pred_name: str
    conf_name: str
    orig_name: str
    vox_size: Union[float, Literal["min"]]
    current_plane: str
    models: Dict[str, Inference]
    view_ops: Dict[str, Dict[str, Any]]
    conform_to_1mm_threshold: Optional[float]

    def __init__(self, args):
        self.pred_name = args.pred_name
        self.conf_name = args.conf_name
        self.orig_name = args.orig_name
        self._threads = getattr(args, "threads", 1)
        torch.set_num_threads(self._threads)
        self._async_io = getattr(args, "async_io", False)

        self.sf = 1.0

        device = find_device(args.device)

        if device.type == "cpu" and args.viewagg_device == "auto":
            self.viewagg_device = device
        else:
            # check, if GPU is big enough to run view agg on it
            # (this currently takes the memory of the passed device)
            self.viewagg_device = torch.device(
                find_device(
                    args.viewagg_device,
                    flag_name="viewagg_device",
                    min_memory=4 * (2**30),
                )
            )

        LOGGER.info(f"Running view aggregation on {self.viewagg_device}")

        try:
            self.lut = du.read_classes_from_lut(args.lut)
        except FileNotFoundError as e:
            raise ValueError(
                f"Could not find the ColorLUT in {args.lut}, please make sure the --lut argument is valid."
            )
        self.labels = self.lut["ID"].values
        self.torch_labels = torch.from_numpy(self.lut["ID"].values)
        self.names = ["SubjectName", "Average", "Subcortical", "Cortical"]
        self.cfg_fin, cfg_cor, cfg_sag, cfg_ax = args2cfg(args)
        # the order in this dictionary dictates the order in the view aggregation
        self.view_ops = {
            "coronal": {"cfg": cfg_cor, "ckpt": args.ckpt_cor},
            "sagittal": {"cfg": cfg_sag, "ckpt": args.ckpt_sag},
            "axial": {"cfg": cfg_ax, "ckpt": args.ckpt_ax},
        }
        self.num_classes = max(
            view["cfg"].MODEL.NUM_CLASSES for view in self.view_ops.values()
        )
        self.models = {}
        for plane, view in self.view_ops.items():
            if view["cfg"] is not None and view["ckpt"] is not None:
                self.models[plane] = Inference(
                    view["cfg"], ckpt=view["ckpt"], device=device, lut=self.lut
                )

        vox_size = args.vox_size
        if vox_size == "min":
            self.vox_size = "min"
        elif 0.0 < float(vox_size) <= 1.0:
            self.vox_size = float(vox_size)
        else:
            raise ValueError(
                f"Invalid value for vox_size, must be between 0 and 1 or 'min', was {vox_size}."
            )
        self.conform_to_1mm_threshold = args.conform_to_1mm_threshold

    @property
    def pool(self) -> Executor:
        if not hasattr(self, "_pool"):
            if not self._async_io:
                self._pool = NoParallelExecutor()
            else:
                from concurrent.futures import ThreadPoolExecutor

                self._pool = ThreadPoolExecutor(self._threads)
        return self._pool

    def __del__(self):
        if hasattr(self, "_pool"):
            # only wait on futures, if we specifically ask (see end of the script, so we do not wait if we encounter a
            # fail case)
            self._pool.shutdown(True)

    def conform_and_save_orig(
        self, subject: SubjectDirectory
    ) -> Tuple[nib.analyze.SpatialImage, np.ndarray]:
        orig, orig_data = du.load_image(subject.orig_name, "orig image")
        LOGGER.info(f"Successfully loaded image from {subject.orig_name}.")

        # Save input image to standard location, but only
        if subject.can_resolve_attribute("copy_orig_name"):
            self.pool.submit(self.save_img, subject.copy_orig_name, orig_data, orig)

        if not conf.is_conform(
            orig,
            conform_vox_size=self.vox_size,
            check_dtype=True,
            verbose=False,
            conform_to_1mm_threshold=self.conform_to_1mm_threshold,
        ):
            LOGGER.info("Conforming image")
            orig = conf.conform(
                orig,
                conform_vox_size=self.vox_size,
                conform_to_1mm_threshold=self.conform_to_1mm_threshold,
            )
            orig_data = np.asanyarray(orig.dataobj)

        # Save conformed input image
        if subject.can_resolve_attribute("conf_name"):
            self.pool.submit(
                self.save_img, subject.conf_name, orig_data, orig, dtype=np.uint8
            )
        else:
            raise RuntimeError(
                "Cannot resolve the name to the conformed image, please specify an absolute path."
            )

        return orig, orig_data

    def set_model(self, plane: str):
        self.current_plane = plane

    def get_prediction(
        self, image_name: str, orig_data: np.ndarray, zoom: Union[np.ndarray, tuple]
    ) -> np.ndarray:
        shape = orig_data.shape + (self.get_num_classes(),)
        kwargs = {
            "device": self.viewagg_device,
            "dtype": torch.float16,
            "requires_grad": False,
        }

        pred_prob = torch.zeros(shape, **kwargs)

        # inference and view aggregation
        for plane, model in self.models.items():
            LOGGER.info(f"Run {plane} prediction")
            self.set_model(plane)
            # pred_prob is updated inplace to conserve memory
            pred_prob = model.run(pred_prob, image_name, orig_data, zoom, out=pred_prob)

        # Get hard predictions
        pred_classes = torch.argmax(pred_prob, 3)
        del pred_prob
        # map to freesurfer label space
        pred_classes = du.map_label2aparc_aseg(pred_classes, self.labels)
        # return numpy array TODO: split_cortex_labels requires a numpy ndarray input, maybe we can also use Mapper here
        pred_classes = du.split_cortex_labels(pred_classes.cpu().numpy())
        return pred_classes

    def save_img(
        self,
        save_as: str,
        data: Union[np.ndarray, torch.Tensor],
        orig: nib.analyze.SpatialImage,
        dtype: Union[None, type] = None,
    ):
        """Saves the image."""
        # Create output directory if it does not already exist.
        if not os.path.exists(os.path.dirname(save_as)):
            LOGGER.info(
                f"Output image directory {os.path.basename(save_as)} does not exist. Creating it now..."
            )
            os.makedirs(os.path.dirname(save_as))

        np_data = data if isinstance(data, np.ndarray) else data.cpu().numpy()
        if dtype is not None:
            _header = orig.header.copy()
            _header.set_data_dtype(dtype)
        else:
            _header = orig.header
        r = du.save_image(_header, orig.affine, np_data, save_as, dtype=dtype)
        LOGGER.info(
            f"Successfully saved image {'asynchronously ' if self._async_io else ''}  as {save_as}."
        )
        return r

    def async_save_img(
        self,
        save_as: str,
        data: Union[np.ndarray, torch.Tensor],
        orig: nib.analyze.SpatialImage,
        dtype: Union[None, type] = None,
    ):
        """Saves the image asynchronously and returns a concurrent.futures.Future to track, when this finished."""
        return self.pool.submit(self.save_img, save_as, data, orig, dtype)

    def set_up_model_params(self, plane, cfg, ckpt):
        self.view_ops[plane]["cfg"] = cfg
        self.view_ops[plane]["ckpt"] = ckpt

    def get_num_classes(self) -> int:
        return self.num_classes

    def pipeline_conform_and_save_orig(
        self, subjects: SubjectList
    ) -> Iterator[Tuple[SubjectDirectory, Tuple[nib.analyze.SpatialImage, np.ndarray]]]:
        if not self._async_io:
            # do not pipeline, direct iteration and function call
            for subject in subjects:
                # yield subject and load orig
                yield subject, self.conform_and_save_orig(subject)
        else:
            # pipeline the same
            for data in pipeline(self.pool, self.conform_and_save_orig, subjects):
                yield data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation metrics")

    # 1. Options for input directories and filenames
    parser = parser_defaults.add_arguments(
        parser, ["t1", "sid", "in_dir", "tag", "csv_file", "lut", "remove_suffix"]
    )

    # 2. Options for output
    parser = parser_defaults.add_arguments(
        parser,
        [
            "asegdkt_segfile",
            "conformed_name",
            "brainmask_name",
            "aseg_name",
            "sd",
            "seg_log",
            "qc_log",
        ],
    )

    # 3. Checkpoint to load
    parser = parser_defaults.add_plane_flags(
        parser,
        "checkpoint",
        {"coronal": VINN_COR, "axial": VINN_AXI, "sagittal": VINN_SAG},
    )

    # 4. CFG-file with default options for network
    parser = parser_defaults.add_plane_flags(
        parser,
        "config",
        {
            "coronal": "FastSurferCNN/config/FastSurferVINN_coronal.yaml",
            "axial": "FastSurferCNN/config/FastSurferVINN_axial.yaml",
            "sagittal": "FastSurferCNN/config/FastSurferVINN_sagittal.yaml",
        },
    )

    # 5. technical parameters
    parser = parser_defaults.add_arguments(
        parser,
        [
            "vox_size",
            "conform_to_1mm_threshold",
            "device",
            "viewagg_device",
            "batch_size",
            "async_io",
            "threads",
            "allow_root",
        ],
    )

    args = parser.parse_args()

    # Warning if run as root user
    args.allow_root or assert_no_root()

    qc_file_handle = None
    if args.qc_log != "":
        try:
            qc_file_handle = open(args.qc_log, "w")
        except NotADirectoryError:
            LOGGER.warning(
                "The directory in the provided QC log file path does not exist!"
            )
            LOGGER.warning("The QC log file will not be saved.")

    # Set up logging
    from FastSurferCNN.utils.logging import setup_logging

    setup_logging(args.log_name)

    # Download checkpoints if they do not exist
    # see utils/checkpoint.py for default paths
    LOGGER.info("Checking or downloading default checkpoints ...")
    get_checkpoints(args.ckpt_ax, args.ckpt_cor, args.ckpt_sag)

    # Set Up Model
    eval = RunModelOnData(args)

    args.copy_orig_name = os.path.join("mri", "orig", "001.mgz")
    # Get all subjects of interest
    subjects = SubjectList(args, segfile="pred_name", copy_orig_name="copy_orig_name")
    subjects.make_subjects_dir()

    qc_failed_subject_count = 0

    iter_subjects = eval.pipeline_conform_and_save_orig(subjects)
    futures = []
    for subject, (orig_img, data_array) in iter_subjects:

        # Run model
        try:
            # The orig_t1_file is only used to populate verbose messages here
            pred_data = eval.get_prediction(
                subject.orig_name, data_array, orig_img.header.get_zooms()
            )
            futures.append(
                eval.async_save_img(
                    subject.segfile, pred_data, orig_img, dtype=np.int16
                )
            )

            # Create aseg and brainmask

            # There is a funny edge case in legacy FastSurfer 2.0, where the behavior is not well-defined, if orig_name
            # is an absolute path, but out_dir is not set. Then, we would create a sub-folder in the folder of orig_name
            # using the subject_id (passed by --sid or extracted from the orig_name) and use that as the subject folder.
            bm = None
            store_brainmask = subject.can_resolve_filename(args.brainmask_name)
            store_aseg = subject.can_resolve_filename(args.aseg_name)
            if store_brainmask or store_aseg:
                LOGGER.info("Creating brainmask based on segmentation...")
                bm = rta.create_mask(copy.deepcopy(pred_data), 5, 4)
            if store_brainmask:
                # get mask
                mask_name = subject.filename_in_subject_folder(args.brainmask_name)
                futures.append(
                    eval.async_save_img(mask_name, bm, orig_img, dtype=np.uint8)
                )
            else:
                LOGGER.info(
                    "Not saving the brainmask, because we could not figure out where to store it. Please "
                    "specify a subject id with {sid[flag]}, or an absolute brainmask path with "
                    "{brainmask_name[flag]}.".format(**subjects.flags)
                )

            if store_aseg:
                # reduce aparc to aseg and mask regions
                LOGGER.info("Creating aseg based on segmentation...")
                aseg = rta.reduce_to_aseg(pred_data)
                aseg[bm == 0] = 0
                aseg = rta.flip_wm_islands(aseg)
                aseg_name = subject.filename_in_subject_folder(args.aseg_name)
                # Change datatype to np.uint8, else mri_cc will fail!
                futures.append(
                    eval.async_save_img(aseg_name, aseg, orig_img, dtype=np.uint8)
                )
            else:
                LOGGER.info(
                    "Not saving the aseg file, because we could not figure out where to store it. Please "
                    "specify a subject id with {sid[flag]}, or an absolute aseg path with "
                    "{aseg_name[flag]}.".format(**subjects.flags)
                )

            # Run QC check
            LOGGER.info("Running volume-based QC check on segmentation...")
            seg_voxvol = np.product(orig_img.header.get_zooms())
            if not check_volume(pred_data, seg_voxvol):
                LOGGER.warning(
                    "Total segmentation volume is too small. Segmentation may be corrupted."
                )
                if qc_file_handle is not None:
                    qc_file_handle.write(subject.id + "\n")
                    qc_file_handle.flush()
                qc_failed_subject_count += 1
        except RuntimeError as e:
            if not handle_cuda_memory_exception(e):
                raise e

    if qc_file_handle is not None:
        qc_file_handle.close()

    # Batch case: report ratio of QC warnings
    if len(subjects) > 1:
        LOGGER.info(
            "Segmentations from {} out of {} processed cases failed the volume-based QC check.".format(
                qc_failed_subject_count, len(subjects)
            )
        )

    # wait for async processes to finish
    for f in futures:
        _ = f.result()

    sys.exit(0)
