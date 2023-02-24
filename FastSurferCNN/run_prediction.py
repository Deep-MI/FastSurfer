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
from typing import Tuple, Union, Literal, Dict, Any, Optional
import glob
import sys
import argparse
import os
import copy

import numpy as np
import torch
import nibabel as nib

from FastSurferCNN.inference import Inference
from FastSurferCNN.utils import logging, parser_defaults
from FastSurferCNN.utils.checkpoint import get_checkpoints, VINN_AXI, VINN_COR, VINN_SAG
from FastSurferCNN.utils.load_config import load_config
from FastSurferCNN.utils.misc import find_device
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
    cfg_fin = cfg_cor if cfg_cor is not None else cfg_sag if cfg_sag is not None else cfg_ax
    return cfg_fin, cfg_cor, cfg_sag, cfg_ax


def removesuffix(string, suffix):
    import sys
    if sys.version_info.minor >= 9:
        # removesuffix is a Python3.9 feature
        return string.removesuffix(suffix)
    else:
        return string[:-len(suffix)] if string.endswith(suffix) else string


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
        self.remove_suffix = args.remove_suffix

        self.sf = 1.0
        self.out_dir = self.set_and_create_outdir(args.out_dir)

        device = find_device(args.device)

        if args.viewagg_device == "auto":
            # check, if GPU is big enough to run view agg on it
            # (this currently takes the memory of the passed device)
            if device.type == "cuda" and torch.cuda.is_available():  # TODO update the automatic device selection
                dev_num = torch.cuda.current_device() if device.index is None else device.index
                total_gpu_memory = torch.cuda.get_device_properties(dev_num).__getattribute__("total_memory")
                # TODO this rule here should include the batch_size ?!
                self.viewagg_device = device if total_gpu_memory > 4000000000 else "cpu"
            else:
                self.viewagg_device = "cpu"

        else:
            try:
                self.viewagg_device = torch.device(args.viewagg_device)
            except:
                LOGGER.exception(f"Invalid device {args.viewagg_device}")
                raise
            # run view agg on the cpu (either if the available GPU RAM is not big enough (<8 GB),
            # or if the model is anyhow run on cpu)

        LOGGER.info(f"Running view aggregation on {self.viewagg_device}")

        try:
            self.lut = du.read_classes_from_lut(args.lut)
        except FileNotFoundError as e:
            raise ValueError(
                f"Could not find the ColorLUT in {args.lut}, please make sure the --lut argument is valid.")
        self.labels = self.lut["ID"].values
        self.torch_labels = torch.from_numpy(self.lut["ID"].values)
        self.names = ["SubjectName", "Average", "Subcortical", "Cortical"]
        self.cfg_fin, cfg_cor, cfg_sag, cfg_ax = args2cfg(args)
        # the order in this dictionary dictates the order in the view aggregation
        self.view_ops = {"coronal": {"cfg": cfg_cor, "ckpt": args.ckpt_cor},
                         "sagittal": {"cfg": cfg_sag, "ckpt": args.ckpt_sag},
                         "axial": {"cfg": cfg_ax, "ckpt": args.ckpt_ax}}
        self.num_classes = max(view["cfg"].MODEL.NUM_CLASSES for view in self.view_ops.values())
        self.models = {}
        for plane, view in self.view_ops.items():
            if view["cfg"] is not None and view["ckpt"] is not None:
                self.models[plane] = Inference(view["cfg"], ckpt=view["ckpt"], device=device)

        vox_size = args.vox_size
        if vox_size == "min":
            self.vox_size = "min"
        elif 0. < float(vox_size) <= 1.:
            self.vox_size = float(vox_size)
        else:
            raise ValueError(f"Invalid value for vox_size, must be between 0 and 1 or 'min', was {vox_size}.")
        self.conform_to_1mm_threshold = args.conform_to_1mm_threshold

    def set_and_create_outdir(self, out_dir: str) -> str:
        if os.path.isabs(self.pred_name):
            # Full path defined for input image, extract out_dir from there
            tmp = os.path.dirname(self.pred_name)

            # remove potential subject/mri doubling in outdir name
            out_dir = tmp if os.path.basename(tmp) != "mri" else os.path.dirname(os.path.dirname(tmp))
        LOGGER.info("Output will be stored in: {}".format(out_dir))

        if not os.path.exists(out_dir):
            LOGGER.info("Output directory does not exist. Creating it now...")
            os.makedirs(out_dir)
        return out_dir

    def get_out_dir(self) -> str:
        return self.out_dir

    def conform_and_save_orig(self, orig_str: str) -> Tuple[nib.analyze.SpatialImage, np.ndarray]:
        orig, orig_data = self.get_img(orig_str)

        # Save input image to standard location
        self.save_img(self.input_img_name, orig_data, orig)

        if not conf.is_conform(orig, conform_vox_size=self.vox_size, check_dtype=True, verbose=False,
                               conform_to_1mm_threshold=self.conform_to_1mm_threshold):
            LOGGER.info("Conforming image")
            orig = conf.conform(orig,
                                conform_vox_size=self.vox_size, conform_to_1mm_threshold=self.conform_to_1mm_threshold)
            orig_data = np.asanyarray(orig.dataobj)

        # Save conformed input image
        self.save_img(self.subject_conf_name, orig_data, orig, dtype=np.uint8)

        return orig, orig_data

    def set_subject(self, subject: str, sid: Union[str, None]):
        self.subject_name = os.path.basename(removesuffix(subject, self.remove_suffix)) if sid is None else sid
        self.subject_conf_name = os.path.join(self.out_dir, self.subject_name, self.conf_name)
        self.input_img_name = os.path.join(self.out_dir, self.subject_name,
                                           os.path.dirname(self.conf_name), 'orig', '001.mgz')

    def get_subject_name(self) -> str:
        return self.subject_name

    def set_model(self, plane: str):
        self.current_plane = plane

    def get_prediction(self, orig_f: str, orig_data: np.ndarray, zoom: Union[np.ndarray, tuple]) -> np.ndarray:
        shape = orig_data.shape + (self.get_num_classes(),)
        kwargs = {
            "device": self.viewagg_device,
            "dtype": torch.float16,
            "requires_grad": False
        }

        pred_prob = torch.zeros(shape, **kwargs)

        # inference and view aggregation
        for plane, model in self.models.items():
            LOGGER.info(f"Run {plane} prediction")
            self.set_model(plane)
            # pred_prob is updated inplace to conserve memory
            pred_prob = model.run(pred_prob, orig_f, orig_data, zoom, out=pred_prob)

        # Get hard predictions
        pred_classes = torch.argmax(pred_prob, 3)
        del pred_prob
        # map to freesurfer label space
        pred_classes = du.map_label2aparc_aseg(pred_classes, self.labels)
        # return numpy array TODO: split_cortex_labels requires a numpy ndarray input
        pred_classes = du.split_cortex_labels(pred_classes.cpu().numpy())
        return pred_classes

    @staticmethod
    def get_img(filename: Union[str, os.PathLike]) -> Tuple[nib.analyze.SpatialImage, np.ndarray]:
        img = nib.load(filename)
        data = np.asanyarray(img.dataobj)

        return img, data

    @staticmethod
    def save_img(save_as: str, data: Union[np.ndarray, torch.Tensor], orig: nib.analyze.SpatialImage,
                 dtype: Union[None, type] = None):
        # Create output directory if it does not already exist.
        if not os.path.exists(os.path.dirname(save_as)):
            LOGGER.info("Output image directory does not exist. Creating it now...")
            os.makedirs(os.path.dirname(save_as))
        np_data = data if isinstance(data, np.ndarray) else data.cpu().numpy()

        if dtype is not None:
            header = orig.header.copy()
            header.set_data_dtype(dtype)
        else:
            header = orig.header

        du.save_image(header, orig.affine, np_data, save_as, dtype=dtype)
        LOGGER.info("Successfully saved image as {}".format(save_as))

    def set_up_model_params(self, plane, cfg, ckpt):
        self.view_ops[plane]["cfg"] = cfg
        self.view_ops[plane]["ckpt"] = ckpt

    def get_num_classes(self) -> int:
        return self.num_classes


def handle_cuda_memory_exception(exception: RuntimeError, exit_on_out_of_memory: bool = True) -> bool:
    if not isinstance(exception, RuntimeError):
        return False
    message = exception.args[0]
    if message.startswith("CUDA out of memory. "):
        LOGGER.critical("ERROR - INSUFFICIENT GPU MEMORY")
        LOGGER.info("The memory requirements exceeds the available GPU memory, try using a smaller batch size "
                    "(--batch_size <int>) and/or view aggregation on the cpu (--viewagg_device 'cpu')."
                    "Note: View Aggregation on the GPU is particularly memory-hungry at approx. 5 GB for standard "
                    "256x256x256 images.")
        memory_message = message[message.find("(") + 1:message.find(")")]
        LOGGER.info(f"Using {memory_message}.")
        if exit_on_out_of_memory:
            sys.exit("----------------------------\nERROR: INSUFFICIENT GPU MEMORY\n")
        else:
            return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation metrics')

    # 1. Options for input directories and filenames
    parser = parser_defaults.add_arguments(parser, ["t1", "sid", "in_dir", "tag", "csv_file", "lut", "remove_suffix"])

    # 2. Options for output
    parser = parser_defaults.add_arguments(parser, ["aparc_aseg_segfile", "conformed_name", "brainmask_name",
                                                    "aseg_name", "sd", "seg_log", "qc_log"])

    # 3. Checkpoint to load
    parser = parser_defaults.add_plane_flags(parser, "checkpoint",
                                             {"coronal": VINN_COR, "axial": VINN_AXI, "sagittal": VINN_SAG})

    # 4. CFG-file with default options for network
    parser = parser_defaults.add_plane_flags(parser, "config",
                                             {"coronal": "FastSurferCNN/config/FastSurferVINN_coronal.yaml",
                                              "axial": "FastSurferCNN/config/FastSurferVINN_axial.yaml",
                                              "sagittal": "FastSurferCNN/config/FastSurferVINN_sagittal.yaml"})

    # 5. technical parameters
    parser = parser_defaults.add_arguments(parser, ["vox_size", "conform_to_1mm_threshold", "device", "viewagg_device",
                                                    "batch_size", "allow_root"])

    args = parser.parse_args()

    # Warning if run as root user
    if not args.allow_root and os.name == 'posix' and os.getuid() == 0:
        sys.exit(
            """----------------------------
            ERROR: You are trying to run 'run_prediction.py' as root. We advice to avoid running 
            FastSurfer as root, because it will lead to files and folders created as root.
            If you are running FastSurfer in a docker container, you can specify the user with 
            '-u $(id -u):$(id -g)' (see https://docs.docker.com/engine/reference/run/#user).
            If you want to force running as root, you may pass --allow_root to run_prediction.py.
            """)

    # Check input and output options
    if args.in_dir is None and args.csv_file is None and not os.path.isfile(args.orig_name):
        parser.print_help(sys.stderr)
        sys.exit(
            '----------------------------\nERROR: Please specify data input directory or full path to input volume\n')

    if args.out_dir is None and not os.path.isabs(args.pred_name):
        parser.print_help(sys.stderr)
        sys.exit(
            '----------------------------\nERROR: Please specify data output directory or absolute path to output volume'
            ' (can be same as input directory)\n')

    qc_file_handle = None
    if args.qc_log != "":
        try:
            qc_file_handle = open(args.qc_log, 'w')
        except NotADirectoryError:
            LOGGER.warning("The directory in the provided QC log file path does not exist!")
            LOGGER.warning("The QC log file will not be saved.")

    # Set up logging
    from utils.logging import setup_logging

    setup_logging(args.log_name)

    # Download checkpoints if they do not exist
    # see utils/checkpoint.py for default paths
    LOGGER.info("Checking or downloading default checkpoints ...")
    get_checkpoints(args.ckpt_ax, args.ckpt_cor, args.ckpt_sag)

    # Set Up Model
    eval = RunModelOnData(args)

    # Get all subjects of interest
    if args.csv_file is not None:
        with open(args.csv_file, "r") as s_dirs:
            s_dirs = [line.strip() for line in s_dirs.readlines()]
        LOGGER.info("Analyzing all {} subjects from csv_file {}".format(len(s_dirs), args.csv_file))

    elif args.in_dir is not None:
        s_dirs = glob.glob(os.path.join(args.in_dir, args.search_tag))
        LOGGER.info("Analyzing all {} subjects from in_dir {}".format(len(s_dirs), args.in_dir))

    else:
        s_dirs = [os.path.dirname(args.orig_name)]
        LOGGER.info("Analyzing single subject {}".format(args.orig_name))

    qc_failed_subject_count = 0

    for subject in s_dirs:
        # Set subject and load orig
        eval.set_subject(subject, args.sid)
        orig_fn = args.orig_name if os.path.isfile(args.orig_name) else os.path.join(subject, args.orig_name)
        orig_img, data_array = eval.conform_and_save_orig(orig_fn)

        # Set prediction name
        out_dir, sbj_name = eval.get_out_dir(), eval.get_subject_name()
        pred_name = args.pred_name if os.path.isabs(args.pred_name) else \
            os.path.join(out_dir, sbj_name, args.pred_name)

        # Run model
        try:
            pred_data = eval.get_prediction(orig_fn, data_array, orig_img.header.get_zooms())
            eval.save_img(pred_name, pred_data, orig_img, dtype=np.int16)

            # Create aseg and brainmask
            # Change datatype to np.uint8, else mri_cc will fail!

            # get mask
            LOGGER.info("Creating brainmask based on segmentation...")
            bm = rta.create_mask(copy.deepcopy(pred_data), 5, 4)
            mask_name = os.path.join(out_dir, sbj_name, args.brainmask_name)
            eval.save_img(mask_name, bm, orig_img, dtype=np.uint8)

            # reduce aparc to aseg and mask regions
            LOGGER.info("Creating aseg based on segmentation...")
            aseg = rta.reduce_to_aseg(pred_data)
            aseg[bm == 0] = 0
            aseg = rta.flip_wm_islands(aseg)
            aseg_name = os.path.join(out_dir, sbj_name, args.aseg_name)
            eval.save_img(aseg_name, aseg, orig_img, dtype=np.uint8)

            # Run QC check
            LOGGER.info("Running volume-based QC check on segmentation...")
            seg_voxvol = np.product(orig_img.header.get_zooms())
            if not check_volume(pred_data, seg_voxvol):
                LOGGER.warning("Total segmentation volume is too small. Segmentation may be corrupted.")
                if qc_file_handle is not None:
                    qc_file_handle.write(subject.split('/')[-1] + "\n")
                    qc_file_handle.flush()
                qc_failed_subject_count += 1
        except RuntimeError as e:
            if not handle_cuda_memory_exception(e):
                raise e

    if qc_file_handle is not None:
        qc_file_handle.close()

    # Batch case: report ratio of QC warnings
    if len(s_dirs) > 1:
        LOGGER.info("Segmentations from {} out of {} processed cases failed the volume-based QC check.".format(
            qc_failed_subject_count, len(s_dirs)))

    sys.exit(0)
