
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
from typing import Tuple, Union

import numpy as np
import torch
import os
import nibabel as nib
import glob
import sys
import argparse

from inference import Inference
from utils.load_config import load_config
from utils.checkpoint import get_checkpoints, VINN_AXI, VINN_COR, VINN_SAG
from utils import logging as logging
from quick_qc import check_volume

import data_loader.data_utils as du
import data_loader.conform as conf

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


##
# Input array preparation
##
class RunModelOnData:

    pred_name: str
    conf_name: str
    orig_name: str
    gn_noise: bool
    hires: bool

    def __init__(self, args):
        self.pred_name = args.pred_name
        self.conf_name = args.conf_name
        self.orig_name = args.orig_name
        self.strip = args.strip

        self.sf = 1.0
        self.out_dir = self.set_and_create_outdir(args.out_dir)

        if args.run_viewagg_on == "gpu":
            # run view agg on the gpu (force)
            self.small_gpu = False

        elif args.run_viewagg_on == "check":
            # check, if GPU is big enough to run view agg on it
            # (this currently takes only the total memory into account, not the occupied on)
            total_gpu_memory = sum([torch.cuda.get_device_properties(i).__getattribute__("total_memory") for i in
                                    range(torch.cuda.device_count())]) if torch.cuda.is_available() else 0
            self.small_gpu = total_gpu_memory < 8000000000

        elif args.run_viewagg_on == "cpu":
            # run view agg on the cpu (either if the available GPU RAM is not big enough (<8 GB),
            # or if the model is anyhow run on cpu)
            self.small_gpu = True

        if self.small_gpu:
            LOGGER.info("Running view aggregation on CPU")
        else:
            LOGGER.info("Running view aggregation on GPU")

        self.lut = du.read_classes_from_lut(args.lut)
        self.labels = self.lut["ID"].values
        self.torch_labels = torch.from_numpy(self.lut["ID"].values)
        self.names = ["SubjectName", "Average", "Subcortical", "Cortical"]
        self.gn_noise = args.gn
        self.cfg_fin, cfg_cor, cfg_sag, cfg_ax = args2cfg(args)
        self.view_ops = {"coronal": {"cfg": cfg_cor, "ckpt": args.ckpt_cor},
                         "sagittal": {"cfg": cfg_sag, "ckpt": args.ckpt_sag},
                         "axial": {"cfg": cfg_ax, "ckpt": args.ckpt_ax}}
        self.ckpt_fin = args.ckpt_cor if args.ckpt_cor is not None else args.ckpt_sag if args.ckpt_sag is not None else args.ckpt_ax
        self.num_classes = self.cfg_fin.MODEL.NUM_CLASSES
        self.model = Inference(self.cfg_fin, device = args.device)
        self.device = self.model.get_device()
        self.dim = self.model.get_max_size()
        self.hires = args.hires

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
        self.save_img(self.input_img_name, orig_data, orig, seg=False)

        if not conf.is_conform(orig, conform_min=self.hires, check_dtype=True, verbose=False):
            LOGGER.info("Conforming image")
            orig = conf.conform(orig, conform_min=self.hires)
            orig_data = np.asanyarray(orig.dataobj)

        # Save conformed input image
        self.save_img(self.subject_conf_name, orig_data, orig, seg=False)
        return orig, orig_data

    def set_subject(self, subject: str):
        self.subject_name = os.path.basename(subject[:-len(self.strip)]) if self.strip else os.path.basename(subject)
        self.subject_conf_name = os.path.join(self.out_dir, self.subject_name, self.conf_name)
        self.input_img_name = os.path.join(self.out_dir, self.subject_name,
                                           os.path.dirname(self.conf_name), 'orig', '001.mgz')

    def get_subject_name(self) -> str:
        return self.subject_name

    def set_model(self, plane: str):
        self.model.set_model(self.view_ops[plane]["cfg"])
        self.model.load_checkpoint(self.view_ops[plane]["ckpt"])
        self.device = self.model.get_device()
        self.dim = self.model.get_max_size()

    @torch.no_grad()
    def run_model(self, out: torch.Tensor, orig_f: str, orig_data: np.ndarray, zooms: Union[np.ndarray, tuple]) -> torch.Tensor:
        # get prediction
        return self.model.run(orig_f, orig_data, zooms,
                              out, noise=self.gn_noise)

    def get_prediction(self, orig_f: str, orig_data: np.ndarray, zoom: Union[np.ndarray, tuple]) -> np.ndarray:
        shape = orig_data.shape + (self.get_num_classes(),)
        kwargs = {
            "device": "cpu" if self.small_gpu else self.device,
            "dtype": torch.float16,  # TODO add a flag to choose between single and half precision?
            "requires_grad": False
        }

        pred_prob = torch.zeros(shape, **kwargs)

        # coronal inference
        if self.view_ops["coronal"]["cfg"] is not None:
            LOGGER.info("Run coronal prediction")
            self.set_model("coronal")
            pred_prob = self.run_model(pred_prob, orig_f, orig_data, zoom)

        # axial inference
        if self.view_ops["axial"]["cfg"] is not None:
            LOGGER.info("Run axial view agg")
            self.set_model("axial")
            pred_prob = self.run_model(pred_prob, orig_f, orig_data, zoom)

        # sagittal inference
        if self.view_ops["sagittal"]["cfg"] is not None:
            LOGGER.info("Run sagittal view agg")
            self.set_model("sagittal")
            pred_prob = self.run_model(pred_prob, orig_f, orig_data, zoom)

        # Get hard predictions
        pred_classes = torch.argmax(pred_prob, 3)
        del pred_prob
        # map to freesurfer label space
        pred_classes = du.map_label2aparc_aseg(pred_classes, self.labels)
        # return numpy array TODO: split_cortex_labels requires a numpy ndarry input
        pred_classes = du.split_cortex_labels(pred_classes.cpu().numpy())
        return pred_classes

    @staticmethod
    def get_img(filename: Union[str, os.PathLike]) -> Tuple[nib.analyze.SpatialImage, np.ndarray]:
        img = nib.load(filename)
        data = np.asanyarray(img.dataobj)

        return img, data

    @staticmethod
    def save_img(save_as: str, data: np.ndarray, orig: nib.analyze.SpatialImage, seg=False):
        # Create output directory if it does not already exist.
        if not os.path.exists(os.path.dirname(save_as)):
            LOGGER.info("Output image directory does not exist. Creating it now...")
            os.makedirs(os.path.dirname(save_as))
        if not isinstance(data, np.ndarray):
            data = data.cpu().numpy()

        if seg:
            header = orig.header.copy()
            header.set_data_dtype(np.int16)
            du.save_image(header, orig.affine, data, save_as)
        else:
            du.save_image(orig.header, orig.affine, data, save_as)
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
                    "(--batch_size <int>) and/or view aggregation on the cpu (--run_viewagg_on 'cpu')."
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
    parser.add_argument('--t1', type=str, dest="orig_name", default='mri/orig.mgz',
                        help="Name of orig input (T1 full head input). Absolute path if single image else "
                             "common image name. Default: mri/orig.mgz")
    parser.add_argument('--strip', type=str, dest="strip", default='',
                        help="Optional: strip suffix from path definition of input file to yield correct subject name."
                             "(e.g. ses-x/anat/ for BIDS or mri/ for FreeSurfer input). Default: do not strip anything.")
    parser.add_argument("--in_dir", type=str, default=None,
                        help="Directory in which input volume(s) are located. "
                             "Optional, if full path is defined for --orig_name.")
    parser.add_argument('--t', '--tag', dest='search_tag', default="*",
                        help='Search tag to process only certain subjects. If a single image should be analyzed, '
                             'set the tag with its id. Default: processes all.')
    parser.add_argument('--csv_file', type=str, help="Csv-file with subjects to analyze (alternative to --tag",
                        default=None)
    parser.add_argument("--lut", type=str, help="Path and name of LUT to use.",
                        default=os.path.join(os.path.dirname(__file__), "config/FastSurfer_ColorLUT.tsv"))
    parser.add_argument("--gn", type=int, default=0,
                        help="How often to sample from gaussian and run inference on same sample with added noise on "
                             "scale factor.")

    # 2. Options for output
    parser.add_argument('--seg', type=str, dest='pred_name', default='mri/aparc.DKTatlas+aseg.deep.mgz',
                        help="Name of intermediate DL-based segmentation file (similar to aparc+aseg). "
                             "When using FastSurfer, this segmentation is already conformed, since inference "
                             "is always based on a conformed image. Absolute path if single image else common "
                             "image name. Default:mri/aparc.DKTatlas+aseg.deep.mgz")
    parser.add_argument('--conformed_name', type=str, dest='conf_name', default='mri/orig.mgz',
                        help="Name under which the conformed input image will be saved, in the same directory "
                             "as the segmentation (the input image is always conformed first, if it is not "
                             "already conformed). The original input image is saved in the output directory "
                             "as $id/mri/orig/001.mgz. Default: mri/orig.mgz.")
    parser.add_argument('--seg_log', type=str, dest='log_name', default="",
                        help="Absolute path to file in which run logs will be saved. If not set, logs will "
                             "not be saved.")
    parser.add_argument('--qc_log', type=str, dest='qc_log', default="",
                        help="Absolute path to file in which a list of subjects that failed QC check "
                             "(when processing multiple subjects) will be saved. "
                             "If not set, the file will not be saved.")
    parser.add_argument("--sd", type=str, default=None, dest="out_dir",
                        help="Directory in which evaluation results should be written. "
                             "Will be created if it does not exist. Optional if full path is defined for --pred_name.")
    parser.add_argument('--hires', action="store_true", default=False, dest='hires',
                        help="Switch on hires processing (no conforming to 1mm, but to smallest voxel size.")

    # 3. Checkpoint to load
    parser.add_argument('--ckpt_cor', type=str, help="coronal checkpoint to load",
                        default=os.path.join(os.path.dirname(__file__), VINN_COR))
    parser.add_argument('--ckpt_ax', type=str, help="axial checkpoint to load",
                        default=os.path.join(os.path.dirname(__file__), VINN_AXI))
    parser.add_argument('--ckpt_sag', type=str, help="sagittal checkpoint to load",
                        default=os.path.join(os.path.dirname(__file__), VINN_SAG))

    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference. Default=8")

    # 4. CFG-file with default options for network
    parser.add_argument("--cfg_cor", dest="cfg_cor", help="Path to the coronal config file",
                        default=os.path.join(os.path.dirname(__file__),
                                             "config/FastSurferVINN_coronal.yaml"), type=str)
    parser.add_argument("--cfg_ax", dest="cfg_ax", help="Path to the axial config file",
                        default=os.path.join(os.path.dirname(__file__),
                                             "config/FastSurferVINN_axial.yaml"), type=str)
    parser.add_argument("--cfg_sag", dest="cfg_sag", help="Path to the sagittal config file",
                        default=os.path.join(os.path.dirname(__file__),
                                             "config/FastSurferVINN_sagittal.yaml"), type=str)

    parser.add_argument('--device', default="auto",
                        help="select device to run inference on: cpu, or cuda (= Nvidia gpu) or "
                             "specify a certain gpu (e.g. cuda:1), default: auto")

    parser.add_argument('--run_viewagg_on', dest='run_viewagg_on', type=str,
                        default="check", choices=["gpu", "cpu", "check"],
                        help="Define where the view aggregation should be run on. \
                             By default, the program checks if you have enough memory \
                             to run the view aggregation on the gpu. The total memory \
                             is considered for this decision. If this fails, or \
                             you actively overwrote the check with setting \
                             > --run_viewagg_on cpu <, view agg is run on the cpu. \
                             Equivalently, if you define > --run_viewagg_on gpu <,\
                             view agg will be run on the gpu (no memory check will be done).")

    args = parser.parse_args()

    # Check input and output options
    if args.in_dir is None and args.csv_file is None and not os.path.isfile(args.orig_name):
        parser.print_help(sys.stderr)
        sys.exit('----------------------------\nERROR: Please specify data input directory or full path to input volume\n')

    if args.out_dir is None and not os.path.isabs(args.pred_name):
        parser.print_help(sys.stderr)
        sys.exit('----------------------------\nERROR: Please specify data output directory or absolute path to output volume'
                 '(can be same as input directory)\n')

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
        eval.set_subject(subject)
        orig_fn = args.orig_name if os.path.isfile(args.orig_name) else os.path.join(subject, args.orig_name)
        orig_img, data_array = eval.conform_and_save_orig(orig_fn)
        # try:
        #     orig_img, data_array = eval.conform_and_save_orig(orig_fn)
        # except FileNotFoundError:
        #     LOGGER.warning("Subject image file was not found! Skipping...")
        #     continue

        pred_name = args.pred_name if os.path.isabs(args.pred_name) else \
            os.path.join(eval.get_out_dir(), eval.get_subject_name(), args.pred_name)

        # Run model
        pred_data = eval.get_prediction(orig_fn, data_array, orig_img.header.get_zooms())
        eval.save_img(pred_name, pred_data, orig_img, seg=True)

        # Run QC check (subject list may later be written out)
        LOGGER.info("Running volume-based QC check on segmentation...")
        seg_voxvol = np.product(orig_img.header["delta"])
        if not check_volume(pred_data, seg_voxvol):
            LOGGER.warning("Total segmentation volume is too small. Segmentation may be corrupted.")
            if qc_file_handle is not None:
                qc_file_handle.write(subject.split('/')[-1]+"\n")
                qc_file_handle.flush()
            qc_failed_subject_count += 1

    if qc_file_handle is not None:
        qc_file_handle.close()

    # Single case: exit with error if qc fails. Batch case: report ratio of failures.
    if len(s_dirs) == 1:
        if qc_failed_subject_count:
            LOGGER.error("Single subject failed the volume-based QC check.")
            sys.exit(1)
    else:
        LOGGER.info("Segmentations from {} out of {} processed cases failed the volume-based QC check.".format(qc_failed_subject_count, len(s_dirs)))

    sys.exit(0)
