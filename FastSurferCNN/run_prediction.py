
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
import numpy as np
import torch
import os
import logging
import nibabel as nib
import time

from eval import Inference
from utils.load_config import load_config
import sys

import data_loader.data_utils as du
import data_loader.conform as conf

##
# Global Variables
##
LOGGER = logging.getLogger("eval")
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(stream=sys.stdout))


##
# Processing
##
def set_up_cfgs(cfg, args):
    cfg = load_config(cfg)
    cfg.OUT_LOG_DIR = args.out_dir if args.out_dir is not None else cfg.LOG_DIR
    cfg.OUT_LOG_NAME = args.model_name
    cfg.TEST.BATCH_SIZE = args.batch_size

    if args.outdims != 0:
        cfg.DATA.PADDED_SIZE = args.outdims
    out_dims = round(args.base_dim * args.scale_out) if args.scale_out is not None else cfg.DATA.PADDED_SIZE
    cfg.MODEL.OUT_TENSOR_WIDTH = out_dims if out_dims > cfg.DATA.PADDED_SIZE else cfg.DATA.PADDED_SIZE
    cfg.MODEL.OUT_TENSOR_HEIGHT = out_dims if out_dims > cfg.DATA.PADDED_SIZE else cfg.DATA.PADDED_SIZE
    return cfg


##
# Input array preparation
##
class RunModelOnData:

    def __init__(self, args):
        self.current_subject = ""
        self.subject_name = ""
        self.gt, self.gt_data, self.orig, self.orig_data = "", "", "", ""
        self.sf = 1.0
        self.model_name = args.model_name
        self.s = ""
        self.out_dir = args.out_dir
        self.orig_filename = os.path.join(self.current_subject, args.orig_name)

        if args.gt_dir is None:
            self.gt_filename = os.path.join(self.current_subject, args.gt_name)
        else:
            self.gt_filename = os.path.join(args.gt_dir, self.subject_name, args.gt_name)

        if args.out_dir is not None:
            self.pred_name = os.path.join(args.out_dir, self.subject_name, args.pred_name)
        else:
            self.pred_name = os.path.join(self.current_subject, args.pred_name)

        self.lut = du.read_classes_from_lut(args.lut)
        self.num_classes = len(self.lut["LabelName"])-1
        self.class_names = self.lut["LabelName"][1:]
        self.labels = self.lut["ID"][1:]
        self.torch_labels = torch.from_numpy(self.lut["ID"].values)
        self.names = ["SubjectName", "Average", "Subcortical", "Cortical"]
        mask = np.ones(self.num_classes, dtype=bool)
        mask[args.substruct] = False
        self.range_sub = mask
        self.gn_noise = args.gn
        self.view_ops, self.cfg_fin, self.ckpt_fin = self.set_view_ops(args)
        self.ckpt_fin = args.ckpt_cor if args.ckpt_cor is not None else args.ckpt_sag if args.ckpt_sag is not None else args.ckpt_ax
        self.model = Inference(self.cfg_fin, self.ckpt_fin)
        self.device = self.model.get_device()
        self.dim = self.model.get_max_size()

    def set_view_ops(self, args):
        cfg_cor = set_up_cfgs(args.cfg_cor, args) if args.cfg_cor is not None else None
        cfg_sag = set_up_cfgs(args.cfg_sag, args) if args.cfg_sag is not None else None
        cfg_ax = set_up_cfgs(args.cfg_ax, args) if args.cfg_ax is not None else None
        cfg_fin = cfg_cor if cfg_cor is not None else cfg_sag if cfg_sag is not None else cfg_ax
        ckpt_fin = args.ckpt_cor if args.ckpt_cor is not None else args.ckpt_sag if args.ckpt_sag is not None else args.ckpt_ax
        return {"coronal": {"cfg": cfg_cor,
                     "ckpt": args.ckpt_cor},
         "sagittal": {"cfg": cfg_sag,
                      "ckpt": args.ckpt_sag},
         "axial": {"cfg": cfg_ax,
                   "ckpt": args.ckpt_ax}}, cfg_fin, ckpt_fin

    def set_orig(self, orig_str):
        self.orig, self.orig_data = self.get_img(orig_str)
        # check and conform image

        if not conf.is_conform(self.orig, check_dtype=True):
            self.orig = conf.conform(self.orig)
            self.orig_data = np.asanyarray(self.orig.dataobj)

    def set_gt(self, gt_str):
        self.gt, self.gt_data = self.get_img(gt_str)

    def set_subject(self, subject):
        self.subject_name = subject

    def set_model(self, plane):
        self.model.set_model(self.view_ops[plane]["cfg"])
        self.model.load_checkpoint(self.view_ops[plane]["ckpt"])
        self.device = self.model.get_device()
        self.dim = self.model.get_max_size()

    def run_model(self, pred_prob):
        # get prediction
        pred_prob = self.model.run(self.orig_filename, self.orig_data, self.orig.header.get_zooms(),
                                   pred_prob, noise=self.gn_noise)

        # Post processing
        h, w, d = self.gt_data.shape
        pred_prob = pred_prob[0:h, 0:w, 0:d, :]

        return pred_prob

    def get_gt(self):
        return self.gt, self.gt_data

    def get_orig(self):
        return self.orig, self.orig_data

    def get_prediction(self):
        oh, ow, od = self.orig.shape

        # coronal inference
        if self.view_ops["coronal"]["cfg"] is not None:
            if self.view_ops["axial"]["cfg"] is not None:
                self.set_model("coronal")
            pred_prob = torch.zeros((self.dim, self.dim, od, self.model.get_num_classes()),
                                    dtype=torch.float).to(self.device)
            pred_prob = self.run_model(pred_prob)

        # axial inference
        if self.view_ops["axial"]["cfg"] is not None:
            LOGGER.info("Run axial view agg")
            if self.view_ops["coronal"]["cfg"] is not None:
                self.set_model("axial")
                ax_prob = torch.zeros((self.dim, ow, self.dim, self.model.get_num_classes()),
                                      dtype=torch.float).to(self.device)
                pred_prob += self.run_model(ax_prob)
                del ax_prob
            else:
                pred_prob = torch.zeros((self.dim, ow, self.dim, self.model.get_num_classes()),
                                      dtype=torch.float).to(self.device)
                pred_prob = self.run_model(pred_prob)

        # sagittal inference
        if self.view_ops["sagittal"]["cfg"] is not None:
            LOGGER.info("Run sagittal view agg")
            if self.view_ops["coronal"]["cfg"] is not None:
                sag_prob = torch.zeros((oh, self.dim, self.dim, self.model.get_num_classes()),
                                       dtype=torch.float).to(self.device)
                self.set_model("sagittal")
                pred_prob += self.run_model(sag_prob)
                del sag_prob
            else:
                pred_prob = torch.zeros((oh, self.dim, self.dim, self.model.get_num_classes()),
                                       dtype=torch.float).to(self.device)
                pred_prob += self.run_model(pred_prob)

        # Get hard predictions and map to freesurfer label space
        _, pred_prob = torch.max(pred_prob, 3)
        pred_prob = du.map_label2aparc_aseg(pred_prob, self.torch_labels)
        pred_prob = du.split_cortex_labels(pred_prob.cpu().numpy())
        # return numpy array
        return pred_prob

    @staticmethod
    def get_img(filename):
        img = nib.load(filename)
        data = np.asanyarray(img.dataobj)

        return img, data

    def save_img(self, save_as, data):
        # Create output directory if it does not already exist.
        if not os.path.exists("/".join(save_as.split("/")[0:-1])):
            LOGGER.info("Output image directory does not exist. Creating it now...")
            os.makedirs("/".join(save_as.split("/")[0:-1]))
        if not isinstance(data, np.ndarray):
            data = data.cpu().numpy()
        du.save_image_as_nifti(self.gt.header, self.gt.header.get_affine(), data, save_as)
        LOGGER.info("Successfully saved image as {}".format(save_as))

    def run(self, csv, save_img, metrics, logger=LOGGER):
        start = time.time()

        for _ in self.get_subjects(csv):
            logger.info(f"Evaluating {self.model_name} on {self.subject_name}:\nnet {self.model_name}, ckpt {self.ckpt_fin}")

            # Load and prepare ground truth data
            load = time.time()
            self.set_gt(self.gt_filename)
            self.set_orig(self.orig_filename)
            logger.info("Ground truth loaded in {:0.4f} seconds".format(time.time() - load))

            # Load prediction or generate it from scratch
            load = time.time()
            pred_data = self.get_prediction()
            logger.info("Model prediction finished in {:0.4f}.".format(time.time() - load))

            # Save Image
            if save_img:
                save = time.time()
                self.save_img(self.pred_name, pred_data)
                logger.info("Image successfully saved as {} in {:0.4f} seconds".format(self.pred_name, time.time() - save))

        logger.info("Processing finished in {:0.4f} seconds".format(time.time() - start))

    def set_up_model_params(self, plane, cfg, ckpt):
        self.view_ops[plane]["cfg"] = cfg
        self.view_ops[plane]["ckpt"] = ckpt

    def get_subjects(self, csv):
        with open(csv, "r") as f:
            for line in f.readlines():
                self.current_subject = line.strip()
                self.subject_name = line.split("/")[-1] if line.split("/")[-1] != "mri" else line.split("/")[-2]
                yield


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluation metrics')

    # 1. Options for input directories and filenames
    parser.add_argument('--gt_name', type=str, default="mri/aseg.mgz",
                        help="Default name for ground truth segmentations. Default: mri/aseg.filled.mgz")
    parser.add_argument('--orig_name', type=str, dest="orig_name", default='mri/orig.mgz', help="Name of orig input")
    parser.add_argument('--pred_name', type=str, dest='pred_name', default='mri/aparc.DKTatlas+aseg.deep.mgz',
                        help="Filename to save prediction. Default: mri/aparc.DKTatlas+aseg.deep.mgz")
    parser.add_argument('--gt_dir', type=str, default=None,
                        help="Directory of ground truth (if different from orig input).")
    parser.add_argument('--csv_file', type=str, help="Csv-file with subjects to analyze (alternative to --pattern)",
                        default=None)
    parser.add_argument("--lut", type=str, default=None,
                        help="Path and name of LUT to use.")
    parser.add_argument("--substruct", action="append",
                        default=[1, 16],
                        help="Substructures to calculate DSC over.")

    # 2. Options for output
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Directory in which evaluation results should be written. "
                             "Will be created if it does not exist")
    parser.add_argument("--single_img", default=False, action="store_true",
                        help="Run single image for testing purposes instead of entire csv-file")
    parser.add_argument('--save_img', action='store_true', default=False, help="Save prediction as mgz on disk.")

    # 3. Scale factor options or membership interpolation to ground truth seg shape
    parser.add_argument("--base_res", type=float, default=1.0,
                        help="Base resolution (what resolution does input image have). Default: 1.0 (for 1mm)")
    parser.add_argument("--base_dim", type=int, default=256,
                        help="Base dimension (what dimension does input image have). Defaul: 256 (for 1mm)")
    parser.add_argument("--outdims", type=int, default=0,
                        help="Output dimension (what dimension should output image have). Default: 0 (for 1mm)")

    # 3. Checkpoint to load
    parser.add_argument('--ckpt_cor', type=str, help="coronal checkpoint to load")
    parser.add_argument('--ckpt_ax', type=str, default=None, help="axial checkpoint to load")
    parser.add_argument('--ckpt_sag', type=str, default=None, help="sagittal checkpoint to load")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference. Default=8")

    # 4. CFG-file with default options for network
    parser.add_argument("--cfg_cor", dest="cfg_cor", help="Path to the config file",
                        default=None, type=str)
    parser.add_argument("--cfg_ax", dest="cfg_ax", help="Path to the axial config file",
                        default=None, type=str)
    parser.add_argument("--cfg_sag", dest="cfg_sag", help="Path to the sagittal config file",
                        default=None, type=str)

    args = parser.parse_args()

    # Get all subjects of interest
    with open(args.csv_file, "r") as s_dirs:
        s_dirs = [line.strip() for line in s_dirs.readlines()]

    LOGGER.info("Analyzing all {} subjects from csv_file {}".format(len(s_dirs), args.csv_file))
    LOGGER.info("Output will be stored in: {}".format(args.out_dir))

    LOGGER.info("Ground truth: {}, Origs: {}".format(args.gt_name, args.orig_name))

    # Create output directory if it does not already exist.
    if args.out_dir is not None and not os.path.exists(args.out_dir):
        LOGGER.info("Output directory does not exist. Creating it now...")
        os.makedirs(args.out_dir)

    # Set Up Model
    eval = RunModelOnData(args)

    if not args.single_img:
        for subject in s_dirs:
            # Set orig and gt for testing now
            eval.set_gt(os.path.join(subject, args.gt_name))
            eval.set_orig(os.path.join(subject, args.orig_name))
            eval.set_subject(subject.split("/")[-1])
            pred_name = os.path.join(subject, args.pred_name + args.model_name + ".mgz")

            # Run model
            if not args.load_pred_from_disk:
                pred_data = eval.get_prediction()
            else:
                pred_data = np.asanyarray(nib.load(pred_name).dataobj)

            gt, gt_data = eval.get_gt()
            eval.save_img(pred_name, pred_data)

    else:
        # Set orig and gt for testing now
        eval.set_gt(os.path.join(s_dirs[0], args.gt_name))
        eval.set_orig(os.path.join(s_dirs[0], args.orig_name))
        eval.set_subject(s_dirs[0].split("/")[-1])
        pred_name = os.path.join(s_dirs[0], args.pred_name)

        # Run model
        pred_data = eval.get_prediction() #
        eval.save_img(pred_name, pred_data)
