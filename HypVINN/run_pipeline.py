# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

import os
import sys
from FastSurferCNN.utils import logging, parser_defaults
from FastSurferCNN.utils.checkpoint import get_checkpoints
from FastSurferCNN.utils.common import assert_no_root
from HypVINN.run_prediction import run_hypo_seg
from HypVINN.run_preproc import run_hypo_preproc
from HypVINN.utils.mode_config import get_hypinn_mode_config
import time
##
# Global Variables
##
LOGGER = logging.get_logger(__name__)

def option_parse():
    import argparse

    parser = argparse.ArgumentParser(description='Hypothalamus Segmentation')

    # 1. Directory information (where to read from, where to write from and to incl. search-tag)
    parser = parser_defaults.add_arguments(
        parser, ["in_dir", "sd", "sid"]
    )

    parser = parser_defaults.add_arguments(parser, ["seg_log"])

    # 2. Options for the MRI volumes
    parser = parser_defaults.add_arguments(
        parser, ["t1"]
    )

    parser.add_argument('--t2', type=str, required=False, help="path to the T2 image to process")

    parser.add_argument('--mode', type=str, default="auto", choices=["t2", "t1", "multi","auto"],
                        help="Modalities to load. t1 : only T1 images, t2 :only T2 images , multi: both T1 and T2 or auto: choose mode based on the passed inputs" )



    # 3. Image processing options
    parser.add_argument("--no_pre_proc", action='store_false', dest="pre_process_pipeline", help="Deactivate preprocessing pipeline")
    parser.add_argument("--no_bc", action='store_false', dest="bias_field_correction", help="Deactivate bias field correction, "
                                                                                            "it is recommended to do bias field correction for calculating volumes taking account partial volume effects")
    parser.add_argument("--no_reg", action='store_false', dest="registration", help="Deactivate registration of T2 to t1,"
                                                                                    "If multi mode is run images need to be register properly")

    parser.add_argument('--reg_type', type=str, default="coreg", choices=["coreg", "robust"],
                        help="Freesurfer Registration type to run. coreg: mri_coreg, robust : mri_robust_register  ")


    # 4. Options for advanced, technical parameters
    advanced = parser.add_argument_group(title="Advanced options")
    parser_defaults.add_arguments(
        advanced,
        ["device", "viewagg_device", "threads", "batch_size", "async_io", "allow_root"],
    )


    from HypVINN.utils.checkpoint import HYPVINN_AXI, HYPVINN_COR, HYPVINN_SAG

    # 5. Checkpoint to load
    parser_defaults.add_plane_flags(
        advanced,
        "checkpoint",
        {"coronal": HYPVINN_COR, "axial": HYPVINN_AXI, "sagittal": HYPVINN_SAG},
    )

    parser = parser_defaults.add_plane_flags(
        parser,
        "config",
        {
            "coronal": "HypVINN/config/HypVINN_coronal_v1.0.0.yaml",
            "axial": "HypVINN/config/HypVINN_axial_v1.0.0.yaml",
            "sagittal": "HypVINN/config/HypVINN_sagittal_v1.0.0.yaml",
        },
    )
    args = parser.parse_args()

    return args
if __name__ == "__main__":

    #arguments
    args = option_parse()
    #mapped freesurfer orig input name to the hypvinn t1 name
    args.t1 = args.orig_name

    # Warning if run as root user
    args.allow_root or assert_no_root()

    #Get configuration to run multi-modal or uni-modal
    args = get_hypinn_mode_config(args)
    start = time.time()

    args.out_dir = os.path.join(args.out_dir, args.sid)

    # Create output directory if it does not already exist.
    if args.out_dir is not None and not os.path.exists(args.out_dir):
        LOGGER.info("Output directory does not exist. Creating it now...")
        os.makedirs(args.out_dir)
        os.makedirs(os.path.join(args.out_dir, 'mri','transforms'), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, 'stats'), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, 'qc_snapshots'), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, 'logs'), exist_ok=True)

    # Set up logging
    from FastSurferCNN.utils.logging import setup_logging
    from HypVINN.utils.checkpoint import URL as HYPVINN_URL

    if args.log_name:
        setup_logging(args.log_name)
    else:
        setup_logging(os.path.join(args.out_dir,'logs','hypvinn_seg.log'))

    LOGGER.info("Checking or downloading default checkpoints ...")
    get_checkpoints(args.ckpt_ax, args.ckpt_cor, args.ckpt_sag, url=HYPVINN_URL)

    LOGGER.info("Analyzing HypVINN segmenation pipeline on Subject: {}".format(args.sid))
    LOGGER.info('HypVINN is setup to {} input mode'.format(args.mode))
    LOGGER.info("Output will be stored in: {}".format(args.out_dir))
    LOGGER.info('T1 image input {}'.format(args.t1))
    LOGGER.info('T2 image input {}'.format(args.t2))

    try:
        # Pre-processing -- Bias Field correction and T1 and T2 registration
        if args.pre_process_pipeline:
            args = run_hypo_preproc(args)
        # Segmentation pipeline
        run_hypo_seg(args)

    except FileNotFoundError as e:
        LOGGER.info("Failed Evaluation on {} with exception:\n{} )".format(args.sid, e))

    LOGGER.info("Processing whole pipeline finished in {:0.4f} seconds".format(time.time() - start))
