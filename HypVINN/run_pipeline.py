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
from FastSurferCNN.utils import logging, parser_defaults
from FastSurferCNN.utils.checkpoint import get_checkpoints
from FastSurferCNN.utils.common import assert_no_root
from HypVINN.run_prediction import run_hypo_seg
from HypVINN.utils.preproc import hyvinn_preproc
from HypVINN.utils.mode_config import get_hypinn_mode_config
from HypVINN.utils.misc import create_expand_output_directory
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

    parser.add_argument('--t2', type=lambda x : None if x == 'None' else str(x), default=None,required=False, help="path to the T2 image to process")

    # 3. Image processing options
    parser.add_argument("--no_reg", action='store_false', dest="registration", help="Deactivate registration of T2 to t1,"
                                                                                    "If multi mode is run images need to be register properly")

    parser.add_argument("--qc_snap", action='store_true', dest="qc_snapshots",
                        help="Create qc snapshots")

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
    check_path = lambda x : None if x == 'None' else str(x)
    args.t1 = check_path(args.orig_name)
    #set output dir
    args.out_dir = os.path.join(args.out_dir, args.sid)
    # Warning if run as root user
    args.allow_root or assert_no_root()
    try:
        start = time.time()
        # Set up logging
        from FastSurferCNN.utils.logging import setup_logging
        if args.log_name:
            setup_logging(args.log_name)
        else:
            setup_logging(os.path.join(args.out_dir,'logs','hypvinn_seg.log'))

        LOGGER.info("Checking or downloading default checkpoints ...")
        from HypVINN.utils.checkpoint import URL as HYPVINN_URL
        get_checkpoints(args.ckpt_ax, args.ckpt_cor, args.ckpt_sag, url=HYPVINN_URL)

        # Get configuration to run multi-modal or uni-modal
        args = get_hypinn_mode_config(args)

        if args.mode:
            # Create output directory if it does not already exist.
            create_expand_output_directory(args)
            LOGGER.info("Analyzing HypVINN segmenation pipeline on Subject: {}".format(args.sid))
            LOGGER.info("Output will be stored in: {}".format(args.out_dir))
            LOGGER.info('T1 image input {}'.format(args.t1))
            LOGGER.info('T2 image input {}'.format(args.t2))

            # Pre-processing -- T1 and T2 registration
            args = hyvinn_preproc(args)
            # Segmentation pipeline
            run_hypo_seg(args)
        else:
            LOGGER.info("Failed Evaluation on {} couldn't determine the processing mode.\n "
                        "Please check that T1 or T2 images are available.\n"
                        "T1 image path: {} \n"
                        "T2 image path {} )".format(args.sid,args.t1,args.t2))
            raise RuntimeError("No T1 or T2 image available.")
    except FileNotFoundError as e:
        LOGGER.info("Failed Evaluation on {} with exception:\n{} )".format(args.sid, e))

    LOGGER.info("Processing whole pipeline finished in {:0.4f} seconds".format(time.time() - start))
