import os
# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
import argparse

from FastSurferCNN.utils import logging, parser_defaults

from CerebNet.utils.load_config import get_config
from CerebNet.inference import Inference
from FastSurferCNN.utils.checkpoint import get_checkpoints
from FastSurferCNN.utils.common import assert_no_root, SubjectList

logger = logging.get_logger(__name__)
DEFAULT_CEREBELLUM_STATSFILE = "stats/cerebellum.CerebNet.stats"

def setup_options():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation')

    # 1. Directory information (where to read from, where to write from and to incl. search-tag)
    parser = parser_defaults.add_arguments(parser, ["in_dir", "tag", "csv_file", "sd", "sid", "remove_suffix"])

    # 2. Options for the MRI volumes
    parser = parser_defaults.add_arguments(parser, ["t1", "conformed_name", "norm_name", "aparc_aseg_segfile"])
    parser.add_argument('--cereb_segfile', dest='cereb_segfile', default='mri/cerebellum.CerebNet.nii.gz',
                        help='Name under which segmentation will be saved. Default: mri/cerebellum.CerebNet.nii.gz.')

    # 3. Options for additional files and parameters
    parser.add_argument('--cereb_statsfile', dest='cereb_statsfile', default=None,
                        help=f'Name under which the statsfield for the cerebellum will be saved. Default: None, do not '
                             f'calculate stats file. This option supports the special option "default", which saves the '
                             f'stats file at {DEFAULT_CEREBELLUM_STATSFILE} in the subject directory.')
    parser = parser_defaults.add_arguments(parser, ["seg_log"])

    # 4. Options for advanced, technical parameters
    advanced = parser.add_argument_group(title="Advanced options")
    parser_defaults.add_arguments(advanced, ["device", "viewagg_device", "threads", "batch_size", "async_io", "allow_root"])

    from CerebNet.utils.checkpoint import CEREBNET_COR, CEREBNET_AXI, CEREBNET_SAG
    parser_defaults.add_plane_flags(advanced, "checkpoint",
                                    {"coronal": CEREBNET_COR, "axial": CEREBNET_AXI, "sagittal": CEREBNET_SAG})

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See CerebNet/config/cerebnet.py for additional options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser


def main(args):
    cfg = get_config(args)
    cfg.TEST.ENABLE = True
    cfg.TRAIN.ENABLE = False

    # Warning if run as root user
    args.allow_root or assert_no_root()

    # Set up logging
    from FastSurferCNN.utils.logging import setup_logging
    from CerebNet.utils.checkpoint import URL as CEREBNET_URL
    setup_logging(args.log_name)

    subjects_kwargs = {}
    cereb_statsfile = getattr(args, 'cereb_statsfile')
    if cereb_statsfile == 'default':
        args.cereb_statsfile = DEFAULT_CEREBELLUM_STATSFILE
    if cereb_statsfile is not None:
        subjects_kwargs["cereb_statsfile"] = "cereb_statsfile"

    logger.info("Checking or downloading default checkpoints ...")
    get_checkpoints(args.ckpt_ax, args.ckpt_cor, args.ckpt_sag, url=CEREBNET_URL)

    # Check input and output options and get all subjects of interest
    subjects = SubjectList(args, aparc_aseg_segfile='pred_name', segfile='cereb_segfile', **subjects_kwargs)

    try:
        tester = Inference(cfg,
                           threads=getattr(args, "threads", 1), device=args.device, viewagg_device=args.viewagg_device)
        return tester.run(subjects)
    except Exception as e:
        logger.exception(e)
        return "Execution failed with error: \n" + "\n".join(str(a) for a in e.args)


if __name__ == '__main__':
    parser = setup_options()
    sys.exit(main(parser.parse_args()))
