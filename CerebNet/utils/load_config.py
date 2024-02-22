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
import argparse
import os.path
import sys
from os.path import join, split, splitext
from FastSurferCNN.utils.checkpoint import get_plane_default
from CerebNet.config import get_cfg_cerebnet


def get_config(args) -> "yacs.CfgNode":
    """
    Given the arguments, load and initialize the config_files.
    """
    # Setup cfg.
    cfg = get_cfg_cerebnet()
    # Load config from cfg.
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "out_dir"):

        cfg.LOG_DIR = args.out_dir    
    path_ax, path_sag, path_cor = [
        getattr(args, name) for name in ["ckpt_ax", "ckpt_sag", "ckpt_cor"]
    ]

    for plane, path in [
        ("AXIAL", path_ax),
        ("SAGITTAL", path_sag),
        ("CORONAL", path_cor),
    ]:
        setattr(cfg.TEST, f"{plane}_CHECKPOINT_PATH", str(path))

    # overwrite the batch size if it is passed as a parameter
    batch_size = getattr(args, "batch_size", None)
    if batch_size is not None:
        cfg.TEST.BATCH_SIZE = batch_size

    return cfg


def setup_options():
    """
    Set up the command-line options for the segmentation.

    Returns
    -------
    argparse.Namespace
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="config_files/CerebNet.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See CerebNet/config/cerebnet.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()
