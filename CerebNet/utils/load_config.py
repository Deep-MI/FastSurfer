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

from CerebNet.utils.checkpoint import CEREBNET_AXI, CEREBNET_SAG, CEREBNET_COR
from CerebNet.config import get_cfg_cerebnet


def get_config(args) -> "yacs.CfgNode":
    """
    Given the arguments, load and initialize the config_files.

    """
    # Setup cfg.
    cfg = get_cfg_cerebnet()
    # Load config from cfg.
    if getattr(args, "cfg_file") is not None:
        if os.path.exists(args.cfg_file):
            cfg.merge_from_file(args.cfg_file)
        else:
            raise RuntimeError(f"The config file {args.cfg_file} does not exist.")
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "out_dir"):
        cfg.LOG_DIR = args.out_dir

    if getattr(args, "cfg_file") is not None:
        # derive some paths relative to the config file
        cfg_file_name = splitext(split(args.cfg_file)[1])[0]

        if cfg.TEST.ENABLE:
            cfg_file_name_first = "_".join(cfg_file_name.split("_"))
            cfg.TEST.RESULTS_DIR = join(cfg.TEST.RESULTS_DIR, cfg_file_name_first)

        cfg.LOG_DIR = join(cfg.LOG_DIR, cfg_file_name)

    # populate default paths for the checkpoints
    default_paths = [
        ("ax_ckpt", CEREBNET_AXI),
        ("sag_ckpt", CEREBNET_SAG),
        ("cor_ckpt", CEREBNET_COR),
    ]
    path_ax, path_sag, path_cor = [
        getattr(args, name, default_path) for name, default_path in default_paths
    ]

    for plane, path in [
        ("axial", path_ax),
        ("sagittal", path_sag),
        ("coronal", path_cor),
    ]:
        setattr(cfg.TEST, f"{plane.upper()}_CHECKPOINT_PATH", path)

    # overwrite the batch size if it is passed as a parameter
    batch_size = getattr(args, "batch_size", None)
    if batch_size is not None:
        cfg.TEST.BATCH_SIZE = batch_size

    return cfg


def setup_options():
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
