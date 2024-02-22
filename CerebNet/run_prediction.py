#!/bin/python

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
from pathlib import Path

from FastSurferCNN.utils import logging, parser_defaults
from CerebNet.utils.load_config import get_config
from CerebNet.inference import Inference
from FastSurferCNN.utils.checkpoint import get_checkpoints, get_plane_default
from FastSurferCNN.utils.common import assert_no_root, SubjectList
from FastSurferCNN.utils.parser_defaults import FASTSURFER_ROOT

logger = logging.get_logger(__name__)
DEFAULT_CEREBELLUM_STATSFILE = Path("stats/cerebellum.CerebNet.stats")
CEREBNET_CHECKPOINT_PATHS_FILE = FASTSURFER_ROOT / "CerebNet/config/checkpoint_paths.yaml"

def setup_options():
    """
    Configure and return an argument parser for the segmentation script.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.
    """
    # Training settings
    parser = argparse.ArgumentParser(description="Segmentation")

    # 1. Directory information (where to read from, where to write from and to incl.
    # search-tag)
    parser = parser_defaults.add_arguments(
        parser, ["in_dir", "tag", "csv_file", "sd", "sid", "remove_suffix"]
    )

    # 2. Options for the MRI volumes
    parser = parser_defaults.add_arguments(
        parser, ["t1", "conformed_name", "norm_name", "asegdkt_segfile"]
    )
    parser.add_argument(
        "--cereb_segfile",
        dest="cereb_segfile",
        default=Path("mri/cerebellum.CerebNet.nii.gz"),
        type=Path,
        help="Name under which segmentation will be saved. "
             "Default: mri/cerebellum.CerebNet.nii.gz.",
    )

    # 3. Options for additional files and parameters
    parser.add_argument(
        "--cereb_statsfile",
        dest="cereb_statsfile",
        type=Path,
        default=None,
        help=f"Name under which the statsfield for the cerebellum will be saved. "
             f"Default: None, do not calculate stats file. This option supports the "
             f"special option 'default', which saves the stats file at "
             f"{DEFAULT_CEREBELLUM_STATSFILE} in the subject directory.",
    )
    parser = parser_defaults.add_arguments(parser, ["seg_log"])

    # 4. Options for advanced, technical parameters
    advanced = parser.add_argument_group(title="Advanced options")
    parser_defaults.add_arguments(
        advanced,
        ["device", "viewagg_device", "threads", "batch_size", "async_io", "allow_root"],
    )


    parser_defaults.add_plane_flags(
        advanced,
        "checkpoint",
        {"coronal": "default", "axial": "default", "sagittal": "default"},
        CEREBNET_CHECKPOINT_PATHS_FILE
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


def main(args: argparse.Namespace) -> int | str:
    """
    Main function to run the inference based on the given command line arguments.
    This implementation is inspired by methods described in CerebNet for cerebellum sub-segmentation.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments parsed by `argparse.ArgumentParser`.

    Returns
    -------
    int
        Returns 0 upon successful execution to indicate success.
    str
        A message indicating the failure reason in case of an exception.

    References
    ----------
    Faber J, Kuegler D, Bahrami E, et al. CerebNet: A fast and reliable deep-learning
    pipeline for detailed cerebellum sub-segmentation. NeuroImage 264 (2022), 119703.
    https://doi.org/10.1016/j.neuroimage.2022.119703
    """
    cfg = get_config(args)
    cfg.TEST.ENABLE = True
    cfg.TRAIN.ENABLE = False

    # Warning if run as root user
    getattr(args, "allow_root", False) or assert_no_root()

    # Set up logging
    from FastSurferCNN.utils.logging import setup_logging

    setup_logging(getattr(args, "log_name"))

    subjects_kwargs = {}
    cereb_statsfile = getattr(args, "cereb_statsfile", None)
    if cereb_statsfile is None or str(cereb_statsfile) == "default":
        cereb_statsfile = DEFAULT_CEREBELLUM_STATSFILE
        args.cereb_statsfile = cereb_statsfile
    if cereb_statsfile is not None:
        subjects_kwargs["cereb_statsfile"] = "cereb_statsfile"
        if not hasattr(args, "norm_name"):
            return (
                f"Execution failed because `--cereb_statsfile {cereb_statsfile}` "
                f"requires `--norm_name <filename>` to be passed!"
            )
        subjects_kwargs["norm_name"] = "norm_name"

    logger.info("Checking or downloading default checkpoints ...")
    
    urls = get_plane_default("URL", filename=CEREBNET_CHECKPOINT_PATHS_FILE)

    get_checkpoints(args.ckpt_ax, args.ckpt_cor, args.ckpt_sag, urls=urls)

    # Check input and output options and get all subjects of interest
    subjects = SubjectList(
        args, asegdkt_segfile="pred_name", segfile="cereb_segfile", **subjects_kwargs
    )

    try:
        tester = Inference(
            cfg,
            threads=getattr(args, "threads", 1),
            device=args.device,
            viewagg_device=args.viewagg_device,
        )
        return tester.run(subjects)
    except Exception as e:
        logger.exception(e)
        return "Execution failed with error: \n" + "\n".join(str(a) for a in e.args)


if __name__ == "__main__":
    parser = setup_options()
    args = parser.parse_args()

    sys.exit(main(args))
