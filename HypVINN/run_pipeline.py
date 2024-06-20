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

import argparse
from pathlib import Path
from typing import Optional
import time

from FastSurferCNN.utils import PLANES, Plane, logging, parser_defaults
from FastSurferCNN.utils.checkpoint import get_checkpoints, load_checkpoint_config_defaults
from FastSurferCNN.utils.common import assert_no_root
from HypVINN.run_prediction import run_hypo_seg
from HypVINN.utils.preproc import hyvinn_preproc
from HypVINN.utils.mode_config import get_hypinn_mode
from HypVINN.utils.misc import create_expand_output_directory
from HypVINN.utils.checkpoint import YAML_DEFAULT as CHECKPOINT_PATHS_FILE
##
# Global Variables
##
LOGGER = logging.get_logger(__name__)


def optional_path(a: str) -> Optional[Path]:
    """
    Convert a string to a Path object or None.

    Parameters
    ----------
    a : str
        The string to convert.

    Returns
    -------
    Optional[Path]
        The Path object or None.
    """
    if a.lower() in ("none", ""):
        return None
    return Path(a)


def option_parse() -> argparse.ArgumentParser:
    """
    A function to create an ArgumentParser object and parse the command line arguments.

    Returns
    -------
    argparse.Ar
        The parser object to parse arguments from the command line.
    """
    from HypVINN.config.hypvinn_files import HYPVINN_SEG_NAME
    parser = argparse.ArgumentParser(
        description="Script for Hypothalamus Segmentation.",
    )

    # 1. Directory information (where to read from, where to write from and to incl. search-tag)
    parser = parser_defaults.add_arguments(
        parser, ["in_dir", "sd", "sid"],
    )

    parser = parser_defaults.add_arguments(parser, ["seg_log"])

    # 2. Options for the MRI volumes
    parser = parser_defaults.add_arguments(
        parser, ["t1"]
    )
    parser.add_argument(
        '--t2',
        type=optional_path,
        default=None,
        required=False,
        help="Path to the T2 image to process.",
    )

    # 3. Image processing options
    parser.add_argument(
        "--qc_snap",
        action='store_true',
        dest="qc_snapshots",
        help="Create qc snapshots in <sd>/<sid>/qc_snapshots.",
    )
    parser.add_argument(
        "--reg_mode",
        type=str,
        default="coreg",
        choices=["none", "coreg", "robust"],
        help="Freesurfer Registration type to run. coreg: mri_coreg, "
             "robust : mri_robust_register, none: entirely deactivates "
             "registration of T2 to T1, if both images are passed, "
             "images need to be register properly externally.",
    )
    parser.add_argument(
        "--hypo_segfile",
        type=Path,
        default=Path("mri") / HYPVINN_SEG_NAME,
        dest="hypo_segfile",
        help=""
    )

    # 4. Options for advanced, technical parameters
    advanced = parser.add_argument_group(title="Advanced options")
    parser_defaults.add_arguments(
        advanced,
        ["device", "viewagg_device", "threads", "batch_size", "async_io", "allow_root"],
    )

    files: dict[Plane, str | Path] = {k: "default" for k in PLANES}
    # 5. Checkpoint to load
    parser_defaults.add_plane_flags(
        advanced,
        "checkpoint",
        files,
        CHECKPOINT_PATHS_FILE,
    )

    parser_defaults.add_plane_flags(
        advanced,
        "config",
        {
            "coronal": Path("HypVINN/config/HypVINN_coronal_v1.0.0.yaml"),
            "axial": Path("HypVINN/config/HypVINN_axial_v1.0.0.yaml"),
            "sagittal": Path("HypVINN/config/HypVINN_sagittal_v1.0.0.yaml"),
        },
        CHECKPOINT_PATHS_FILE,
    )
    return parser


def main(args: argparse.Namespace) -> int | str:
    """
    Main function of the hypothalamus segmentation module.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments to the script as created by `options_parse`.

    Returns
    -------
    int, str
        0, if successful, an error message describing the cause for the
        failure otherwise.
    """
    
    # mapped freesurfer orig input name to the hypvinn t1 name
    args.t1 = optional_path(args.orig_name)
    # set output dir
    args.out_dir = args.out_dir / args.sid
    # Warning if run as root user
    args.allow_root or assert_no_root()
    start = time.time()
    try:
        # Set up logging
        from FastSurferCNN.utils.logging import setup_logging
        if not args.log_name:
            args.log_name = args.out_dir / "scripts" / "hypvinn_seg.log"
        setup_logging(args.log_name)

        LOGGER.info("Checking or downloading default checkpoints ...")
        urls = load_checkpoint_config_defaults(
            "url",
            filename=CHECKPOINT_PATHS_FILE,
        )
        get_checkpoints(args.ckpt_ax, args.ckpt_cor, args.ckpt_sag, urls=urls)

        # Get configuration to run multi-modal or uni-modal
        mode = get_hypinn_mode(
            getattr(args, "t1", None),
            getattr(args, "t2", None),
        )
        args.mode = mode

        if mode:
            # Create output directory if it does not already exist.
            create_expand_output_directory(args.out_dir, args.qc_snapshots)
            LOGGER.info(
                f"Running HypVINN segmentation pipeline on subject {args.sid}"
            )
            LOGGER.info(f"Output will be stored in: {args.out_dir}")
            LOGGER.info(f"T1 image input {args.t1}")
            LOGGER.info(f"T2 image input {args.t2}")

            # Pre-processing -- T1 and T2 registration
            if mode == "t1t2":
                # Note, that args.t1 and args.t2 are guaranteed to be not None
                # via get_hypvinn_mode, which only returns t1t2, if t1 and t2
                # exist.
                # hypvinn_preproc returns the path to the t2 that is registered
                # to the t1
                args.t2 = hyvinn_preproc(
                    mode,
                    getattr(args, "reg_mode", "coreg"),
                    Path(args.t1),
                    Path(args.t2),
                    Path(args.out_dir),
                )
            # Segmentation pipeline
            run_hypo_seg(
                args,
                subject_name=args.sid,
                out_dir=Path(args.out_dir),
                t1_path=Path(args.t1),
                t2_path=Path(args.t2),
                mode=args.mode,
                threads=args.threads,
                seg_file=args.hypo_segfile,
            )
        else:
            return (
                f"Failed Evaluation on {args.sid} couldn't determine the "
                f"processing mode. Please check that T1 or T2 images are "
                f"available.\nT1 image path: {args.t1}\nT2 image path "
                f"{args.t2}.\nNo T1 or T2 image available."
            )
    except (FileNotFoundError, RuntimeError) as e:
        LOGGER.info(f"Failed Evaluation on {args.sid}:")
        LOGGER.exception(e)
    else:
        LOGGER.info(
            f"Processing whole pipeline finished in {time.time() - start:.4f} "
            f"seconds"
        )


if __name__ == "__main__":
    # arguments
    parser = option_parse()
    args = parser.parse_args()
    import sys
    sys.exit(main(args))
