from typing import Iterable, Mapping, Union, Literal
import argparse
from os import path

from FastSurferCNN.utils.arg_types import vox_size as __vox_size, conform_to_one_mm as __conform_to_one_mm

FASTSURFER_ROOT = path.dirname(path.dirname(path.dirname(__file__)))
PLANE_SHORT = {"checkpoint": "ckpt", "config": "cfg"}
PLANE_HELP = {"checkpoint": "{} checkpoint to load",
              "config": "Path to the {} config file"}
VoxSize = Union[Literal['min'], float]


def __arg(*args, **kwargs):
    def _stub(parser):
        return parser.add_argument(*args, **kwargs)
    return _stub


ALL_FLAGS = {
    "t1": __arg(
        '--t1', type=str, dest="orig_name", default='mri/orig.mgz',
        help="Name of T1 full head MRI. Absolute path if single image else "
             "common image name. Default: mri/orig.mgz"),
    "remove_suffix": __arg(
        '--remove_suffix', type=str, dest="remove_suffix", default='',
        help="Optional: remove suffix from path definition of input file to yield correct subject name "
             "(e.g. /ses-x/anat/ for BIDS or /mri/ for FreeSurfer input). Default: do not remove anything."),
    "sid": __arg(
        '--sid', type=str, dest="sid", default=None,
        help="Optional: directly set the subject id to use. Can be used for single subject input. For multi-subject "
             "processing, use remove suffix if sid is not second to last element of input file passed to --t1"),
    "aparc_aseg_segfile": __arg(
        '--aparc_aseg_segfile', type=str, dest='pred_name', default='mri/aparc.DKTatlas+aseg.deep.mgz',
        help="Name of intermediate DL-based segmentation file (similar to aparc+aseg). "
             "When using FastSurfer, this segmentation is already conformed, since inference "
             "is always based on a conformed image. Absolute path if single image else common "
             "image name. Default: mri/aparc.DKTatlas+aseg.deep.mgz"),
    "conformed_name": __arg(
        '--conformed_name', type=str, dest='conf_name', default='mri/orig.mgz',
        help="Name under which the conformed input image will be saved, in the same directory "
             "as the segmentation (the input image is always conformed first, if it is not "
             "already conformed). The original input image is saved in the output directory "
             "as $id/mri/orig/001.mgz. Default: mri/orig.mgz."),
    "brainmask_name": __arg(
        '--brainmask_name', type=str, dest='brainmask_name', default='mri/mask.mgz',
        help="Name under which the brainmask image will be saved, in the same directory "
             "as the segmentation. The brainmask is created from the aparc_aseg segmentation "
             "(dilate 5, erode 4, largest component). Default: mri/mask.mgz."),
    "aseg_name": __arg(
        '--aseg_name', type=str, dest='aseg_name', default='mri/aseg.auto_noCCseg.mgz',
        help="Name under which the reduced aseg segmentation will be saved, in the same directory "
             "as the aparc-aseg segmentation (labels of full aparc segmentation are reduced to aseg). "
             "Default: mri/aseg.auto_noCCseg.mgz."),
    "seg_log": __arg(
        '--seg_log', type=str, dest='log_name', default="",
        help="Absolute path to file in which run logs will be saved. If not set, logs will "
             "not be saved."),
    "device": __arg(
        '--device', default="auto",
        help="Select device to run inference on: cpu, or cuda (= Nvidia gpu) or specify a certain gpu "
             "(e.g. cuda:1), default: auto"),
    "viewagg_device": __arg(
        '--viewagg_device', dest='viewagg_device', type=str,
        default="auto",
        help="Define the device, where the view aggregation should be run. By default, the program checks "
             "if you have enough memory to run the view aggregation on the gpu (cuda). The total memory is "
             "considered for this decision. If this fails, or you actively overwrote the check with setting "
             "> --viewagg_device cpu <, view agg is run on the cpu. Equivalently, if you define "
             "> --viewagg_device cuda <, view agg will be run on the gpu (no memory check will be done)."),
    "in_dir": __arg(
        "--in_dir", type=str, default=None,
        help="Directory in which input volume(s) are located. "
             "Optional, if full path is defined for --t1."),
    "tag": __arg(
        '--tag', dest='search_tag', default="*",
        help='Search tag to process only certain subjects. If a single image should be analyzed, '
             'set the tag with its id. Default: processes all.'),
    "csv_file": __arg(
        '--csv_file', type=str, help="Csv-file with subjects to analyze (alternative to --tag",
        default=None),
    "batch_size": __arg(
        '--batch_size', type=int, default=1, help="Batch size for inference. Default=1"),
    "sd": __arg(
        "--sd", type=str, default=None, dest="out_dir",
        help="Directory in which evaluation results should be written. "
             "Will be created if it does not exist. Optional if full path is defined for --pred_name."),
    "qc_log": __arg(
        '--qc_log', type=str, dest='qc_log', default="",
        help="Absolute path to file in which a list of subjects that failed QC check  (when processing multiple "
             "subjects) will be saved. If not set, the file will not be saved."),
    "vox_size": __arg(
        '--vox_size', type=__vox_size, default="min", dest='vox_size',
        help="Choose the primary voxelsize to process, must be either a number between 0 and 1 (below 0.7 is "
             "experimental) or 'min' (default). A number forces processing at that specific voxel size, 'min' "
             "determines the voxel size from the image itself (conforming to the minimum voxel size, or 1 if "
             "the minimum voxel size is above 0.95mm). "),
    "conform_to_1mm_threshold": __arg(
        '--conform_to_1mm_threshold', type=__conform_to_one_mm, default=0.95, dest="conform_to_1mm_threshold",
        help="The voxelsize threshold, above which images will be conformed to 1mm isotropic, if the --vox_size "
             "argument is also 'min' (the --vox_size default setting). Contrary to conform.py, the default behavior"
             "of %(prog)s is to resample all images _above 0.95mm_ to 1mm."),
    "lut": __arg(
        "--lut", type=str, help="Path and name of LUT to use.",
        default=path.join(FASTSURFER_ROOT, "FastSurferCNN/config/FastSurfer_ColorLUT.tsv")),
    "allow_root": __arg(
        "--allow_root", action="store_true", dest="allow_root",
        help="Allow execution as root user."
    )
}


def add_arguments(parser: argparse.ArgumentParser, flags: Iterable[str]) -> argparse.ArgumentParser:
    """
    Add default flags to the parser from the flags list in order.

    Args:
        parser: The parser to add flags to.
        flags: the flags to add from 'device', 'viewagg_device'.
    """
    for flag in flags:
        if flag.startswith("--"):
            flag = flag[2:]
        add_flag = ALL_FLAGS.get(flag, None)
        if add_flag is not None:
            add_flag(parser)
        else:
            raise ValueError(f"The flag '{flag}' is not defined in FastSurferCNN.utils.parse.add_arguments().")
    return parser


def add_plane_flags(parser: argparse.ArgumentParser, type: str, files: Mapping[str, str]) -> argparse.ArgumentParser:
    """
    Helper function to add plane arguments. Arguments will be added for each entry in files, where the key is the "plane"
    and the values is the file name (relative for path relative to FASTSURFER_HOME.

    Args:
        parser: The parser to add flags to.
        type: The type of files (for help text and prefix from "checkpoint" and "config".
            "checkpoint" will lead to flags like "--ckpt_{plane}", "config" to "--cfg_ {plane}"
        files: A dictionary of plane to filename. Relative files are assumed to be relative to the FastSurfer root
            directory.

    Returns:
        The parser object.
    """

    if type not in PLANE_SHORT:
        raise ValueError("type must be either config or checkpoint.")

    for key, filepath in files.items():
        if not path.isabs(filepath):
            filepath = path.join(FASTSURFER_ROOT, filepath)
        # find the first vowel in the key
        flag = key.strip().lower()
        index = min(i for i in (flag.find(v) for v in "aeiou") if i >= 0)
        flag = flag[:index+2]
        parser.add_argument(f'--{PLANE_SHORT[type]}_{flag}', type=str,
                            dest=f"{PLANE_SHORT[type]}_{flag}",
                            help=PLANE_HELP[type].format(key),
                            default=filepath)
    return parser
