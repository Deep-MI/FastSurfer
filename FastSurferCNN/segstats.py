#!/bin/python

# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases(DZNE), Bonn
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
import logging
from functools import partial, reduce
from itertools import product
from numbers import Number
from pathlib import Path
from typing import (
    Any,
    Callable,
    cast,
    Iterable,
    IO,
    Literal,
    Optional,
    overload,
    Sequence,
    Sized,
    Type,
    TypedDict,
    TypeVar, Container,
)
from concurrent.futures import Executor, ThreadPoolExecutor


import numpy as np
import pandas as pd
from numpy import typing as npt

from FastSurferCNN.utils.arg_types import float_gt_zero_and_le_one as robust_threshold
from FastSurferCNN.utils.arg_types import int_ge_zero as id_type
from FastSurferCNN.utils.arg_types import int_gt_zero as patch_size_type
from FastSurferCNN.utils.parser_defaults import add_arguments
from FastSurferCNN.utils.threads import get_num_threads

# Constants
USAGE = ("python segstats.py (-norm|-pv) <input_norm> -i <input_seg> "
         "-o <output_seg_stats> [optional arguments] [{measures,mri_segstats} ...]")
DESCRIPTION = ("Script to calculate partial volumes and other segmentation statistics "
               "of a segmentation file.")
VERSION = "1.1"
HELPTEXT = f"""
Dependencies:

    Python 3.10

    Numpy
    http://www.numpy.org

    Nibabel to read images
    http://nipy.org/nibabel/
    
    Pandas to read/write stats files etc.
    https://pandas.pydata.org/

Original Author: David Kügler
Date: Dec-30-2022
Modified: Dec-07-2023

Revision: {VERSION}
"""
FILTER_SIZES = (3, 15)
COLUMNS = ["Index", "SegId", "NVoxels", "Volume_mm3", "StructName", "Mean", "StdDev",
           "Min", "Max", "Range"]

# Type definitions
_NumberType = TypeVar("_NumberType", bound=Number)
_IntType = TypeVar("_IntType", bound=np.integer)
_DType = TypeVar("_DType", bound=np.dtype)
_ArrayType = TypeVar("_ArrayType", bound=np.ndarray)
SlicingTuple = tuple[slice, ...]
SlicingSequence = Sequence[slice]
VirtualLabel = dict[int, Sequence[int]]
_GlobalStats = tuple[int, int, Optional[_NumberType], Optional[_NumberType],
                     Optional[float], Optional[float], float, npt.NDArray[bool]]
SubparserCallback = Type[argparse.ArgumentParser.add_subparsers]


class _RequiredPVStats(TypedDict):
    SegId: int
    NVoxels: int
    Volume_mm3: float


class _OptionalPVStats(TypedDict, total=False):
    StructName: str
    Mean: float
    StdDev: float
    Min: float
    Max: float
    Range: float


class PVStats(_RequiredPVStats, _OptionalPVStats):
    """Dictionary of volume statistics for partial volume evaluation and global stats"""
    pass


class HelpFormatter(argparse.HelpFormatter):
    """
    Help formatter that forces line breaks in texts where the text is <br>.
    """

    def _linebreak_sub(self):
        """
        Get the linebreak substitution string.

        Returns
        -------
        str
            The linebreak substitution string ("<br>").
        """
        return getattr(self, "linebreak_sub", "<br>")

    def _item_symbol(self):
        return getattr(self, "item_symbol", "- ")

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        """
        Fill text with line breaks based on the linebreak substitution string.

        Parameters
        ----------
        text : str
            The input text.
        width : int
            The width for filling the text.
        indent : int
            The indentation level.

        Returns
        -------
        str
            The formatted text with line breaks.
        """
        cond_len, texts = self._itemized_lines(text)
        lines = (super(HelpFormatter, self)._fill_text(t[p:], width, indent + " " * p)
                 for t, (c, p) in zip(texts, cond_len))
        return "\n".join("- " + t[p:] if c else t for t, (c, p) in zip(lines, cond_len))

    def _itemized_lines(self, text):
        texts = text.split(self._linebreak_sub())
        item = self._item_symbol()
        il = len(item)
        cond_len = [(c, il if c else 0) for c in map(lambda t: t[:il] == item, texts)]
        texts = [t[p:] for t, (c, p) in zip(texts, cond_len)]
        return cond_len, texts

    def _split_lines(self, text: str, width: int) -> list[str]:
        """
        Split lines in the text based on the linebreak substitution string.

        Parameters
        ----------
        text : str
            The input text.
        width : int
            The width for splitting lines.

        Returns
        -------
        list[str]
            The list of lines.
        """
        def indent_list(items: list[str]) -> list[str]:
            return ["- " + items[0]] + ["  " + l for l in items[1:]]

        cond_len, texts = self._itemized_lines(text)
        from itertools import chain
        lines = (super(HelpFormatter, self)._split_lines(tex, width - p)
                 for tex, (c, p) in zip(texts, cond_len))
        lines = ((indent_list(lst) if c[0] else lst) for lst, c in zip(lines, cond_len))
        return list(chain.from_iterable(lines))


def make_arguments(helpformatter: bool = False) -> argparse.ArgumentParser:
    """
    Create an argument parser object with all parameters of the script.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.
    """
    if helpformatter:
        kwargs = {
            "epilog": HELPTEXT.replace("\n", "<br>"),
            "formatter_class": HelpFormatter,
        }
    else:
        kwargs = {"epilog": HELPTEXT}
    parser = argparse.ArgumentParser(
        usage=USAGE,
        description=DESCRIPTION,
        add_help=False,
        **kwargs,
    )
    add_two_help_messages(parser)
    parser.add_argument(
        "--pvfile",
        "-pv",
        type=Path,
        dest="pvfile",
        help="Path to image used to compute the partial volume effects (default: the "
             "file passed as normfile). This file is required, either directly or "
             "indirectly via normfile.",
    )
    parser.add_argument(
        "-norm",
        "--normfile",
        type=Path,
        dest="normfile",
        help="Path to biasfield-corrected image (the same image space as "
             "segmentation). This file is used to calculate intensity values. Also, if "
             "no pvfile is defined, it is used as pvfile. One of normfile or pvfile is "
             "required.",
    )
    parser.add_argument(
        "-i",
        "--segfile",
        type=Path,
        dest="segfile",
        required=True,
        help="Segmentation file to read and use for evaluation (required).",
    )
    parser.add_argument(
        "-o",
        "--segstatsfile",
        type=Path,
        required=True,
        dest="segstatsfile",
        help="Path to output segstats file.",
    )

    parser.add_argument(
        "--excludeid",
        type=id_type,
        nargs="*",
        default=[],
        help="List of segmentation ids (integers) to exclude in analysis, "
             "e.g. `--excludeid 0 1 10` (default: None).",
    )
    parser.add_argument(
        "--ids",
        type=id_type,
        nargs="*",
        help="List of exclusive segmentation ids (integers) to use "
             "(default: all ids in --lut or all ids in image).",
    )
    parser.add_argument(
        "--merged_label",
        type=id_type,
        nargs="+",
        dest="merged_labels",
        default=[],
        action="append",
        help="Add a 'virtual' label (first value) that is the combination of all "
             "following values, e.g. `--merged_label 100 3 4 8` will compute the "
             "statistics for label 100 by aggregating labels 3, 4 and 8.",
    )
    parser.add_argument(
        "--robust",
        type=robust_threshold,
        dest="robust",
        default=None,
        help="Whether to calculate robust segmentation metrics. This parameter "
             "expects the fraction of values to keep, e.g. `--robust 0.95` will "
             "ignore the 2.5%% smallest and the 2.5%% largest values in the "
             "segmentation when calculating the statistics (default: no robust "
             "statistics == `--robust 1.0`).",
    )
    subparsers = parser.add_subparsers(title="Suboptions", dest="subparser")
    add_measure_parser(subparsers.add_parser)
    advanced = parser.add_argument_group(title="Advanced options (not shown in -h)")
    if "-h" in sys.argv:
        return parser
    advanced.add_argument(
        "--threads",
        dest="threads",
        default=get_num_threads(),
        type=int,
        help=f"Number of threads to use (defaults to number of hardware threads: "
             f"{get_num_threads()})",
    )
    advanced.add_argument(
        "--patch_size",
        type=patch_size_type,
        dest="patch_size",
        default=32,
        help="Patch size to use in calculating the partial volumes (default: 32).",
    )
    advanced.add_argument(
        "--empty",
        action="store_true",
        dest="empty",
        help="Keep ids for the table that do not exist in the segmentation "
             "(default: drop).",
    )
    advanced = add_arguments(advanced, ["device", "sid", "sd", "allow_root"])
    advanced.add_argument(
        "--lut",
        type=Path,
        metavar="lut",
        dest="lut",
        help="Path and name of LUT to use.",
    )
    advanced.add_argument(
        "--legacy_freesurfer",
        action="store_true",
        dest="legacy_freesurfer",
        help="Reproduce FreeSurfer mri_segstats numbers (default: off). \n"
             "Please note, that exact agreement of numbers cannot be guaranteed, "
             "because the condition number of FreeSurfers algorithm (mri_segstats) "
             "combined with the fact that mri_segstats uses 'float' to measure the "
             "partial volume corrected volume. This yields differences of more than "
             "60mm3 or 0.1%% in large structures. This uniquely impacts highres images "
             "with more voxels (on the boundary) and smaller voxel sizes (volume per "
             "voxel).",
    )
    # Additional info:
    # Changing the data type in mri_segstats to double can reduce this difference to
    # nearly zero.
    # mri_segstats has two operations affecting a bad condition number:
    # 1. pv = (val - mean_nbr) / (mean_label - mean_nbr)
    # 2. volume += vox_vol * pv
    #    This is further affected by the small vox_vol (volume per voxel) of highres
    #    images (0.7iso -> 0.343)
    # Their effects stack and can result in differences of more than 60mm3 or 0.1% in
    # a comparison between double and single-precision evaluations.
    advanced.add_argument(
        "--mixing_coeff",
        type=Path,
        dest="mix_coeff",
        default="",
        help="Save the mixing coefficients (default: off).",
    )
    advanced.add_argument(
        "--alternate_labels",
        type=Path,
        dest="nbr",
        default="",
        help="Save the alternate labels (default: off).",
    )
    advanced.add_argument(
        "--alternate_mixing_coeff",
        type=Path,
        dest="nbr_mix_coeff",
        default="",
        help="Save the alternate labels' mixing coefficients (default: off).",
    )
    advanced.add_argument(
        "--seg_means",
        type=Path,
        dest="seg_means",
        default="",
        help="Save the segmentation labels' means (default: off).",
    )
    advanced.add_argument(
        "--alternate_means",
        type=Path,
        dest="nbr_means",
        default="",
        help="Save the alternate labels' means (default: off).",
    )
    advanced.add_argument(
        "--volume_precision",
        type=id_type,
        dest="volume_precision",
        default="3",
        help="Number of digits after dot in summary stats file (default: 3). Note, "
             "--legacy_freesurfer sets this to 1.",
    )
    advanced.add_argument(
        "--norm_name",
        type=str,
        dest="norm_name",
        default="norm",
        help="Option to change the name of the in volume (default: norm)."
    )
    advanced.add_argument(
        "--norm_unit",
        type=str,
        dest="norm_unit",
        default="MR",
        help="Option to change the unit of the in volume (default: MR)."
    )
    return parser


def empty(__arg: Any) -> bool:
    """
    Checks if the argument is an empty list (or None).
    """
    return __arg is None or (isinstance(__arg, Sized) and len(__arg) == 0)


def add_measure_parser(subparser_callback: SubparserCallback) -> None:
    """
    Add a parser that supports adding measures to the parameters.
    """
    measure_parser = subparser_callback(
        "measures",
        usage="python segstats.py (...) measures [optional arguments]",
        argument_default="measures",
        help="Configures options to measures",
        description="Options to configure measures",
        formatter_class=HelpFormatter,
        add_help=False,
    )
    add_two_help_messages(measure_parser)
    measure_parser.add_argument(
        "--compute",
        type=str,
        nargs="+",
        default=[],
        dest="computed_measures",
        help="Additional Measures to compute based on imported/computed measures:<br>"
             "Cortex, CerebralWhiteMatter, SubCortGray, TotalGray, "
             "BrainSegVol-to-eTIV, MaskVol-to-eTIV, SurfaceHoles, "
             "EstimatedTotalIntraCranialVol",
    )
    measure_parser.add_argument(
        '--import',
        type=str,
        nargs="+",
        default=[],
        dest="imported_measures",
        help="Additional Measures to import from the measurefile.<br>"
             "Example measures ('all' to import all measures in the measurefile):<br>"
             "BrainSeg, BrainSegNotVent, SupraTentorial, SupraTentorialNotVent, "
             "SubCortGray, lhCortex, rhCortex, Cortex, TotalGray, "
             "lhCerebralWhiteMatter, rhCerebralWhiteMatter, CerebralWhiteMatter, Mask, "
             "SupraTentorialNotVentVox, BrainSegNotVentSurf, VentricleChoroidVol, "
             "BrainSegVol-to-eTIV, MaskVol-to-eTIV, lhSurfaceHoles, rhSurfaceHoles, "
             "SurfaceHoles, EstimatedTotalIntraCranialVol<br>"
             "Note, 'all' will always be overwritten by any explicitly mentioned "
             "measures.",
    )
    measure_parser.add_argument(
        "--file",
        type=Path,
        dest="measurefile",
        default="brainvol.stats",
        help="Default file to read measures (--import ...) from. If the path is "
             "relative, it is interpreted as relative to subjects_dir/subject_id from"
             "--sd and --subject_id.",
    )
    measure_parser.add_argument(
        "--from_seg",
        type=Path,
        dest="aseg_replace",
        default=None,
        help="Replace the default segfile to compute measures from by -i/--segfile. "
             "This will default to 'mri/aseg.mgz' for --legacy_freesurfer and to the "
             "value of -i/--segfile otherwise."
    )


def add_two_help_messages(parser: argparse.ArgumentParser) -> None:
    """
    Adds separate help flags -h and --help to the parser for simple and detailed help.
    Both trigger the help action.

    Parameters
    ----------
    parser: argparse.ArgumentParser
        parser to add the flags to
    """
    def this_msg(msg: str, flag: str) -> str:
        import sys
        return f"{msg} (this message)" if flag in sys.argv else msg
    parser.add_argument(
        "-h", action="help",
        help=this_msg("show a short help message and exit", "-h"))
    parser.add_argument(
        "--help", action="help",
        help=this_msg("show a long, detailed help message and exit", "--help"))


def _check_arg_path(
    __args: argparse.Namespace,
    __attr: str,
    subjects_dir: Path | None,
    subject_id: str | None,
    allow_subject_dir: bool = True,
    require_exist: bool = True,
) -> Path:
    """
    Check an argument that is supposed to be a Path object and finding the absolute
    path, which can be derived from the subject_dir.

    Parameters
    ----------
    __args : argparse.Namespace
        The arguments object.
    __attr: str
        The name of the attribute in the Namespace object.
    allow_subject_dir : bool, optional
        Whether relative paths are supposed to be understood with respect to
        subjects_dir / subject_id (default: True).
    require_exist : bool, optional
        Raise a ValueError, if the indicated file does not exist (default: True).

    Returns
    -------
    Path
        The resulting Path object.

    Raises
    ------
    ValueError
        If attribute does not exist, is not a Path (or convertible to a Path), or if
        the file does not exist, but reuire_exist is True.
    """
    if (_attr_val := getattr(__args, __attr), None) is None:
        raise ValueError(f"No {__attr} passed.")
    if isinstance(_attr_val, str):
        _attr_val = Path(_attr_val)
    elif not isinstance(_attr_val, Path):
        raise ValueError(f"{_attr_val} is not a Path object.")
    if allow_subject_dir and not _attr_val.is_absolute():
        if isinstance(subjects_dir, Path) and subject_id is not None:
            _attr_val = subjects_dir / subject_id / _attr_val
    if require_exist and not _attr_val.exists():
        raise ValueError(f"Path {_attr_val} did not exist for {__attr}.")
    return _attr_val


def _check_arg_defined(attr: str, /, args: argparse.Namespace) -> bool:
    """
    Check whether the attribute attr is defined in args.

    Parameters
    ----------
    attr: str
        The name of the attribute.
    args: argparse.Namespace
        The argument container object.

    Returns
    -------
    bool
        Whether the argument is defined (not None, not an empty container/str).
    """
    value = getattr(args, attr, None)
    return not (value is None or empty(value))


def check_shape_affine(
    img1: "nib.analyze.SpatialImage",
    img2: "nib.analyze.SpatialImage",
    name1: str,
    name2: str,
) -> None:
    """
    Check whether the shape and affine of

    Parameters
    ----------
    img1 : nibabel.SpatialImage
        Image 1.
    img2 : nibabel.SpatialImage
        Image 2.
    name1 : str
        Name of image 1.
    name2 : str
        Name of image 2.

    Raises
    -------
    RuntimeError
        If shapes or affines are not the same.
    """
    if img1.shape != img2.shape or not np.allclose(img1.affine, img2.affine):
        raise RuntimeError(
            f"The shapes or affines of the {name1} and the {name2} image are not "
            f"similar, both must be the same!"
        )


def parse_files(
    args: argparse.Namespace,
    subjects_dir: Path | str | None = None,
    subject_id: str | None = None,
    require_measurefile: bool = False,
) -> tuple[Path, Path | None, Path | None, Path, Path | None]:
    """
    Parse and read paths of files.

    Parameters
    ----------
    args : argparse.Namespace
        Parameters object from make_arguments.
    subjects_dir : Path, str, optional
        Path to SUBJECTS_DIR, where subject directories are.
    subject_id : str, optional
        The subject_id string.
    require_measurefile: bool, default=False
        require the measurefile to exist.

    Returns
    -------
    segfile : Path
        Path to the segmentation file, most likely an absolute path.
    pvfile : Path, None
        Path to the pvfile file, most likely an absolute path.
    normfile : Path, None
        Path to the norm file, most likely an absolute path, or None if not passed.
    segstatsfile : Path
        Path to the output segstats file, most likely an absolute path.
    measurefile : Path, None
        Path to the measure file, most likely an absolute path, not None is not passed.

    Raises
    ------
    ValueError
        If there is a necessary parameter missing or invalid.
    """
    if subjects_dir is not None:
        subjects_dir = Path(subjects_dir)
    check_arg_path = partial(
        _check_arg_path, subjects_dir=subjects_dir, subject_id=subject_id
    )
    segfile = check_arg_path(args, "segfile")
    not_has_arg = partial(_check_arg_defined, args=args)
    if not any(map(not_has_arg, ("normfile", "pvfile"))):
        pvfile = None
        normfile = None
    elif getattr(args, "normfile", None) is None:
        pvfile = check_arg_path(args, "pvfile")
        normfile = None
    else:
        normfile = check_arg_path(args, "normfile")
        if getattr(args, "pvfile", None) is None:
            pvfile = normfile
        else:
            pvfile = check_arg_path(args, "pvfile")

    segstatsfile = check_arg_path(args, "segstatsfile", require_exist=False)
    if not segstatsfile.is_absolute():
        raise ValueError("segstatsfile must be an absolute path!")

    if getattr(args, "measurefile", None) is None:
        measurefile = None
    else:
        measurefile = check_arg_path(
            args,
            "measurefile",
            require_exist=require_measurefile,
        )

    return segfile, pvfile, normfile, segstatsfile, measurefile


def infer_labels_excludeid(
    args: argparse.Namespace,
    lut: "pd.DataFrame",
    data: "npt.NDArray[int]",
) -> tuple["npt.NDArray[int]", list[int]]:
    """
    Infer the labels and excluded ids from command line arguments, the lookup table, or
    the segmentation image.

    Parameters
    ----------
    args : argparse.Namespace
        The commandline arguments object.
    lut : pd.DataFrame
        The ColorLUT lookup table object, e.g. FreeSurferColorLUT.
    data : npt.NDArray[int]
        The segmentation array.

    Returns
    -------
    labels : npt.NDArray[int]
        The array of all labels to calculate partial volumes for.
    exclude_id : list[int]
        A list of labels exlicitly excluded from the output table.
    """
    explicit_ids = False
    if __ids := getattr(args, "ids", None):
        labels = np.asarray(__ids)
        explicit_ids = True
    elif lut is not None:
        labels = lut["ID"]  # the column ID contains all ids
    else:
        labels = np.unique(data)

    # filter for excludeif entries
    exclude_id = []
    if _excl_id := getattr(args, "excludeid", None):
        exclude_id = list(_excl_id)
        # check whether
        if explicit_ids:
            _exclude = list(filter(lambda x: x in exclude_id, labels))
            excluded_expl_ids = np.asarray(_exclude)
            if excluded_expl_ids.size > 0:
                raise ValueError(
                    "Some IDs explicitly passed via --ids are also in the list of "
                    "ids to exclude (--excludeid)."
                )
        labels = np.asarray([x for x in labels if x not in exclude_id], dtype=int)
    return labels, exclude_id


def main(args: argparse.Namespace) -> Literal[0] | str:
    """
    Main segstats function, based on mri_segstats.

    Parameters
    ----------
    args : object
        Parameter object as defined by `make_arguments().parse_args()`

    Returns
    -------
    Literal[0], str
        Either as a successful return code or a string with an error message
    """
    from time import perf_counter_ns
    from FastSurferCNN.utils.common import assert_no_root
    from FastSurferCNN.utils.brainvolstats import Manager, read_volume_file, ImageTuple

    start = perf_counter_ns()
    getattr(args, "allow_root", False) or assert_no_root()

    subjects_dir = getattr(args, "out_dir", None)
    if subjects_dir is not None:
        subjects_dir = Path(subjects_dir)
    subject_id = str(getattr(args, "sid", None))
    legacy_freesurfer = bool(getattr(args, "legacy_freesurfer", False))
    manager_kwargs = {}

    # Check the file name parameters segfile, pvfile, normfile, segstatsfile, and
    # measurefile
    try:
        segfile, pvfile, normfile, segstatsfile, measurefile = parse_files(
            args,
            subjects_dir,
            subject_id,
            require_measurefile=not empty(getattr(args, "imported_measures", [])),
        )
        brainvolstats_only = pvfile is None
        if brainvolstats_only:
            print("No files are defined via -pv/--pvfile or -norm/--normfile:")
            print("Only computing brainvol stats in legacy mode.")
            manager_kwargs["compat"] = True
    except ValueError as e:
        return e.args[0]

    threads = getattr(args, "threads", 0)
    if threads <= 0:
        threads = get_num_threads()

    compute_threads = ThreadPoolExecutor(threads)

    # the manager object supports preloading of files (see below) for io parallelization
    # and calculates the measure
    manager = Manager(args, **manager_kwargs)
    from FastSurferCNN.data_loader.data_utils import read_classes_from_lut
    read_lut = manager.make_read_hook(read_classes_from_lut)
    if lut_file := getattr(args, "lut", None):
        read_lut(lut_file, blocking=False)
    # load these files in different threads to avoid waiting on IO
    # (not parallel due to GIL though)
    load_image = manager.make_read_hook(read_volume_file)
    preload_image = partial(load_image, blocking=False)
    preload_image(segfile)
    if normfile is not None:
        preload_image(normfile)
    if not brainvolstats_only:
        preload_image(pvfile)

    with manager.with_subject(subjects_dir, subject_id):
        try:
            _seg: ImageTuple = load_image(segfile, blocking=True)
            seg, seg_data = _seg
            pv_img, pv_data = None, None
            norm, norm_data = None, None

            # trigger preprocessing operations on the pvfile like --mul <factor>
            pv_preproc_future = None
            if not brainvolstats_only:
                _pv: ImageTuple = load_image(pvfile, blocking=True)
                pv_img, pv_data = _pv

                if not empty(pvfile_preproc := getattr(args, "pvfile_preproc", None)):
                    pv_preproc_future = compute_threads.submit(
                        preproc_image, pvfile_preproc, pv_data
                    )

                check_shape_affine(seg, pv_img, "segmentation", "pv_guide")
            if normfile is not None:
                _norm: ImageTuple = load_image(normfile, blocking=True)
                norm, norm_data = _norm
                check_shape_affine(seg, norm, "segmentation", "norm")

        except (IOError, RuntimeError, FileNotFoundError) as e:
            return e.args[0]

        lut: Optional[pd.DataFrame] = None
        if lut_file:
            try:
                lut = read_lut(lut_file)
                # manager.lut = lut
            except FileNotFoundError:
                return (
                    f"Could not find the ColorLUT in {lut_file}, make sure the --lut "
                    f"argument is valid."
                )
            except Exception as exception:
                return exception.args[0]
        try:
            # construct the list of labels to calculate PV for
            labels, exclude_id = infer_labels_excludeid(args, lut, seg_data)
        except ValueError as e:
            return e.args[0]

        if (_merged_labels := getattr(args, "merged_labels", None)) is None:
            _merged_labels: Sequence[Sequence[int]] = ()
        merged_labels, measure_labels = infer_merged_labels(
            manager,
            labels,
            merged_labels=_merged_labels,
            merge_labels_start=10000,
        )
        vox_vol = np.prod(seg.header.get_zooms()).item()
        # more args to pass to pv_calc
        kwargs = {
            "vox_vol": vox_vol,
            "legacy_freesurfer": legacy_freesurfer,
            "threads": compute_threads,
            "robust_percentage": getattr(args, "robust", None),
            "patch_size": getattr(args, "patch_size", 16),
            "merged_labels": merged_labels,
        }
        # more args to pass to write_segstatsfile
        write_kwargs = {
            "vox_vol": vox_vol,
            "legacy_freesurfer": legacy_freesurfer,
            "exclude": exclude_id,
            "segfile": segfile,
            "normfile": normfile,
            "lut": lut_file,
            "volume_precision": getattr(args, "volume_precision", "1"),
        }
    # ------
    # finished manager io here
    # ------
    manager.compute_non_derived_pv(compute_threads)

    if brainvolstats_only:
        # if we are not computing partial volume effects, do not perform pv_calc
        try:
            manager.wait_write_brainvolstats(segstatsfile)
        except RuntimeError as e:
            return e.args[0]
        print(f"Brain volume stats written to {segstatsfile}.")
        duration = (perf_counter_ns() - start) / 1e9
        print(f"Calculation took {duration:.2f} seconds using up to {threads} threads.")
        return 0

    if pv_preproc_future is not None:
        # wait for preprocessing options on pvfile
        pv_data = pv_preproc_future.result()

    names = ["nbr", "nbr_means", "seg_means", "mix_coeff", "nbr_mix_coeff"]
    save_maps = any(getattr(args, n, "") for n in names)
    out = pv_calc(seg_data, pv_data, norm_data, labels, return_maps=save_maps, **kwargs)

    _io_futures = []
    if save_maps:
        table, maps = out
        dtypes = [np.int16] + [np.float32] * 4
        for name, dtype in zip(names, dtypes):
            if not bool(file := getattr(args, name, "")) or file == Path():
                # skip "fullview"-files that are not defined
                continue
            print(f"Saving {name} to {file}...")
            from FastSurferCNN.data_loader.data_utils import save_image

            _header = seg.header.copy()
            _header.set_data_dtype(dtype)
            _io_futures.append(
                manager.executor.submit(
                    save_image,
                    _header,
                    seg.affine,
                    maps[name],
                    file,
                    dtype,
                ),
            )
        print("Done.")
    else:
        table: list[PVStats] = out

    if lut is not None:
        update_structnames(table, lut, merged_labels)

    dataframe = table_to_dataframe(
        table,
        bool(getattr(args, "empty", False)),
        must_keep_ids=merged_labels.keys(),
    )
    lines = format_parameters(SUBJECT_DIR=subjects_dir, subjectname=subject_id)

    # wait for computation of measures and return an error message if errors occur
    errors = list(manager.wait_compute())
    if not empty(errors):
        error_messages = ["Some errors occurred during measure computation:"]
        error_messages.extend(map(lambda e: f"{type(e).__name__}: {e.args[0]}", errors))
        return "\n - ".join(error_messages)
    dataframe = manager.update_pv_from_table(dataframe, measure_labels)
    lines.extend(manager.format_measures())

    write_statsfile(
        segstatsfile,
        dataframe,
        extra_header=lines,
        **write_kwargs,
    )
    print(f"Partial volume stats for {dataframe.shape[0]} labels written to "
          f"{segstatsfile}.")
    duration = (perf_counter_ns() - start) / 1e9
    print(f"Calculation took {duration:.2f} seconds using up to {threads} threads.")

    for _io_fut in _io_futures:
        if (e := _io_fut.exception()) is not None:
            logging.getLogger(__name__).exception(e)

    return 0


def infer_merged_labels(
        manager: "Manager",
        used_labels: Iterable[int],
        merged_labels: Sequence[Sequence[int]] = (),
        merge_labels_start: int = 0,
) -> tuple[dict[int, Sequence[int]], dict[int, Sequence[int]]]:
    """

    Parameters
    ----------
    manager : Manager
        The brainvolstats Manager object to get virtual labels.
    used_labels : Iterable[int]
        A list of labels at that are already in use.
    merged_labels : Sequence[Sequence[int]], default=()
        The list of merge labels (first value is SegId, then SegIds it sums across).
    merge_labels_start : int, default=0
        Start index to start at for finding multi-class merged label groups.

    Returns
    -------
    all_merged_labels : dict[int, Sequence[int]]
        The dictionary of all merged labels (via :class:`PVMeasure`s as well as
        `merged_labels`).
    """
    _merged_labels = {}
    if not empty(merged_labels):
        _merged_labels = {lab: vals for lab, *vals in merged_labels}
    all_labels = list(_merged_labels.keys()) + list(used_labels)
    _pv_merged_labels = manager.get_virtual_labels(
        i for i in range(merge_labels_start, np.iinfo(int).max) if i not in all_labels
    )

    all_merged_labels = _merged_labels.copy()
    all_merged_labels.update(_pv_merged_labels)
    return all_merged_labels, _pv_merged_labels


def table_to_dataframe(
        table: list[PVStats],
        report_empty: bool = True,
        must_keep_ids: Optional[Container[int]] = None,
) -> pd.DataFrame:
    """
    Convert the list of PVStats dictionaries into a dataframe.

    Parameters
    ----------
    table : list[PVStats]
        List of partial volume stats dictionaries.
    report_empty : bool, default=True
        Whether empty regions should be part of the dataframe.
    must_keep_ids : Container[int], optional
        Specifies a list of segids to never remove from the table.

    Returns
    -------
    pandas.DataFrame
        The DataFrame object of all columns and rows in table.
    """
    has_must_keep_ids = must_keep_ids and isinstance(must_keep_ids, Container)

    def must_keep_fn(x) -> bool:
        return x in must_keep_ids if has_must_keep_ids else False

    df = pd.DataFrame(table, index=np.arange(len(table)))
    if not report_empty:
        df = df[df["NVoxels"] != 0 | df["SegId"].map(must_keep_fn)]
    df = df.sort_values("SegId")
    df.index = np.arange(1, len(df) + 1)
    return df


def update_structnames(
    table: list[PVStats],
    lut: pd.DataFrame,
    merged_labels: Optional[dict[_IntType, Sequence[_IntType]]] = None
) -> None:
    """
    Update StructNames from `lut` and `merged_labels` in `table`.

    Parameters
    ----------
    table : list[PVStats]
        List of partial volume stats dictionaries.
    lut : pandas.DataFrame
        A pandas DataFrame object containing columns 'ID' and 'LabelName', which serves
        as a lookup table for the structure names.
    merged_labels : dict[int, Sequence[int]], optional
        The dictionary with merged labels.
    """
    # table is a list of dicts, so we can add the StructName to the dict
    for i in range(len(table)):
        lut_idx = lut["ID"] == table[i]["SegId"]
        if lut_idx.any():
            # get the label name from the lut, if it is in there
            table[i]["StructName"] = lut[lut_idx]["LabelName"].item()
        elif merged_labels is not None and table[i]["SegId"] in merged_labels.keys():
            # auto-generate a name for merged labels
            table[i]["StructName"] = "Merged-Label-" + str(table[i]["SegId"])
        else:
            # make the label unknown
            table[i]["StructName"] = "Unknown-Label"
    # lut_idx = {i: lut["ID"] == i for i in exclude_id}
    # _ids = [(i, lut_idx[i]) for i in exclude_id]


def format_parameters(**kwargs) -> list[str]:
    """
    Formats each keyword argument passed as a pair of key and value.

    Returns
    -------
    list[str]
        A list of one string per keyword arg formatted as a string.
    """
    return [f"{k} {v}" for k, v in kwargs.items() if v]


def write_statsfile(
    segstatsfile: Path | str,
    dataframe: pd.DataFrame,
    vox_vol: float,
    exclude: Optional[Sequence[int | str]] = None,
    segfile: Optional[Path | str] = None,
    normfile: Optional[Path | str] = None,
    pvfile: Optional[Path | str] = None,
    lut: Optional[Path | str] = None,
    report_empty: bool = False,
    extra_header: Sequence[str] = (),
    norm_name: str = "norm",
    norm_unit: str = "MR",
    volume_precision: str = "1",
    legacy_freesurfer: bool = False,
) -> None:
    """
    Write a segstatsfile very similar and compatible with mri_segstats output.

    Parameters
    ----------
    segstatsfile : Path, str
        Path to the output file.
    dataframe : pd.DataFrame
        Data to write into the file.
    vox_vol : float
        Voxel volume for the header.
    exclude : Sequence[Union[int, str]], optional
        Sequence of ids and class names that were excluded from the pv analysis
        (default: None).
    segfile : Path, str, optional
        Path to the segmentation file (default: empty).
    normfile : Path, str, optional
        Path to the bias-field corrected image (default: empty).
    pvfile : Path, str, optional
        Path to file used to compute the PV effects (default: empty).
    lut : Path, str, optional
        Path to the lookup table to find class names for label ids (default: empty).
    report_empty : bool, default=False
        Do not skip non-empty regions in the lut.
    extra_header : Sequence[str], default=()
        Sequence of additional lines to add to the header. The initial # and newline
        characters will be added. Should not include newline characters (expect at the
        end of strings).
    norm_name : str, default="norm"
        Name of the intensity image.
    norm_unit : str, default="MR"
        Unit of the intensity image.
    volume_precision : str, default="1"
        Number of digits after the comma for volume. Forced to 1 for legacy_freesurfer.
    legacy_freesurfer : bool, default=False
        Whether the script ran with the legacy freesurfer option.
    """
    import datetime

    volume_precision = "1" if legacy_freesurfer else volume_precision

    def _title(file: IO) -> None:
        """
        Write the file title to a file.
        """
        file.write("# Title Segmentation Statistics\n#\n")

    def _system_info(file: IO) -> None:
        """
        Write the call and system information comments of the header to a file.
        """
        import os
        import sys
        from FastSurferCNN.version import read_and_close_version
        file.write(
            "# generating_program segstats.py\n"
            "# FastSurfer_version " + read_and_close_version() + "\n"
            "# cmdline " + " ".join(sys.argv) + "\n"
        )
        if os.name == 'posix':
            file.write(
                f"# sysname  {os.uname().sysname}\n"
                f"# hostname {os.uname().nodename}\n"
                f"# machine  {os.uname().machine}\n"
            )
        else:
            from socket import gethostname
            file.write(
                f"# platform {sys.platform}\n"
                f"# hostname {gethostname()}\n"
            )
        from getpass import getuser

        try:
            file.write(f"# user       {getuser()}\n")
        except KeyError:
            file.write(f"# user       UNKNOWN\n")

    def _extra_header(file: IO, lines_extra_header: Iterable[str]) -> None:
        """
        Write the extra_header (including measures) to a file.
        """
        warn_msg_sent = False
        for i, line in enumerate(lines_extra_header):
            if line.endswith("\n"):
                line = line[:-1]
            if line.startswith("# "):
                line = line[2:]
            elif line.startswith("#"):
                line = line[1:]
            if "\n" in line:
                line = line.replace("\n", " ")
                from warnings import warn

                warn_msg_sent or warn(
                    f"extra_header[{i}] includes embedded newline characters. "
                    "Replacing all newline characters with <space>."
                )
                warn_msg_sent = True
            file.write(f"# {line}\n")

    def _file_annotation(file: IO, name: str, path_to_annotate: Optional[Path]) -> None:
        """
        Write the annotation to file/path to a file.
        """
        if path_to_annotate is not None:
            file.write(f"# {name} {path_to_annotate}\n")
            stat = path_to_annotate.stat()
            if stat.st_mtime:
                mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
                file.write(f"# {name}Timestamp {mtime:%Y/%m/%d %H:%M:%S}\n")

    def _extra_parameters(
        file: IO,
        _voxvol: float,
        _exclude: Sequence[int | str],
        _report_empty: bool = False,
        _lut: Optional[Path] = None,
        _leg_freesurfer: bool = False,
    ) -> None:
        """
        Write the comments of the table header to a file.
        """
        if _exclude is not None and len(_exclude) > 0:
            exclude_str = list(filter(lambda x: isinstance(x, str), _exclude))
            exclude_int = list(filter(lambda x: isinstance(x, int), _exclude))
            if len(exclude_str) > 0:
                excl_names = ', '.join(exclude_str)
                file.write(f"# Excluding {excl_names}\n")
            if len(exclude_int) > 0:
                file.write(f"# ExcludeSegId {' '.join(map(str, exclude_int))}\n")
        if _lut is not None and not _report_empty:
            file.write("# Only reporting non-empty segmentations\n")
        file.write("# compatibility with freesurfer's mri_segstats: " +
                   ("legacy" if _leg_freesurfer else "fixed") + "\n")
        file.write(f"# VoxelVolume_mm3 {_voxvol}\n")

    def _is_norm_column(name: str) -> bool:
        """Check whether the column `name` is a norm-column."""
        return name in ("Mean", "StdDev", "Min", "Max", "Range")

    def _column_name(name: str) -> str:
        """Convert the column name"""
        return norm_name + name if _is_norm_column(name) else name

    def _column_unit(name: str) -> str:
        if _is_norm_column(name):
            return norm_unit
        elif name == "Volume_mm3":
            return "mm^3"
        elif name == "NVoxels":
            return "unitless"
        return "NA"

    def _column_description(name: str) -> str:
        if _is_norm_column(name):
            return f"Intensity {_column_name(name)}"
        return {
            "Index": "Index", "SegId": "Segmentation Id", "NVoxels": "Number of Voxels",
            "Volume_mm3": "Volume", "StructName": "Structure Name"
        }.get(name, "Unknown Column")

    def _column_format(name: str) -> str:
        if _is_norm_column(name):
            return ".4f"
        elif name == "Volume_mm3":
            return f".{volume_precision}f"
        elif name in ("Index", "SegId", "NVoxels"):
            return "d"
        return "s"

    def _table_header(file: IO, _dataframe: pd.DataFrame) -> None:
        """Write the comments of the table header to a file."""
        columns = [col for col in COLUMNS if col in _dataframe.columns]
        for i, col in enumerate(columns):
            file.write(f"# TableCol {i + 1: 2d} ColHeader {_column_name(col)}\n"
                       f"# TableCol {i + 1: 2d} FieldName {_column_description(col)}\n"
                       f"# TableCol {i + 1: 2d} Units     {_column_unit(col)}\n")
        file.write(f"# NRows {len(_dataframe)}\n"
                   f"# NTableCols {len(columns)}\n")
        file.write("# ColHeaders  " + " ".join(map(_column_name, columns)) + "\n")

    def _table_body(file: IO, _dataframe: pd.DataFrame) -> None:
        """Write the volume stats from _dataframe to a file."""

        def fmt_field(code: str, data: pd.DataFrame) -> str:
            is_s, is_f, is_d = code[-1] == "s", code[-1] == "f", code[-1] == "d"
            filler = "<" if is_s else " >"
            if is_s:
                prec = int(data.dropna().map(len).max())
            else:
                prec = int(np.ceil(np.log10(data.max())))
            if is_f:
                prec += int(code[-2]) + 1
            return filler + str(prec) + code

        columns = [col for col in COLUMNS if col in _dataframe.columns]
        fmt = " ".join(
            ("{:" + fmt_field(_column_format(k), _dataframe[k]) + "}"
             for k in columns))
        for index, row in _dataframe.iterrows():
            data = [row[k] for k in columns]
            file.write(fmt.format(*data) + "\n")

    if not isinstance(segstatsfile, Path):
        segstatsfile = Path(segstatsfile)
    if normfile is not None and not isinstance(normfile, Path):
        normfile = Path(normfile)
    if segfile is not None and not isinstance(segfile, Path):
        segfile = Path(segfile)

    segstatsfile.parent.mkdir(exist_ok=True)
    with open(segstatsfile, "w") as fp:
        _title(fp)
        _system_info(fp)
        fp.write(f"# anatomy_type volume\n#\n")
        _extra_header(fp, extra_header)

        _file_annotation(fp, "SegVolFile", segfile)
        # Annot subject hemi annot
        # Label subject hemi LabelFile
        _file_annotation(fp, "ColorTable", lut)
        # ColorTableFromGCA
        # GCATimeStamp
        # masking applies to PV, not to the Measure Mask
        # MaskVolFile MaskThresh MaskSign MaskFrame MaskInvert
        _file_annotation(fp, "InVolFile", normfile)
        _file_annotation(fp, "PVVolFile", pvfile)
        _extra_parameters(fp, vox_vol, exclude, report_empty, lut, legacy_freesurfer)
        # add the Index column, if it is not in dataframe
        if "Index" not in dataframe.columns:
            index_df = pd.DataFrame.from_dict({"Index": dataframe.index})
            index_df.index = dataframe.index
            dataframe = index_df.join(dataframe)
        _table_header(fp, dataframe)
        _table_body(fp, dataframe)


def preproc_image(
        ops: Sequence[str],
        data: npt.NDArray[_NumberType]
) -> npt.NDArray[_NumberType]:
    """
    Apply preprocessing operations to data. Performs, --mul, --abs, --sqr, --sqrt
    operations in that order.

    Parameters
    ----------
    ops : Sequence[str]
        Sequence of operations to perform from 'mul=<factor>', 'div=<factor>', 'sqr',
        'abs', and 'sqrt'.
    data : np.ndarray
        Data to perform operations on.

    Returns
    -------
    np.ndarray
        Data after ops are performed on it.
    """
    mul_ops = np.asarray([o.startswith("mul=") or o.startswith("div=") for o in ops])
    if np.any(mul_ops):
        mul_op = ops[mul_ops.nonzero()[0][-1].item()]
        factor = float(mul_op[4:])
        data = (np.multiply if mul_op.startswith("mul=") else np.divide)(data, factor)
    if "abs" in ops:
        data = np.abs(data)
    if "sqr" in ops:
        data = data * data
    if "sqrt" in ops:
        data = np.sqrt(data)
    return data

def seg_borders(
    _array: _ArrayType,
    label: np.integer | bool,
    out: Optional[npt.NDArray[bool]] = None,
    cmp_dtype: npt.DTypeLike = "int8",
) -> npt.NDArray[bool]:
    """
    Handle to fast 6-connected border computation.

    Parameters
    ----------
    _array: numpy.ndarray
        Image to compute borders from, typically either a label image or a binary mask.
    label: int, bool
        Which classes to consider for border computation (True/False for binary mask).
    out: nt.NDArray[bool], optional
        The array for inplace computation.
    cmp_dtype: npt.DTypeLike, default=int8
        The data type to use for border laplace computation.

    Returns
    -------
    npt.NDArray[bool]
        A binary mask with border voxels as True.
    """
    # binarize
    bin_array: npt.NDArray[bool]
    bin_array = _array if np.issubdtype(_array.dtype, bool) else np.equal(_array, label)
    # scipy laplace is about 20% faster than skimage laplace on cpu
    from scipy.ndimage import laplace

    if np.issubdtype(cmp_dtype, bool):
        laplace_data = laplace(bin_array).astype(bool)
        if out is not None:
            out[:] = laplace_data
            laplace_data = out
        return laplace_data
    else:
        zeros = np.asarray(0., dtype=cmp_dtype)
        # laplace
        laplace_data = laplace(bin_array.astype(cmp_dtype))
        return np.not_equal(laplace_data, zeros, out=out)


def borders(
    _array: _ArrayType,
    labels: Iterable[np.integer] | bool,
    max_label: Optional[np.integer] = None,
    six_connected: bool = True,
    out: Optional[npt.NDArray[bool]] = None,
) -> npt.NDArray[bool]:
    """
    Handle to fast border computation.

    This is an efficient implementation, for multiple/many classes between which borders
    should be computed.

    Parameters
    ----------
    _array : _ArrayType
        Input labeled image or binary image.
    labels : Iterable[np.int], bool
        List of labels for which borders will be computed.
        If labels is True, _array is treated as a binary mask.
    max_label : np.int, optional
        The maximum label ot consider. If None, the maximum label in the array is used.
    six_connected : bool, default=True
        If True, 6-connected borders (must share a face) are computed,
        otherwise 26-connected borders (must share a vertex) are computed.
    out : npt.NDArray[bool], optional
        Output array to store the computed borders.

    Returns
    -------
    npt.NDArray[bool]
        Binary mask of border voxels.

    Raises
    ------
    ValueError
        If labels does not fit to _array (binary mask and integer and vice-versa).
    """
    dim = _array.ndim
    _shape_plus2 = [s + 2 for s in _array.shape]

    if labels is True:  # already binarized
        if not np.issubdtype(_array, bool):
            raise ValueError("If labels is True, the array should be boolean.")
        cmp = np.logical_xor
    else:
        if np.issubdtype(_array, bool):
            raise ValueError(
                "If labels is a list/iterable, the array should not be boolean."
            )

        def cmp(a, b):
            return a == b

        if max_label is None:
            max_label = _array.max().item()
        lookup = np.zeros((max_label + 1,), dtype=_array.dtype)
        # filter labels from labels that are bigger than max_label
        labels = list(filter(lambda x: x <= max_label, labels))
        if 0 not in labels:
            labels = [0] + labels
        lookup[labels] = np.arange(len(labels), dtype=lookup.dtype)
        _array = lookup[_array]
    logical_or = np.logical_or
    # pad array by 1 voxel of zeros all around
    padded = np.pad(_array, 1)

    if six_connected:
        def indexer(axis: int, is_mid: bool) -> tuple[SlicingTuple, SlicingTuple]:
            full_slice = (slice(1, -1),) if is_mid else (slice(None),)
            more_axes = dim - axis - 1
            return ((full_slice * axis + (slice(0, -1),) + full_slice * more_axes),
                    (full_slice * axis + (slice(1, None),) + full_slice * more_axes))

        # compare the [padded] image/array in all directions, x, y, z...
        # ([0], 0, 2, 2, 2, [0]) ==> (False, True, False, False, True)  for each dim
        # is_mid=True: drops padded values in unaffected axes
        indexes = (indexer(i, is_mid=True) for i in range(dim))
        nbr_same = [cmp(padded[i], padded[j]) for i, j in indexes]
        # merge neighbors so each border is 2 thick (left and right of change)
        # (False, True, False, False, True) ==>
        # ((False, True), (True, False), (False, False), (False, True))  for each dim
        # is_mid=False: padded values already dropped
        indexes = (indexer(i, is_mid=False) for i in range(dim))
        nbr_same = [(nbr_[i], nbr_[j]) for (i, j), nbr_ in zip(indexes, nbr_same)]
        from itertools import chain
        nbr_same = list(chain.from_iterable(nbr_same))
    else:
        # all indexes of the neighbors: ((0, 0, 0), (0, 0, 1) ... (2, 2, 2))
        ndindexes = tuple(np.ndindex((3,) * dim))

        def nbr_i(__array: _ArrayType, neighbor_index: int) -> _ArrayType:
            """Assuming a padded array __array, returns just the neighbor_index-th
            neighbors throughout the array."""
            # sample from 1d neighbor index to ndindex
            nbr_ndid = ndindexes[neighbor_index]  # e.g. (1, 0, 2)
            slice_ndindex = tuple(slice(o, None if o == 2 else o - 3) for o in nbr_ndid)
            return __array[slice_ndindex]

        # compare the array (center point) with all neighboring voxels
        # neighbor samples the neighboring voxel in the padded array
        nbr_same = [cmp(_array, nbr_i(padded, i)) for i in range(3**dim) if i != 2**dim]

    # reduce the per-direction/per-neighbor binary arrays into one array
    return np.logical_or.reduce(nbr_same, out=out)


def pad_slicer(
    slicer: Sequence[slice],
    whalf: int,
    img_size: np.ndarray | Sequence[float],
) -> tuple[SlicingTuple, SlicingTuple]:
    """
    Create two slicing tuples for indexing ndarrays/tensors that 'grow' and
    re-'ungrow' the patch `patch` by `whalf` (also considering the image shape).

    Parameters
    ----------
    slicer : Sequence[slice]
        Input slicing tuple.
    whalf : int
        How much to pad/grow the slicing tuple all around.
    img_size : np.ndarray, Sequence[float]
        Shape of the image.

    Returns
    -------
    SlicingTuple
        Tuple of slice-objects to go from image to padded patch.
    SlicingTuple
        Tuple of slice-objects to go from padded patch to patch.

    """
    # patch start/stop
    _patch = np.asarray([(s.start, s.stop) for s in slicer])
    start, stop = _patch.T

    # grown patch start/stop
    _start, _stop = np.maximum(0, start - whalf), np.minimum(stop + whalf, img_size)

    def _slice(start_end: npt.NDArray[int]) -> slice:
        _start, _end = start_end
        return slice(_start.item(), None if _end.item() == 0 else _end.item())
    # make grown patch and grown patch to patch
    padded_slicer = tuple(slice(s.item(), e.item()) for s, e in zip(_start, _stop))
    unpadded_slicer = tuple(map(_slice, zip(start - _start, stop - _stop)))
    return padded_slicer, unpadded_slicer


def uniform_filter(
    data: _ArrayType,
    filter_size: int,
    fillval: float = 0.,
    slicer_patch: Optional[SlicingTuple] = None,
) -> _ArrayType:
    """
    Apply a uniform filter (with kernel size `filter_size`) to `input`.

    The uniform filter is normalized (weights add to one).

    Parameters
    ----------
    data : _ArrayType
        Data to perform uniform filter on.
    filter_size : int
        Size of the filter.
    fillval : float, default=0
        Value to fill around the image.
    slicer_patch : SlicingTuple, optional
        Sub_region of data to crop to (e.g. to undo the padding (default: full image).

    Returns
    -------
    _ArrayType
        The filtered data.

    """
    _patch = (slice(None),) if slicer_patch is None else slicer_patch
    data = data.astype(float)
    from scipy.ndimage import uniform_filter

    def _uniform_filter(_arr, out=None):
        uni_filt = uniform_filter(
            _arr,
            size=filter_size,
            mode="constant",
            cval=fillval,
            output=out,
        )
        return uni_filt[_patch]

    return _uniform_filter(data)


@overload
def pv_calc(
    seg: npt.NDArray[_IntType],
    pv_guide: np.ndarray,
    norm: np.ndarray,
    labels: npt.ArrayLike,
    patch_size: int = 32,
    vox_vol: float = 1.0,
    eps: float = 1e-6,
    robust_percentage: Optional[float] = None,
    merged_labels: Optional[VirtualLabel] = None,
    threads: int | Executor = -1,
    return_maps: False = False,
    legacy_freesurfer: bool = False,
) -> list[PVStats]:
    ...


@overload
def pv_calc(
    seg: npt.NDArray[_IntType],
    pv_guide: np.ndarray,
    norm: np.ndarray,
    labels: npt.ArrayLike,
    patch_size: int = 32,
    vox_vol: float = 1.0,
    eps: float = 1e-6,
    robust_percentage: Optional[float] = None,
    merged_labels: Optional[VirtualLabel] = None,
    threads: int | Executor = -1,
    return_maps: True = True,
    legacy_freesurfer: bool = False,
) -> tuple[list[PVStats], dict[str, dict[int, np.ndarray]]]:
    ...


def pv_calc(
    seg: npt.NDArray[_IntType],
    pv_guide: np.ndarray,
    norm: np.ndarray,
    labels: npt.ArrayLike,
    patch_size: int = 32,
    vox_vol: float = 1.0,
    eps: float = 1e-6,
    robust_percentage: float | None = None,
    merged_labels: VirtualLabel | None = None,
    threads: int | Executor = -1,
    return_maps: bool = False,
    legacy_freesurfer: bool = False,
) -> list[PVStats] | tuple[list[PVStats], dict[str, np.ndarray]]:
    """
    Compute volume effects.

    Parameters
    ----------
    seg : np.ndarray
        Segmentation array with segmentation labels.
    pv_guide : np.ndarray
        Image to use to calculate partial volume effects from.
    norm : np.ndarray
        Intensity image to use to calculate image statistics from.
    labels : array_like
        Which labels are of interest.
    patch_size : int, default=32
        Size of patches.
    vox_vol : float, default=1.0
        Volume per voxel.
    eps : float, default=1e-6
        Threshold for computation of equality.
    robust_percentage : float, optional
        Fraction for robust calculation of statistics.
    merged_labels : VirtualLabel, optional
        Defines labels to compute statistics for that are.
    threads : int, concurrent.futures.Executor, default=-1
        Number of parallel threads to use in calculation, alternatively an executor
        object.
    return_maps : bool, default=False
        Returns a dictionary containing the computed maps.
    legacy_freesurfer : bool, default=False
        Whether to use a freesurfer legacy compatibility mode to exactly replicate
        freesurfer.

    Returns
    -------
    pv_stats : list[PVStats]
        Table (list of dicts) with keys SegId, NVoxels, Volume_mm3, Mean, StdDev, Min,
        Max, and Range.
    maps : dict[str, np.ndarray], optional
        Only returned, if return_maps is True:
        A dictionary with the 5 meta-information pv-maps:
        nbr: An image of alternative labels that were considered instead of the voxel's
            label.
        nbr_means: The local mean intensity of the label nbr at the specific voxel.
        seg_means: The local mean intensity of the primary label at the specific voxel.
        mixing_coeff: The partial volume of the primary label at the location.
        nbr_mixing_coeff: The partial volume of the alternative (nbr) label at the
            location.

    """
    input_checker = {
        "seg": (seg, np.integer),
        "pv_guide": (pv_guide, np.number),
        "norm": (norm, np.number),
    }
    for name, (img, _type) in input_checker.items():
        if (img is not None and not isinstance(img, np.ndarray) or
                not np.issubdtype(img.dtype, _type)):
            raise TypeError(f"The {name} object is not a numpy.ndarray of {_type}.")
    _labels = np.asarray(labels)
    if not isinstance(labels, Sequence):
        labels = _labels.tolist()
    if not np.issubdtype(_labels.dtype, np.integer):
        raise TypeError("The labels list is not an arraylike of ints.")

    if seg.shape != pv_guide.shape:
        raise RuntimeError(f"The shapes of the segmentation and the pv_guide must "
                           f"be identical, but shapes are {seg.shape} and "
                           f"{pv_guide.shape}!")
    if seg.shape != norm.shape:
        raise RuntimeError(
            f"The shape of the segmentation and the norm must be identical, but shapes "
            f"are {seg.shape} and {norm.shape}!"
        )

    mins, maxes, voxel_counts, robust_voxel_counts = [{} for _ in range(4)]
    borders, sums, sums_2, volumes = [{} for _ in range(4)]

    if isinstance(merged_labels, dict) and len(merged_labels) > 0:
        _more_labels = list(merged_labels.values())
        all_labels = set(labels)
        all_labels |= reduce(set.union, _more_labels[1:], set(_more_labels[0]))
    else:
        all_labels = labels

    # initialize global_crop with the full image
    global_crop: SlicingTuple = tuple(slice(0, _shape) for _shape in seg.shape)
    # ignore all regions of the image that are background only
    if 0 not in all_labels:
        # crop global_crop to the data (plus one extra voxel)
        not_background = cast(npt.NDArray[bool], seg != 0)
        any_in_global, global_crop = crop_patch_to_mask(
            not_background,
            sub_patch=global_crop
        )
        # grow global_crop by one, so all border voxels are included
        global_crop = pad_slicer(global_crop, 1, seg.shape)[0]
        if not any_in_global:
            raise RuntimeError("Segmentation map only consists of background")

    global_stats_filled = partial(
        global_stats,
        norm=None if norm is None else norm[global_crop],
        seg=seg[global_crop],
        robust_percentage=robust_percentage,
    )

    if threads == 0:
        raise ValueError("Zero is not a valid number of threads.")
    elif isinstance(threads, int) and threads > 0:
        nthreads = threads
    elif isinstance(threads, (Executor, int)):
        nthreads: int = get_num_threads()
    else:
        raise TypeError("threads must be int or concurrent.futures.Executor object.")
    from math import ceil
    executor = ThreadPoolExecutor(nthreads) if isinstance(threads, int) else threads
    map_kwargs = {"chunksize": 1 if nthreads < 0 else ceil(len(labels) / nthreads)}

    global_stats_future = executor.map(global_stats_filled, all_labels, **map_kwargs)

    if return_maps:
        from concurrent.futures import ProcessPoolExecutor
        if isinstance(executor, ProcessPoolExecutor):
            raise NotImplementedError(
                "The ProcessPoolExecutor is not compatible with return_maps=True!"
            )
        full_nbr_label = np.zeros(seg.shape, dtype=seg.dtype)
        full_nbr_mean = np.zeros(pv_guide.shape, dtype=float)
        full_seg_mean = np.zeros(pv_guide.shape, dtype=float)
        full_pv = np.ones(pv_guide.shape, dtype=float)
        full_ipv = np.zeros(pv_guide.shape, dtype=float)
    else:
        full_nbr_label, full_seg_mean, full_nbr_mean, full_pv, full_ipv = [None] * 5

    for lab, data in global_stats_future:
        if data[0] != 0:
            voxel_counts[lab], robust_voxel_counts[lab] = data[:2]
            mins[lab], maxes[lab], sums[lab], sums_2[lab] = data[2:-2]
            volumes[lab], borders[lab] = data[-2] * vox_vol, data[-1]

    # un_global_crop border here
    any_border = np.any(list(borders.values()), axis=0)
    pad_width = np.asarray(
        [(slc.start, shp - slc.stop) for slc, shp in zip(global_crop, seg.shape)],
        dtype=int)
    any_border = np.pad(any_border, pad_width)
    if not np.array_equal(any_border.shape, seg.shape):
        raise RuntimeError("border and seg_array do not have same shape.")

    # iterate through patches of the image
    patch_iters = [range(slc.start, slc.stop, patch_size) for slc in global_crop]
    # 4 chunks per core
    num_valid_labels = len(voxel_counts)
    map_kwargs["chunksize"] = np.ceil(num_valid_labels / nthreads / 4).item()
    patch_filter_func = partial(patch_filter, mask=any_border,
                                global_crop=global_crop, patch_size=patch_size)
    _patches = executor.map(patch_filter_func, product(*patch_iters), **map_kwargs)
    patches = (patch for has_pv_vox, patch in _patches if has_pv_vox)

    patchwise_pv_calc_func = partial(
        pv_calc_patch,
        global_crop=global_crop,
        borders=borders,
        border=any_border,
        seg=seg,
        pv_guide=pv_guide,
        full_nbr_label=full_nbr_label,
        full_seg_mean=full_seg_mean,
        full_pv=full_pv,
        full_ipv=full_ipv,
        full_nbr_mean=full_nbr_mean,
        eps=eps,
        legacy_freesurfer=legacy_freesurfer,
    )
    for vols in executor.map(patchwise_pv_calc_func, patches, **map_kwargs):
        for lab in volumes.keys():
            volumes[lab] += vols.get(lab, 0.0) * vox_vol

    # ColHeaders: Index SegId NVoxels Volume_mm3 StructName Mean StdDev ...
    # Min Max Range
    def prep_dict(lab: int):
        nvox = voxel_counts.get(lab, 0)
        vol = volumes.get(lab, 0.)
        return {"SegId": lab, "NVoxels": nvox, "Volume_mm3": vol}

    table = list(map(prep_dict, labels))
    if norm is not None:
        robust_vc_it = robust_voxel_counts.items()
        means = {lab: sums.get(lab, 0.) / cnt for lab, cnt in robust_vc_it if cnt > eps}

        def get_std(lab: _IntType, nvox: int) -> float:
            # *std = sqrt((sum * (*mean) - 2 * (*mean) * sum + sum2) / (nvoxels - 1));
            return np.sqrt((sums_2[lab] - means[lab] * sums[lab]) / (nvox - 1))

        stds = {lab: get_std(lab, nvox) for lab, nvox in robust_vc_it if nvox > eps}

        for lab, this in zip(labels, table):
            this.update(
                Mean=means.get(lab, 0.0),
                StdDev=stds.get(lab, 0.0),
                Min=mins.get(lab, 0.0),
                Max=maxes.get(lab, 0.0),
                Range=maxes.get(lab, 0.0) - mins.get(lab, 0.0),
            )
    if merged_labels is not None:
        table.extend(calculate_merged_labels(
            merged_labels,
            voxel_counts,
            robust_voxel_counts,
            volumes,
            mins,
            maxes,
            sums,
            sums_2,
            eps,
        ))

    if return_maps:
        maps = {
            "nbr": full_nbr_label,
            "seg_means": full_seg_mean,
            "nbr_means": full_nbr_mean,
            "mixing_coeff": full_pv,
            "nbr_mixing_coeff": full_ipv,
        }
        return table, maps
    return table


def calculate_merged_labels(
        merged_labels: VirtualLabel,
        voxel_counts: dict[_IntType, int],
        robust_voxel_counts: dict[_IntType, int],
        volumes: dict[_IntType, float],
        mins: dict[_IntType, float] | None = None,
        maxes: dict[_IntType, float] | None = None,
        sums: dict[_IntType, float] | None = None,
        sums_of_squares: dict[_IntType, float] | None = None,
        eps: float = 1e-6,
) -> list[PVStats]:
    """
    Calculate the statistics for meta-labels, i.e. labels based on other labels
    (`merge_labels`). Add respective items to `table`.

    Parameters
    ----------
    merged_labels : VirtualLabel
        A dictionary of key 'merged id' to value list of ids it references.
    voxel_counts : dict[int, int]
        A dict of voxel counts for labels in the image/referenced in `merged_labels`.
    robust_voxel_counts : dict[int, int]
        A dict of the robust number of voxels referenced in `merged_labels`.
    volumes : dict[int, float]
        A dict of the volumes associated with each label.
    mins : dict[int, float], optional
        A dict of the minimum intensity associated with each label.
    maxes : dict[int, float], optional
        A dict of the minimum intensity associated with each label.
    sums : dict[int, float], optional
        A dict of the sums of voxel intensities associated with each label.
    sums_of_squares : dict[int, float], optional
        A dict of the sums of squares of voxel intensities associated with each label.
    eps : float, default=1e-6
        An epsilon value for numeric stability.

    Yields
    ------
    PVStats
        A dictionary per entry in `merged_labels`.
    """
    def num_robust_voxels(lab):
        return robust_voxel_counts.get(lab, 0)

    def aggregate(source, merge_labels, f: Callable[..., np.ndarray] = np.sum):
        """aggregate labels `merge_labels` from `source` with function `f`"""
        _data = [source.get(l, 0) for l in merge_labels if num_robust_voxels(l) > eps]
        return f(_data).item()

    def aggregate_std(sums, sums2, merge_labels, nvox):
        """aggregate std of labels `merge_labels` from `source`"""
        s2 = [(s := sums.get(l, 0)) * s / r for l in group
              if (r := num_robust_voxels(l)) > eps]
        return np.sqrt((aggregate(sums2, merge_labels) - np.sum(s2)) / nvox).item()

    for lab, group in merged_labels.items():
        stats = {"SegId": lab}
        if all(l not in robust_voxel_counts for l in group):
            logging.getLogger(__name__).warning(
                f"None of the labels {group} for merged label {lab} exist in the "
                f"segmentation."
            )
            stats.update(NVoxels=0, Volume_mm3=0.0)
            for k, v in {"Min": mins, "Max": maxes, "Mean": sums}.items():
                if v is not None:
                    stats[k] = 0.
            if all(v is not None for v in (mins, maxes)):
                stats["Range"] = 0.
            if all(v is not None for v in (sums, sums_of_squares)):
                stats["StdDev"] = 0.
        else:
            num_voxels = aggregate(voxel_counts, group)
            stats.update(NVoxels=num_voxels, Volume_mm3=aggregate(volumes, group))
            if mins is not None:
                stats["Min"] = aggregate(mins, group, np.min)
            if maxes is not None:
                stats["Max"] = aggregate(maxes, group, np.max)
                if "Min" in stats:
                    stats["Range"] = stats["Max"] - stats["Min"]
            if sums is not None:
                stats["Mean"] = aggregate(sums, group) / num_voxels
                if sums_of_squares is not None:
                    stats["StdDev"] = aggregate_std(
                        sums,
                        sums_of_squares,
                        group,
                        num_voxels - 1,
                    )
        yield stats


def global_stats(
    lab: _IntType,
    norm: npt.NDArray[_NumberType] | None,
    seg: npt.NDArray[_IntType],
    out: Optional[npt.NDArray[bool]] = None,
    robust_percentage: Optional[float] = None,
) -> tuple[_IntType, _GlobalStats]:
    """
    Compute Label, Number of voxels, 'robust' number of voxels, norm minimum, maximum,
    sum, sum of squares and 6-connected border of label lab (out references the border).

    Parameters
    ----------
    lab : _IntType
        Label to compute statistics for.
    norm : npt.NDArray[_NumberType], optional
        The intensity image (default: None, do not compute intensity stats such as
        normMin, normMax, etc.).
    seg : npt.NDArray[_IntType]
        The segmentation image.
    out : npt.NDArray[bool], optional
        Output array to store the computed borders.
    robust_percentage : float, optional
        A robustness percentile to compute the statistics with (default: None/off = 1).

    Returns
    -------
    label : int
        The label the stats belong to (input).
    stats : _GlobalStats
        A tuple of number_of_voxels, number_of_within_robustness_thresholds,
        minimum_intensity, maximum_intensity, sum_of_intensities,
        sum_of_intensity_squares, and border with respect to the label.

    """
    label_mask = cast(npt.NDArray[bool], seg == lab)
    if norm is None:
        nvoxels = int(label_mask.sum())
        return lab, (nvoxels, nvoxels, None, None, None, None, 0., out)

    data_dtype = int if np.issubdtype(norm.dtype, np.integer) else float
    data = norm[label_mask].astype(data_dtype)
    nvoxels: int = data.shape[0]
    # if lab is not in the image at all
    if nvoxels == 0:
        return lab, (0, 0, None, None, None, None, 0., out)
    # compute/update the border
    if out is None:
        out = seg_borders(label_mask, True, cmp_dtype="int8").astype(bool)
    else:
        out[:] = seg_borders(label_mask, True, cmp_dtype="int").astype(bool)

    if robust_percentage is not None:
        data = np.sort(data)
        sym_drop_samples = int((1 - robust_percentage / 2) * nvoxels)
        data = data[sym_drop_samples:-sym_drop_samples]
        _min: _NumberType = data[0].item()
        _max: _NumberType = data[-1].item()
        __voxel_count = nvoxels - 2 * sym_drop_samples
    else:
        _min = data.min().item()
        _max = data.max().item()
        __voxel_count = nvoxels
    _sum: float = data.sum().item()
    sum_2: float = (data * data).sum().item()
    # this is independent of the robustness criterium
    _volume_mask = np.logical_and(label_mask, np.logical_not(out))
    volume: float = np.sum(_volume_mask).astype(float).item()
    return lab, (nvoxels, __voxel_count, _min, _max, _sum, sum_2, volume, out)


def patch_filter(
    patch_corner: tuple[int, int, int],
    mask: npt.NDArray[bool],
    global_crop: SlicingTuple,
    patch_size: int = 32,
) -> tuple[bool, SlicingSequence]:
    """
    Return, whether there are mask-True voxels in the patch starting at pos with size
    patch_size and the resulting patch shrunk to mask-True regions.

    Parameters
    ----------
    patch_corner : tuple[int, int, int]
        The top left corner of the patch.
    mask : npt.NDArray[bool]
        The mask of interest in the patch.
    global_crop : SlicingTuple
        A image-wide slicing mask to constrain the 'search space'.
    patch_size : int, default=32
        The size of the patch.

    Returns
    -------
    bool
        Whether there is any data in the patch at all.
    SlicingSequence
        Sequence of slice objects that describe patches with patch_corner and patch_size.
    """

    def _slice(patch_start, _patch_size, image_stop):
        return slice(patch_start, min(patch_start + _patch_size, image_stop))

    # create slices for current patch context (constrained by the global_crop)
    patch = [_slice(pc, patch_size, s.stop) for pc, s in zip(patch_corner, global_crop)]
    # crop patch context to the image content
    return crop_patch_to_mask(mask, sub_patch=patch)


def crop_patch_to_mask(
    mask: npt.NDArray[_NumberType],
    sub_patch: Optional[SlicingSequence] = None,
) -> tuple[bool, SlicingSequence]:
    """
    Crop the patch to regions of the mask that are non-zero.

    Assumes mask is always positive. Returns whether there is any mask>0 in the patch
    and a slicer/patch shrunk to mask>0 regions. The optional subpatch constrains this
    operation to the sub-region defined by a sequence of slicing operations.

    Parameters
    ----------
    mask : npt.NDArray[_NumberType]
        Mask to crop to.
    sub_patch : Optional[Sequence[slice]]
        Subregion of mask to only consider (default: full mask).

    Returns
    -------
    not_empty : bool
        Whether there is any voxel in the patch at all.
    target_slicer : SlicingSequence
        Sequence of slice-objects to extract the subregion of mask that is 'True'.
    """
    _target_slicer = []
    if sub_patch is None:
        slicer_context = tuple(slice(0, s) for s in mask.shape)
    else:
        slicer_context = tuple(sub_patch)
    slicer_in_patch_coords = tuple([slice(0, s.stop - s.start) for s in slicer_context])
    in_mask = True
    _mask = mask[slicer_context].sum(axis=2)
    for i, pat in enumerate(slicer_in_patch_coords):
        p = pat.start
        if in_mask:
            if i == 2:
                _mask = mask[slicer_context][tuple(_target_slicer)].sum(axis=0)
            slicer_ith_axis = tuple(_target_slicer[1:] if i != 2 else [])
            # can we shrink the patch context in i-th axis?
            pat_has_mask_in_axis = _mask[slicer_ith_axis].sum(axis=int(i == 0)) > 0
            # modify both the _patch_size and the coordinate p to shrink the patch
            pat_mask_indices = np.argwhere(pat_has_mask_in_axis)
            if pat_mask_indices.shape[0] == 0:
                # none in here
                _patch_size = 0
                in_mask = False
            else:
                # some in the mask, find first and distance to last
                offset = pat_mask_indices[0].item()
                p += offset
                _patch_size = pat_mask_indices[-1].item() - offset + 1
        else:
            _patch_size = 0
        _target_slicer.append(slice(p, p + _patch_size))

    def _move_slice(the_slice: slice, offset: int) -> slice:
        return slice(the_slice.start + offset, the_slice.stop + offset)

    target_slicer = [_move_slice(ts, sc.start) for ts, sc in zip(_target_slicer,
                                                                 slicer_context)]
    return _target_slicer[0].start != _target_slicer[0].stop, target_slicer


def pv_calc_patch(
    slicer_patch: SlicingTuple,
    global_crop: SlicingTuple,
    borders: dict[_IntType, npt.NDArray[bool]],
    seg: npt.NDArray[_IntType],
    pv_guide: npt.NDArray,
    border: npt.NDArray[bool],
    full_pv: Optional[npt.NDArray[float]] = None,
    full_ipv: Optional[npt.NDArray[float]] = None,
    full_nbr_label: Optional[npt.NDArray[_IntType]] = None,
    full_seg_mean: Optional[npt.NDArray[float]] = None,
    full_nbr_mean: Optional[npt.NDArray[float]] = None,
    eps: float = 1e-6,
    legacy_freesurfer: bool = False,
) -> dict[_IntType, float]:
    """
    Calculate PV for patch.

    If full* keyword arguments are passed, the function also fills in per voxel results
    for the respective voxels in the patch.

    Parameters
    ----------
    slicer_patch : SlicingTuple
        Tuple of slice-objects, with indexing origin at the image origin.
    global_crop : SlicingTuple
        Tuple of slice-objects, a global mask to limit computing to relevant parts of
        the image.
    borders : dict[int, npt.NDArray[bool]]
        Dictionary containing the borders for each label.
    seg : numpy.typing.NDArray[numpy.integer]
        The segmentation (full image) defining the labels.
    pv_guide : numpy.ndarray
        The (full) image with intensities to guide the PV calculation.
    border : npt.NDArray[bool]
        Binary mask, True, where a voxel is considered to be a border voxel.
    full_pv : npt.NDArray[float], optional
        PV image to fill with values for debugging.
    full_ipv : npt.NDArray[float], optional
        IPV image to fill with values for debugging.
    full_nbr_label : npt.NDArray[_IntType], optional
        NBR image to fill with values for debugging.
    full_seg_mean : npt.NDArray[float], optional
        Mean pv_guide-values for current segmentation label-image to fill with values
        for debugging.
    full_nbr_mean : npt.NDArray[float], optional
        Mean pv_guide-values for nbr label-image to fill with values for debugging.
    eps : float, default=1e-6
        Epsilon for considering a voxel being in the neighborhood.
    legacy_freesurfer : bool, default=False
        Whether to use the legacy freesurfer mri_segstats formula or the corrected
        formula.

    Returns
    -------
    dict[int, float]
        Dictionary of per-label PV-corrected volume of affected voxels in the patch.

    """

    # Variable conventions:
    # pat_* : *, but sliced to the patch, i.e. a 3D/4D array
    # pat1d_* : like pat_*, but only those voxels, that are part of the border and
    #           flattened

    log_eps = -int(np.log10(eps))

    slicer_patch = tuple(slicer_patch)
    slicer_small_patch, slicer_small_to_patch = pad_slicer(slicer_patch,
                                                           (FILTER_SIZES[0] - 1) // 2,
                                                           seg.shape)
    slicer_large_patch, slicer_large_to_patch = pad_slicer(slicer_patch,
                                                           (FILTER_SIZES[1] - 1) // 2,
                                                           seg.shape)
    slicer_large_to_small = tuple(
        slice(l2p.start - s2p.start,
              None if l2p.stop == s2p.stop else l2p.stop - s2p.stop)
        for s2p, l2p in zip(slicer_small_to_patch, slicer_large_to_patch))
    patch_in_gc = tuple(
        slice(p.start - gc.start,
              p.stop - gc.start)
        for p, gc in zip(slicer_patch, global_crop))

    label_lookup = np.unique(seg[slicer_small_patch])
    maxlabels = label_lookup[-1] + 1
    if maxlabels > 100_000:
        raise RuntimeError("Maximum number of labels above 100000!")
    # create a view for the current patch border
    pat_border = border[slicer_patch]
    pat_is_border, pat_is_nbr, pat_label_counts, pat_label_sums = patch_neighbors(
        label_lookup,
        pv_guide,
        seg,
        pat_border,
        borders,
        slicer_large_patch,
        patch_in_gc,
        slicer_large_to_small,
        slicer_small_to_patch,
        slicer_large_to_patch,
        eps=eps,
        legacy_freesurfer=legacy_freesurfer,
    )

    # both counts and sums are "normalized" by the local neighborhood size (15**3)
    label_lookup_fwd = np.zeros((maxlabels,), dtype="int")
    label_lookup_fwd[label_lookup] = np.arange(label_lookup.shape[0])

    # shrink 3d patch to 1d list of border voxels
    pat1d_pv = pv_guide[slicer_patch][pat_border]
    pat1d_seg = seg[slicer_patch][pat_border]
    pat1d_label_counts = pat_label_counts[:, pat_border]
    pat1d_robust_lblcnt = np.maximum(pat1d_label_counts, eps * 3e-4)
    # both sums and counts are normalized by neighborhood-size**3, both are float
    pat1d_label_means = pat_label_sums[:, pat_border] / pat1d_robust_lblcnt
    pat1d_label_means = pat1d_label_means.round(log_eps + 4)

    # get the mean label intensity of the "local label"
    pat1d_seg_reindexed = np.expand_dims(label_lookup_fwd[pat1d_seg], 0)
    _mean_label = np.take_along_axis(pat1d_label_means, pat1d_seg_reindexed, axis=0)
    mean_label = _mean_label[0]
    # get the index of the "alternative label"
    pat1d_is_this_6border = pat_is_border[:, pat_border]
    # calculate which classes to consider:
    pat1d_mean_intensity_higher = pat1d_label_means > np.expand_dims(pat1d_pv, 0)
    pat1d_mean_intensity_lower = np.expand_dims(mean_label > pat1d_pv, 0)
    pat1d_mean_different = np.expand_dims(np.abs(mean_label - pat1d_pv) > eps, 0)
    pat1d_is_valid = np.all(
        [
            # 1. considered (mean of) alternative label must be on the other side of pv
            # as the (mean of) the segmentation label of the current voxel
            np.logical_xor(pat1d_mean_intensity_higher, pat1d_mean_intensity_lower),
            # 2. considered (mean of) alternative label must be different to pv of voxel
            pat1d_label_means != np.expand_dims(pat1d_pv, 0),
            # 3. (mean of) segmentation label must be different to pv of voxel
            np.broadcast_to(pat1d_mean_different, pat1d_label_means.shape),
            # 4. label must be a neighbor
            pat_is_nbr[:, pat_border],
            # 3. label must not be the segmentation
            pat1d_seg[np.newaxis] != label_lookup[:, np.newaxis],
        ],
        axis=0,
    )

    pat1d_none_valid = ~pat1d_is_valid.any(axis=0, keepdims=False)
    # select the label, that is valid or not valid but also exists and is not the
    # current label
    pat1d_label_frequency = np.round(pat1d_label_counts * pat1d_is_valid, log_eps)
    pat1d_max_frequency_index = pat1d_label_frequency.argmax(axis=0, keepdims=False)

    pat1d_nbr_label = label_lookup[pat1d_max_frequency_index]  # label with max_counts
    pat1d_nbr_label[pat1d_none_valid] = 0

    # get the mean label intensity of the "alternative label"
    pat1d_label_lookup_nbr = np.expand_dims(label_lookup_fwd[pat1d_nbr_label], 0)
    mean_nbr = np.take_along_axis(pat1d_label_means, pat1d_label_lookup_nbr, axis=0)[0]

    # interpolate between the "local" and "alternative label"
    mean_to_mean_nbr = mean_label - mean_nbr
    delta_gt_eps = np.abs(mean_to_mean_nbr) > eps
    # make sure no division by zero
    pat1d_pv = (pat1d_pv - mean_nbr) / np.where(delta_gt_eps, mean_to_mean_nbr, eps)

    # set pv fraction to 1 if division by zero
    pat1d_pv[~delta_gt_eps] = 1.0
    # set pv fraction to 1 for voxels that have no valid nbr
    pat1d_pv[pat1d_none_valid] = 1.0
    pat1d_pv[pat1d_pv > 1.0] = 1.0
    pat1d_pv[pat1d_pv < 0.0] = 0.0

    pat1d_inv_pv = 1.0 - pat1d_pv

    if legacy_freesurfer:
        # re-create the "supposed" freesurfer inconsistency that does not count vertex
        # neighbors, if the voxel label is not of question
        mask_by_6border = np.take_along_axis(
            pat1d_is_this_6border, pat1d_label_lookup_nbr, axis=0
        )
        pat1d_inv_pv = pat1d_inv_pv * mask_by_6border[0]

    if full_pv is not None:
        full_pv[slicer_patch][pat_border] = pat1d_pv
    if full_nbr_label is not None:
        full_nbr_label[slicer_patch][pat_border] = pat1d_nbr_label
    if full_ipv is not None:
        full_ipv[slicer_patch][pat_border] = pat1d_inv_pv
    if full_nbr_mean is not None:
        full_nbr_mean[slicer_patch][pat_border] = mean_nbr
    if full_seg_mean is not None:
        full_seg_mean[slicer_patch][pat_border] = mean_label

    def _vox_calc_pv(lab: _IntType) -> float:
        """
        Compute the PV of voxels labels lab and voxels not labeled lab, but chosen as
        mixing label.
        """
        pv_sum = pat1d_pv.sum(where=pat1d_seg == lab).item()
        inv_pv_sum = pat1d_inv_pv.sum(where=pat1d_nbr_label == lab).item()
        return pv_sum + inv_pv_sum

    return {lab: _vox_calc_pv(lab) for lab in label_lookup}


def patch_neighbors(
    labels: Sequence[_IntType],
    pv_guide: npt.NDArray,
    seg: npt.NDArray[_IntType],
    border_patch: npt.NDArray[bool],
    borders: dict[_IntType, npt.NDArray[bool]],
    slicer_large_patch: SlicingTuple,
    slicer_patch: SlicingTuple,
    slicer_large_to_small: SlicingTuple,
    slicer_small_to_patch: SlicingTuple,
    slicer_large_to_patch: SlicingTuple,
    eps: float = 1e-6,
    legacy_freesurfer: bool = False,
) -> tuple[
    "npt.NDArray[bool]",
    "npt.NDArray[bool]",
    "npt.NDArray[float]",
    "npt.NDArray[float]",
]:
    """
    Calculate the neighbor statistics of labels for a specific patch. The patch is
    defined by patch_padded_large, patch_in_gc, patch_shrink6,

    Parameters
    ----------
    labels : Sequence[int]
        A sequence of all labels that we want to compute the PV for.
    pv_guide : numpy.ndarray
        The (full) image with intensities to guide the PV calculation.
    seg : numpy.typing.NDArray[numpy.integer]
        The segmentation (full image) defining the labels.
    border_patch : npt.NDArray[bool]
        Binary mask for the current patch, True, where a voxel is considered to be a
        border voxel.
    borders : dict[_IntType, npt.NDArray[bool]]
        Dictionary containing the borders for each label.
    slicer_large_patch : SlicingTuple
        Slicing tuple to obtain a patch of shape like the patch but padded to the large
        filter size.
    slicer_patch : SlicingTuple
        Tuple of slice-objects to extract the patch from the full image.
    slicer_large_to_small : SlicingTuple
        Tuple of slice-objects to extract the small patch (patch plus small filter
        window) from the large patch (patch plus large filter window).
    slicer_small_to_patch : SlicingTuple
        Tuple of slice-objects to extract the patch from the patch padded by the small
        filter size.
    slicer_large_to_patch : SlicingTuple
        Tuple of slice-objects to extract the patch from the patch padded by the large
        filter size.
    eps : float, default=1e-6
        Epsilon for considering a voxel being in the neighborhood.
    legacy_freesurfer : bool, default=False
        Whether to use the legacy freesurfer mri_segstats formula or the corrected
        formula.

    Returns
    -------
    pat_is_border : npt.NDArray[bool]
        Array indicating whether each label is on the patch border.
    pat_is_nbr : npt.NDArray[bool]
        Array indicating whether each label is a neighbor in the patch.
    pat_label_count : npt.NDArray[float]
        Array containing label counts in the patch (divided by the neighborhood size).
    pat_label_sums : npt.NDArray[float]
        Array containing the sum of normalized values for each label in the patch.
    """
    shape_of_patch = (len(labels),) + border_patch.shape

    pat_label_counts, pat_label_sums = np.zeros((2,) + shape_of_patch, dtype=float)
    pat_is_nbr, pat_is_border = np.zeros((2,) + shape_of_patch, dtype=bool)  # all False
    for i, lab in enumerate(labels):
        # in legacy freesurfer mode, we want to fill the binary labels with True if we
        # are looking at the background
        fillvalue_binary_label = float(legacy_freesurfer and lab == 0)

        same_label_large_patch = cast(npt.NDArray[bool], seg[slicer_large_patch] == lab)
        same_label_small_patch = same_label_large_patch[slicer_large_to_small]
        # implicitly also a border detection: is lab a neighbor of the "current voxel"
        # returns 'small patch'-array of float (shape: (patch_size + filter_size)**3)
        # for label 'lab'
        tmp_nbr_label_counts = uniform_filter(
            same_label_small_patch,
            FILTER_SIZES[0],
            fillvalue_binary_label,
        )
        if tmp_nbr_label_counts.sum() > eps:
            # lab is at least once a nbr in the patch (grown by one)
            if lab in borders:
                pat_is_border[i] = borders[lab][slicer_patch]
            else:
                pat7_is_border = seg_borders(
                    same_label_small_patch,
                    label=True,
                    cmp_dtype="int8",
                )
                pat_is_border[i] = pat7_is_border[slicer_small_to_patch].astype(bool)

            pat_is_nbr[i] = tmp_nbr_label_counts[slicer_small_to_patch] > eps
            # as float (*filter_size**3)
            pat_label_counts[i] = uniform_filter(
                same_label_large_patch,
                FILTER_SIZES[1],
                fillvalue_binary_label,
                slicer_patch=slicer_large_to_patch
            )
            pat_large_filter_pv = pv_guide[slicer_large_patch] * same_label_large_patch
            pat_label_sums[i] = uniform_filter(
                pat_large_filter_pv,
                FILTER_SIZES[1],
                fillval=0, slicer_patch=slicer_large_to_patch
            )
        # else: lab is not present in the patch
    return pat_is_border, pat_is_nbr, pat_label_counts, pat_label_sums


# timeit cmd arg:
# python -m timeit <<EOF
# from FastSurferCNN.segstats import main, make_arguments
# args = ['-norm', '$TSUB/mri/norm.mgz', '-i', '$TSUB/mri/wmparc.DKTatlas.mapped.mgz',
#         '-o', '$TSUB/stats/wmparc.DKTatlas.mapped.pyvstats', '--lut',
#         '$FREESURFER_HOME/WMParcStatsLUT.txt'];
# main(make_arguments().parse_args(.split(' ')))"
# EOF
if __name__ == "__main__":
    import sys

    args = make_arguments(helpformatter=True)
    opts = args.parse_args()
    if getattr(opts, "out_dir", None) is None:
        from os import environ as env

        if (sd := env.get("SUBJECTS_DIR")) is not None:
            setattr(opts, "out_dir", sd)
    sys.exit(main(opts))
