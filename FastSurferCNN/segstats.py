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
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import nibabel as nib
import numpy as np
import pandas as pd
from numpy import typing as npt

from FastSurferCNN.utils.arg_types import float_gt_zero_and_le_one as robust_threshold
from FastSurferCNN.utils.arg_types import int_ge_zero as id_type
from FastSurferCNN.utils.arg_types import int_gt_zero as patch_size
from FastSurferCNN.utils.parser_defaults import add_arguments
from FastSurferCNN.utils.threads import get_num_threads

USAGE = ("python seg_stats.py  -norm <input_norm> -i <input_seg> -o <output_seg_stats> "
         "[optional arguments]")
DESCRIPTION = ("Script to calculate partial volumes and other segmentation statistics "
               "of a segmentation file.")

HELPTEXT = """
Dependencies:

    Python 3.8+

    Numpy
    http://www.numpy.org

    Nibabel to read images
    http://nipy.org/nibabel/
    
    Pandas to read/write stats files etc.
    https://pandas.pydata.org/

Original Author: David Kügler
Date: Dec-30-2022
Modified: May-08-2023
"""

_NumberType = TypeVar("_NumberType", bound=Number)
_IntType = TypeVar("_IntType", bound=np.integer)
_DType = TypeVar("_DType", bound=np.dtype)
_ArrayType = TypeVar("_ArrayType", bound=np.ndarray)
SlicingTuple = Tuple[slice, ...]
SlicingSequence = Sequence[slice]
PVStats = Dict[str, Union[int, float]]
VirtualLabel = Dict[int, Sequence[int]]
_GlobalStats = Tuple[int, int, Optional[_NumberType], Optional[_NumberType],
                     Optional[float], Optional[float], float, npt.NDArray[bool]]

FILTER_SIZES = (3, 15)

UNITS = {
    "Volume_mm3": "mm^3",
    "normMean": "MR",
    "normStdDev": "MR",
    "normMin": "MR",
    "normMax": "MR",
    "normRange": "MR",
}
FIELDS = {
    "Index": "Index",
    "SegId": "Segmentation Id",
    "NVoxels": "Number of Voxels",
    "Volume_mm3": "Volume",
    "StructName": "Structure Name",
    "normMean": "Intensity normMean",
    "normStdDev": "Intensity normStdDev",
    "normMin": "Intensity normMin",
    "normMax": "Intensity normMax",
    "normRange": "Intensity normRange",
}
FORMATS = {
    "Index": "d",
    "SegId": "d",
    "NVoxels": "d",
    "Volume_mm3": ".3f",
    "StructName": "s",
    "normMean": ".4f",
    "normStdDev": ".4f",
    "normMin": ".4f",
    "normMax": ".4f",
    "normRange": ".4f",
}


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

    def _fill_text(self, text, width, indent):
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
        texts = text.split(self._linebreak_sub())
        return "\n".join(
            [super(HelpFormatter, self)._fill_text(tex, width, indent) for tex in texts]
        )

    def _split_lines(self, text: str, width: int):
        """
        Split lines in the text based on the linebreak substitution string.

        Parameters
        text : str
            The input text.
        width : int
            The width for splitting lines.

        Returns
        -------
        list[str]
            The list of lines.
        """
        texts = text.split(self._linebreak_sub())
        from itertools import chain

        return list(
            chain.from_iterable(
                super(HelpFormatter, self)._split_lines(tex, width) for tex in texts
            )
        )


def make_arguments(helpformatter: bool = False) -> argparse.ArgumentParser:
    """
    Create and configure the argparse.ArgumentParser.

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
        **kwargs
    )
    parser.add_argument(
        "-norm",
        "--normfile",
        type=str,
        required=True,
        dest="normfile",
        help="Biasfield-corrected image in the same image space as segmentation "
             "(required).",
    )
    parser.add_argument(
        "-i",
        "--segfile",
        type=str,
        dest="segfile",
        required=True,
        help="Segmentation file to read and use for evaluation (required).",
    )
    parser.add_argument(
        "-o",
        "--segstatsfile",
        type=str,
        required=True,
        dest="segstatsfile",
        help="Path to output segstats file.",
    )

    parser.add_argument(
        "--excludeid",
        type=id_type,
        nargs="*",
        default=[0],
        help="List of segmentation ids (integers) to exclude in analysis, "
             "e.g. `--excludeid 0 1 10` (default: 0).",
    )
    parser.add_argument(
        "--ids",
        type=id_type, nargs="*",
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
    advanced = parser.add_argument_group(title="Advanced options")
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
        type=patch_size,
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
    advanced = add_arguments(advanced, ["device", "lut", "sid", "in_dir", "allow_root"])
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
             "with more voxels (on the boundry) and smaller voxel sizes (volume per "
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
        type=str,
        dest="mix_coeff",
        default="",
        help="Save the mixing coefficients (default: off).",
    )
    advanced.add_argument(
        "--alternate_labels",
        type=str,
        dest="nbr",
        default="",
        help="Save the alternate labels (default: off).",
    )
    advanced.add_argument(
        "--alternate_mixing_coeff",
        type=str,
        dest="nbr_mix_coeff",
        default="",
        help="Save the alternate labels' mixing coefficients (default: off).",
    )
    advanced.add_argument(
        "--seg_means",
        type=str,
        dest="seg_means",
        default="",
        help="Save the segmentation labels' means (default: off).",
    )
    advanced.add_argument(
        "--alternate_means",
        type=str,
        dest="nbr_means",
        default="",
        help="Save the alternate labels' means (default: off).",
    )
    advanced.add_argument(
        "--volume_precision",
        type=id_type,
        dest="volume_precision",
        default=None,
        help="Number of digits after dot in summary stats file (default: 3). Note, "
        "--legacy_freesurfer sets this to 1.",
    )
    return parser


def loadfile_full(file: str, name: str) -> tuple[nib.analyze.SpatialImage, np.ndarray]:
    """
    Load full image and data.

    Parameters
    ----------
    file : str
        Filename.
    name : str
        Subject name.

    Returns
    -------
    tuple[nib.analyze.SpatialImage, np.ndarray]
        A tuple containing the loaded image and its corresponding data.
    """
    try:
        img = cast(nib.analyze.SpatialImage, nib.load(file))
        if not isinstance(img, nib.analyze.SpatialImage):
            raise RuntimeError(f"Loading the {name} '{file}' was invalid, no "
                               f"SpatialImage.")
    except (IOError, FileNotFoundError) as e:
        args = e.args[0]
        raise IOError(f"Failed loading the {name} '{file}' with error: {args}") from e
    data = np.asarray(img.dataobj)
    return img, data


def main(args):
    """Main segstats function, handles io, input checking and calls pv_calc etc.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments parsed using argparse.

    Returns
    -------
    int
        Exit code. Returns 0 upon successful execution.
    """
    import os
    import time

    start = time.perf_counter_ns()
    from FastSurferCNN.utils.common import assert_no_root

    getattr(args, "allow_root", False) or assert_no_root()

    if not hasattr(args, "segfile") or not os.path.exists(args.segfile):
        return "No segfile was passed or it does not exist."
    if not hasattr(args, "normfile") or not os.path.exists(args.normfile):
        return "No normfile was passed or it does not exist."
    if not hasattr(args, "segstatsfile"):
        return "No segstats file was passed"

    threads = args.threads
    if threads <= 0:
        threads = get_num_threads()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(threads) as tpe:
        # load these files in different threads to avoid waiting on IO
        # (not parallel due to GIL though)
        seg_future = tpe.submit(loadfile_full, args.segfile, "segfile")
        norm_future = tpe.submit(loadfile_full, args.normfile, "normfile")

        if hasattr(args, "lut") and args.lut is not None:
            try:
                lut = read_classes_from_lut(args.lut)
            except FileNotFoundError as e:
                return (f"Could not find the ColorLUT in {args.lut}, please make sure "
                        f"the --lut argument is valid.")
        else:
            lut = None
        try:
            seg: nib.analyze.SpatialImage
            norm: nib.analyze.SpatialImage
            seg_data: np.ndarray
            norm_data: np.ndarray

            seg, seg_data = seg_future.result()
            norm, norm_data = norm_future.result()

            if seg_data.shape != norm_data.shape or \
                    not np.allclose(seg.affine, norm.affine):
                return ("The shapes or affines of the segmentation and the norm image "
                        "are not similar, both must be the same!")

        except (IOError, RuntimeError, FileNotFoundError) as e:
            return e.args[0]
    explicit_ids = False
    if hasattr(args, "ids") and args.ids is not None and len(args.ids) > 0:
        labels = np.asarray(args.ids)
        explicit_ids = True
    elif lut is not None:
        labels = lut["ID"]  # the column ID contains all ids
    else:
        labels = np.unique(seg_data)

    if (
        hasattr(args, "excludeid")
        and args.excludeid is not None
        and len(args.excludeid) > 0
    ):
        exclude_id = list(args.excludeid)
        if explicit_ids:
            _exclude = list(filter(lambda x: x in exclude_id, labels))
            excluded_expl_ids = np.asarray(_exclude)
            if excluded_expl_ids.size > 0:
                return ("Some IDs explicitly passed via --ids are also in the list of "
                        "ids to exclude (--excludeid).")
        labels = np.asarray(list(filter(lambda x: x not in exclude_id, labels)))
    else:
        exclude_id = []

    kwargs = {
        "vox_vol": np.prod(seg.header.get_zooms()).item(),
        "robust_percentage": getattr(args, "robust", None),
        "threads": threads,
        "legacy_freesurfer": bool(getattr(args, "legacy_freesurfer", False)),
        "patch_size": args.patch_size,
    }

    if getattr(args, "volume_precision", None) is not None:
        FORMATS["Volume_mm3"] = f".{getattr(args, 'volume_precision'):d}f"
    elif kwargs["legacy_freesurfer"]:
        FORMATS["Volume_mm3"] = f".1f"

    if args.merged_labels is not None and len(args.merged_labels) > 0:
        kwargs["merged_labels"] = {lab: vals for lab, *vals in args.merged_labels}

    names = ["nbr", "nbr_means", "seg_means", "mix_coeff", "nbr_mix_coeff"]
    var_names = ["nbr", "nbrmean", "segmean", "pv", "ipv"]
    dtypes = [np.int16] + [np.float32] * 4
    if any(getattr(args, n, "") != "" for n in names):
        table, maps = pv_calc(seg_data, norm_data, labels, return_maps=True, **kwargs)

        for n, v, dtype in zip(names, var_names, dtypes):
            file = getattr(args, n, "")
            if file == "":
                continue
            try:
                print(f"Saving {n} to {file}")
                from FastSurferCNN.data_loader.data_utils import save_image

                _header = seg.header.copy()
                _header.set_data_dtype(dtype)
                save_image(_header, seg.affine, maps[v], file, dtype)
            except Exception:
                import traceback

                traceback.print_exc()

    else:
        table: List[PVStats] = pv_calc(seg_data, norm_data, labels, **kwargs)

    if lut is not None:
        for i in range(len(table)):
            lut_idx = lut["ID"] == table[i]["SegId"]
            if lut_idx.any():
                table[i]["StructName"] = lut[lut_idx]["LabelName"].item()
            elif table[i]["SegId"] in kwargs.get("merged_labels", {}).keys():
                # noinspection PyTypeChecker
                table[i]["StructName"] = "Merged-Label-" + str(table[i]["SegId"])
            else:
                # make the label unknown
                table[i]["StructName"] = "Unknown-Label"
        lut_idx = {i: lut["ID"] == i for i in exclude_id}
        _ids = [(i, lut_idx[i]) for i in exclude_id]

        def get_labelname(lid) -> str:
            return lut[lid]["LabelName"].item()
        exclude = {i: get_labelname(lid) if lid.any() else "" for i, lid in _ids}
    else:
        exclude = {i: "" for i in exclude_id}
    dataframe = pd.DataFrame(table, index=np.arange(len(table)))
    if not bool(getattr(args, "empty", False)):
        dataframe = dataframe[dataframe["NVoxels"] != 0]
    dataframe = dataframe.sort_values("SegId")
    dataframe.index = np.arange(1, len(dataframe) + 1)
    lines = []
    if getattr(args, "in_dir", None):
        lines.append(f'SUBJECTS_DIR {getattr(args, "in_dir")}')
    if getattr(args, "sid", None):
        lines.append(f'subjectname {getattr(args, "sid")}')
    lines.append(
        "compatibility with freesurfer's mri_segstats: "
        + ("legacy" if kwargs["legacy_freesurfer"] else "fixed")
    )

    write_statsfile(
        args.segstatsfile,
        dataframe,
        exclude=exclude,
        vox_vol=kwargs["vox_vol"],
        segfile=args.segfile,
        normfile=args.normfile,
        lut=getattr(args, "lut", None),
        extra_header=lines,
    )
    print(
        f"Partial volume stats for {dataframe.shape[0]} labels written to "
        f"{args.segstatsfile}."
    )
    duration = (time.perf_counter_ns() - start) / 1e9
    print(f"Calculation took {duration:.2f} seconds using up to {threads} threads.")
    return 0


def write_statsfile(
    segstatsfile: str,
    dataframe: pd.DataFrame,
    vox_vol: float,
    exclude: Optional[dict[int, str]] = None,
    segfile: str = None,
    normfile: str = None,
    lut: str = None,
    extra_header: Sequence[str] = (),
):
    """
    Write a segstatsfile very similar and compatible with mri_segstats output.

    Parameters
    ----------
    segstatsfile : str
        Path to the output file.
    dataframe : pd.DataFrame
        Data to write into the file.
    vox_vol : float
        Voxel volume for the header.
    exclude : Optional[Dict[int, str]]
        Dictionary of ids and class names that were excluded from the pv analysis
        (default: None).
    segfile : str
        Path to the segmentation file (default: empty).
    normfile : str
        Path to the bias-field corrected image (default: empty).
    lut : str
        Path to the lookup table to find class names for label ids (default: empty).
    extra_header : Sequence[str]
        Sequence of additional lines to add to the header. The initial # and newline
        characters will be added. Should not include newline characters (expect at the
        end of strings). (default: empty sequence).
    """
    import datetime
    import os
    import sys

    def file_annotation(_fp, name: str, file: Optional[str]) -> None:
        if file is not None:
            _fp.write(f"# {name} {file}\n")
            stat = os.stat(file)
            if stat.st_mtime:
                mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
                _fp.write(f"# {name}Timestamp {mtime:%Y/%m/%d %H:%M:%S}\n")

    os.makedirs(os.path.dirname(segstatsfile), exist_ok=True)
    with open(segstatsfile, "w") as fp:
        fp.write(
            "# Title Segmentation Statistics\n#\n"
            "# generating_program segstats.py\n"
            "# cmdline " + " ".join(sys.argv) + "\n"
        )
        if os.name == "posix":
            fp.write(
                f"# sysname  {os.uname().sysname}\n"
                f"# hostname {os.uname().nodename}\n"
                f"# machine  {os.uname().machine}\n"
            )
        else:
            from socket import gethostname

            fp.write(f"# platform {sys.platform}\n" f"# hostname {gethostname()}\n")
        from getpass import getuser

        try:
            fp.write(f"# user       {getuser()}\n")
        except KeyError:
            fp.write(f"# user       UNKNOWN\n")

        fp.write(f"# anatomy_type volume\n#\n")

        file_annotation(fp, "SegVolFile", segfile)
        file_annotation(fp, "ColorTable", lut)
        file_annotation(fp, "PVVolFile", normfile)
        if exclude is not None and len(exclude) > 0:
            if any(len(e) > 0 for e in exclude.values()):
                excl_names = ', '.join(filter(lambda x: len(x) > 0, exclude.values()))
                fp.write(f"# Excluding {excl_names}\n")
            fp.write("".join([f"# ExcludeSegId {id}\n" for id in exclude.keys()]))
        warn_msg_sent = False
        for i, line in enumerate(extra_header):
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
            fp.write(f"# {line}\n")
        fp.write(f"#\n")
        if lut is not None:
            fp.write("# Only reporting non-empty segmentations\n")
        fp.write(f"# VoxelVolume_mm3 {vox_vol}\n")
        # add the Index column, if it is not in dataframe
        if "Index" not in dataframe.columns:
            index_df = pd.DataFrame.from_dict({"Index": dataframe.index})
            index_df.index = dataframe.index
            dataframe = index_df.join(dataframe)

        for i, col in enumerate(dataframe.columns):
            for name, v in zip(
                    ("ColHeader", "FieldName", "Units    "),
                    (col, FIELDS.get(col, "Unknown Column"), UNITS.get(col, "NA"))):
                fp.write(f"# TableCol {i+1: 2d} {name} {v}\n")
        fp.write(f"# NRows {len(dataframe)}\n"
                 f"# NTableCols {len(dataframe.columns)}\n")
        fp.write("# ColHeaders  " + " ".join(dataframe.columns) + "\n")
        max_index = int(np.ceil(np.log10(np.max(dataframe.index))))

        def fmt_field(code: str, data) -> str:
            is_s, is_f, is_d = code[-1] == "s", code[-1] == "f", code[-1] == "d"
            filler = "<" if is_s else " >"
            if is_s:
                prec = int(data.dropna().map(len).max())
            else:
                prec = int(np.ceil(np.log10(data.max())))
            if is_f:
                prec += int(code[-2]) + 1
            return filler + str(prec) + code

        fmts = map(lambda k: "{:" + fmt_field(FORMATS[k], dataframe[k]) + "}",
                   dataframe.columns)
        fmt = " ".join(fmts) + "\n"
        for index, row in dataframe.iterrows():
            data = [row[k] for k in dataframe.columns]
            fp.write(fmt.format(*data))


# Label mapping functions (to aparc (eval) and to label (train))
def read_classes_from_lut(lut_file: str | Path):
    """
    Modify from datautils to allow support for FreeSurfer-distributed ColorLUTs.

    Read in **FreeSurfer-like** LUT table.

    Parameters
    ----------
    lut_file : Path, str
        Path and name of FreeSurfer-style LUT file with classes of interest.
        Example entry:
        ID LabelName  R   G   B   A
        0   Unknown   0   0   0   0
        1   Left-Cerebral-Exterior 70  130 180 0.

    Returns
    -------
    pd.DataFrame
        DataFrame with ids present, name of ids, color for plotting.
    """
    if Path(lut_file).suffix == ".tsv":
        return pd.read_csv(lut_file, sep="\t")

    # Read in file
    names = {
        "ID": "int",
        "LabelName": "str",
        "Red": "int",
        "Green": "int",
        "Blue": "int",
        "Alpha": "int",
    }
    return pd.read_csv(
        lut_file,
        sep="\\s+",
        index_col=False,
        skip_blank_lines=True,
        comment="#",
        header=None,
        names=names.keys(),
        dtype=names,
    )


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
        The image to compute borders from, typically either a label image or a binary
        mask.
    label: int, bool
        Which classes to consider for border computation (True/False for binary mask).
    out: nt.NDArray[bool], optional
        The array for inplace computation.
    cmp_dtype: npt.DTypeLike, default: int8
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
        Input labeled array or binary image.
    labels : Iterable[np.int], bool
        List of labels for which borders will be computed.
        If labels is True, _array is treated as a binary mask.
    max_label : np.int, optional
        The maximum label ot consider. If None, the maximum label in the array is used.
    six_connected : bool, default=True
        If True, 6-connected borders are computed,
        otherwise 26-connected borders are computed.
    out : _ArrayType, optional
        Output array to store the computed borders (Optional).

    Returns
    -------
    _ArrayType
        A binary image where borders are marked as True.

    Raises
    ------
    ValueError
        if labels does not fit to _array (binary mask and integer and vice-versa)
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
        def indexer(axis: int, is_mid: bool) \
                -> tuple[SlicingTuple, SlicingTuple]:
            full_slice = (slice(1, -1),) if is_mid else (slice(None),)
            more_axes = dim - axis - 1
            return ((full_slice * axis + (slice(0, -1),) + full_slice * more_axes),
                    (full_slice * axis + (slice(1, None),) + full_slice * more_axes))

        # compare the [padded] image/array in all directions, x, y, z...
        # ([0], 0, 2, 2, 2, [0]) ==> (False, True, False, False, True)  for each dim
        # is_mid=True: drops padded values in unaffected axes
        ii = partial(indexer, is_mid=True)
        nbr_same = [cmp(padded[ii(i)[0]], padded[ii(i)[1]]) for i in range(dim)]
        # merge neighbors so each border is 2 thick (left and right of change)
        # (False, True, False, False, True) ==> (True, True, False, True)  for each dim
        # is_mid=False: padded values already dropped
        ii = partial(indexer, is_mid=False)
        nbr_same = [logical_or(_ns[ii(i)[0]], _ns[ii(i)[1]]) for i, _ns in enumerate(nbr_same)]
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


def unsqueeze(matrix, axis: int | Sequence[int] = -1):
    """
    Unsqueeze the matrix.

    Allows insertions of axis into the data/tensor, see numpy.expand_dims. This expands the torch.unsqueeze
    syntax to allow unsqueezing multiple axis at the same time.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to unsqueeze.
    axis : int, Sequence[int]
        Axis for unsqueezing.

    Returns
    -------
    np.ndarray
        The unsqueezed matrix.
    """
    if isinstance(matrix, np.ndarray):
        return np.expand_dims(matrix, axis=axis)


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
        tuple of slice-objects to go from image to padded patch
    SlicingTuple
        tuple of slice-objects to go from padded patch to patch

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
        return uniform_filter(
            _arr,
            size=filter_size,
            mode="constant",
            cval=fillval,
            output=out)[_patch]

    return _uniform_filter(data)


@overload
def pv_calc(
    seg: npt.NDArray[_IntType],
    norm: np.ndarray,
    labels: Sequence[_IntType],
    patch_size: int = 32,
    vox_vol: float = 1.0,
    eps: float = 1e-6,
    robust_percentage: Optional[float] = None,
    merged_labels: Optional[VirtualLabel] = None,
    threads: int = -1,
    return_maps: False = False,
    legacy_freesurfer: bool = False,
) -> list[PVStats]:
    ...


@overload
def pv_calc(
    seg: npt.NDArray[_IntType],
    norm: np.ndarray,
    labels: Sequence[_IntType],
    patch_size: int = 32,
    vox_vol: float = 1.0,
    eps: float = 1e-6,
    robust_percentage: Optional[float] = None,
    merged_labels: Optional[VirtualLabel] = None,
    threads: int = -1,
    return_maps: True = True,
    legacy_freesurfer: bool = False,
) -> tuple[list[PVStats], dict[str, dict[int, np.ndarray]]]:
    ...


def pv_calc(
    seg: npt.NDArray[_IntType],
    norm: np.ndarray,
    labels: Sequence[_IntType],
    patch_size: int = 32,
    vox_vol: float = 1.0,
    eps: float = 1e-6,
    robust_percentage: Optional[float] = None,
    merged_labels: Optional[VirtualLabel] = None,
    threads: int = -1,
    return_maps: bool = False,
    legacy_freesurfer: bool = False,
) -> list[PVStats] | tuple[list[PVStats], dict[str, np.ndarray]]:
    """
    Compute volume effects.

    Parameters
    ----------
    seg : npt.NDArray[_IntType]
        Segmentation array with segmentation labels.
    norm : np.ndarray
        Bias.
    labels : Sequence[_IntType]
        Which labels are of interest.
    patch_size : int
        Size of patches (Default value = 32).
    vox_vol : float
        Volume per voxel (Default value = 1.0).
    eps : float
        Threshold for computation of equality (Default value = 1e-6).
    robust_percentage : Optional[float]
        Fraction for robust calculation of statistics (Default value = None).
    merged_labels : Optional[VirtualLabel]
        Defines labels to compute statistics for that are (Default value = None).
    threads : int
        Number of parallel threads to use in calculation (Default value = -1).
    return_maps : bool
        Returns a dictionary containing the computed maps (Default value = False).
    legacy_freesurfer : bool
        Whether to use a freesurfer legacy compatibility mode to exactly replicate
        freesurfer (Default value = False).

    Returns
    -------
    pv_stats : list[PVStats]
        Table (list of dicts) with keys SegId, NVoxels, Volume_mm3, StructName,
        normMean, normStdDev, normMin, normMax, and normRange
        (Note: StructName is unfilled).
    maps : dict[str,np.ndarray], optional
        Only returned, if return_maps is True:
        a dictionary with the 5 meta-information pv-maps:
        - nbr: An image of alternative labels that were considered to mix with the
          voxel's label.
        - nbrmean: The local mean intensity of the label nbr at the specific voxel.
        - segmean: The local mean intensity of the primary label at the specific voxel.
        - pv: The partial volume of the primary label at the location.
        - ipv: The partial volume of the alternative (nbr) label at the location.

    """
    if not isinstance(seg, np.ndarray) or np.issubdtype(seg.dtype, np.integer):
        raise TypeError("The seg object is not a numpy.ndarray of int type.")
    if not isinstance(norm, np.ndarray) or np.issubdtype(seg.dtype, np.number):
        raise TypeError("The norm object is not a numpy.ndarray of number type.")
    if not isinstance(labels, Sequence) or all(isinstance(lab, int) for lab in labels):
        raise TypeError("The labels list is not a sequence of ints.")

    if seg.shape != norm.shape:
        raise RuntimeError(
            f"The shape of the segmentation and the norm must be identical, but shapes "
            f"are {seg.shape} and {norm.shape}!"
        )

    mins, maxes, voxel_counts, robust_voxel_counts = [{} for _ in range(4)]
    borders, sums, sums_2, volumes = [{} for _ in range(4)]

    if merged_labels is not None:
        all_labels = set(labels)
        all_labels |= reduce(lambda i, j: i | j, map(set, merged_labels.values()))
    else:
        all_labels = labels

    # initialize global_crop with the full image
    global_crop: SlicingTuple = tuple(slice(0, _shape) for _shape in seg.shape)
    # ignore all regions of the image that are background only
    if 0 not in all_labels:
        # crop global_crop to the data (plus one extra voxel)
        not_background = cast(seg != 0, npt.NDArray[bool])
        any_in_global, global_crop = crop_patch_to_mask(not_background,
                                                        sub_patch=global_crop)
        # grow global_crop by one, so all border voxels are included
        global_crop = pad_slicer(global_crop, 1, seg.shape)[0]
        if not any_in_global:
            raise RuntimeError("Segmentation map only consists of background")

    global_stats_filled = partial(
        global_stats,
        norm=norm[global_crop],
        seg=seg[global_crop],
        robust_percentage=robust_percentage,
    )
    if threads < 0:
        threads = get_num_threads()
    elif threads == 0:
        raise ValueError("Zero is not a valid number of threads.")
    map_kwargs = {"chunksize": np.ceil(len(labels) / threads)}

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(threads) as pool:
        global_stats_future = pool.map(global_stats_filled, all_labels, **map_kwargs)

        if return_maps:
            full_nbr_label = np.zeros(seg.shape, dtype=seg.dtype)
            full_nbr_mean = np.zeros(norm.shape, dtype=float)
            full_seg_mean = np.zeros(norm.shape, dtype=float)
            full_pv = np.ones(norm.shape, dtype=float)
            full_ipv = np.zeros(norm.shape, dtype=float)
        else:
            full_nbr_label, full_seg_mean, full_nbr_mean, full_pv, full_ipv = [None] * 5

        for lab, *data in global_stats_future:
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
        map_kwargs["chunksize"] = int(np.ceil(num_valid_labels / get_num_threads() / 4))
        patch_filter_func = partial(patch_filter, mask=any_border,
                                    global_crop=global_crop, patch_size=patch_size)
        _patches = pool.map(patch_filter_func, product(*patch_iters), **map_kwargs)
        patches = (patch for has_pv_vox, patch in _patches if has_pv_vox)

        patchwise_pv_calc_func = partial(
            pv_calc_patch,
            global_crop=global_crop,
            loc_border=borders,
            border=any_border,
            seg=seg,
            norm=norm,
            full_nbr_label=full_nbr_label,
            full_seg_mean=full_seg_mean,
            full_pv=full_pv,
            full_ipv=full_ipv,
            full_nbr_mean=full_nbr_mean,
            eps=eps,
            legacy_freesurfer=legacy_freesurfer,
        )
        for vols in pool.map(patchwise_pv_calc_func, patches, **map_kwargs):
            for lab in volumes.keys():
                volumes[lab] += vols.get(lab, 0.0) * vox_vol

    robust_vc_it = robust_voxel_counts.items()
    means = {lab: sums.get(lab, 0.) / cnt for lab, cnt in robust_vc_it if cnt > eps}

    def get_std(lab: _IntType, nvox: int) -> float:
        # *std = sqrt((sum * (*mean) - 2 * (*mean) * sum + sum2) / (nvoxels - 1));
        return np.sqrt((sums_2[lab] - means[lab] * sums[lab]) / (nvox - 1))
    stds = {lab: get_std(lab, nvox) for lab, nvox in robust_vc_it if nvox > eps}

    # ColHeaders: Index SegId NVoxels Volume_mm3 StructName normMean normStdDev ...
    # normMin normMax normRange
    table = [
        {
            "SegId": lab,
            "NVoxels": voxel_counts.get(lab, 0),
            "Volume_mm3": volumes.get(lab, 0.0),
            "StructName": "",
            "normMean": means.get(lab, 0.0),
            "normStdDev": stds.get(lab, 0.0),
            "normMin": mins.get(lab, 0.0),
            "normMax": maxes.get(lab, 0.0),
            "normRange": maxes.get(lab, 0.0) - mins.get(lab, 0.0),
        }
        for lab in labels
    ]
    if merged_labels is not None:
        def aggregate(
                source: Dict[int, _NumberType],
                merge_labels: Iterable[int],
                f: Callable[..., np.ndarray] = np.sum,
        ) -> _NumberType:
            _data = [
                source.get(l, 0)
                for l in merge_labels if robust_voxel_counts.get(l, 0) > eps
            ]
            return f(_data).item()

        def aggregate_std(this_sums, merge_labels, nvoxels) -> float:
            _tmp = [
                s * s / robust_voxel_counts.get(l, 0)
                for l, s in this_sums.items() if robust_voxel_counts.get(l, 0) > eps
            ]
            _tmp = (aggregate(sums_2, merge_labels) - np.sum(_tmp))
            return np.sqrt(_tmp / (nvoxels - 1)).item()

        for lab, merge in merged_labels.items():
            if all(robust_voxel_counts.get(l) is None for l in merge):
                logging.getLogger(__name__).warning(
                    f"None of the labels {merge} for merged label {lab} exist in the "
                    f"segmentation.")
                continue

            nvoxels = aggregate(voxel_counts, merge)
            _min = aggregate(mins, merge, np.min)
            _max = aggregate(maxes, merge, np.max)
            _sums = [(l, sums.get(l, 0)) for l in merge]
            _std = aggregate_std(_sums, merge, nvoxels)
            merge_row = {
                "SegId": lab,
                "NVoxels": nvoxels,
                "Volume_mm3": aggregate(volumes, merge),
                "StructName": "",
                "normMean": aggregate(sums, merge) / nvoxels,
                "normStdDev": _std,
                "normMin": _min,
                "normMax": _max,
                "normRange": _max - _min,
            }
            table.append(merge_row)

    if return_maps:
        return table, {
            "nbr": full_nbr_label,
            "segmean": full_seg_mean,
            "nbrmean": full_nbr_mean,
            "pv": full_pv,
            "ipv": full_ipv,
        }
    return table


def global_stats(
    lab: _IntType,
    norm: npt.NDArray[_NumberType],
    seg: npt.NDArray[_IntType],
    out: Optional[npt.NDArray[bool]] = None,
    robust_percentage: Optional[float] = None
) -> tuple[_IntType, _GlobalStats]:
    """
    Compute Label, Number of voxels, 'robust' number of voxels, norm minimum, maximum,
    sum, sum of squares and 6-connected border of label lab (out references the border).

    Parameters
    ----------
    lab : _IntType
        Label to compute statistics for.
    norm : pt.NDArray[_NumberType]
        The intensity image.
    seg : npt.NDArray[_IntType]
        Segmentation image.
    out : npt.NDArray[bool], optional
        Output array to store the computed borders.
    robust_percentage : float, optional
        A robustness percentile to compute the statistics with (default: None/off = 1).

    Returns
    -------
    label : _IntType
        The label the stats belong to (input).
    stats : _GlobalStats
        A tuple of number_of_voxels, number_of_within_robustness_thresholds,
        minimum_intensity, maximum_intensity, sum_of_intensities,
        sum_of_intensity_squares, and border with respect to the label.

    """
    label_mask = cast(npt.NDArray[bool], seg == lab)
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
    volume: float = np.sum(np.logical_and(label_mask, not out)).astype(float).item()
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
        sequence of slice-objects to extract the subregion of mask that is 'True'.
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
    norm: npt.NDArray,
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
    borders : dict[_IntType, npt.NDArray[bool]]
        Dictionary containing the borders for each label.
    seg : numpy.typing.NDArray[numpy.integer]
        The segmentation (full image) defining the labels.
    norm : numpy.typing.NDArray
        The (full) image with intensities to guide the PV calculation.
    border : npt.NDArray[bool]
        Binary mask, True, where a voxel is considered to be a border voxel.
    full_pv : npt.NDArray[float], optional
        [MISSING].
    full_ipv : npt.NDArray[float], optional
        [MISSING].
    full_nbr_label : npt.NDArray[_IntType], optional
        [MISSING].
    full_seg_mean : npt.NDArray[float], optional
        [MISSING].
    full_nbr_mean : npt.NDArray[float], optional
        [MISSING].
    eps : float, default=1e-6
        Epsilon for considering a voxel being in the neighborhood.
    legacy_freesurfer : bool, default=False
        Whether to use the legacy freesurfer mri_segstats formula or the corrected
        formula.

    Returns
    -------
    dict[_IntType, float]
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
        norm,
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
    pat1d_norm = norm[slicer_patch][pat_border]
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
    pat1d_mean_intensity_higher = pat1d_label_means > np.expand_dims(pat1d_norm, 0)
    pat1d_mean_intensity_lower = np.expand_dims(mean_label > pat1d_norm, 0)
    pat1d_mean_different = np.expand_dims(np.abs(mean_label - pat1d_norm) > eps, 0)
    pat1d_is_valid = np.all(
        # 1. considered (mean of) alternative label must be on the other side of norm
        # as the (mean of) the segmentation label of the current voxel
        [np.logical_xor(pat1d_mean_intensity_higher, pat1d_mean_intensity_lower),
         # 2. considered (mean of) alternative label must be different to norm of voxel
         pat1d_label_means != np.expand_dims(pat1d_norm, 0),
         # 3. (mean of) segmentation label must be different to norm of voxel
         np.broadcast_to(pat1d_mean_different, pat1d_label_means.shape),
         # 4. label must be a neighbor
         pat_is_nbr[:, pat_border],
         # 3. label must not be the segmentation
         pat1d_seg[np.newaxis] != label_lookup[:, np.newaxis]], axis=0)

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
    pat1d_pv = (pat1d_norm - mean_nbr) / np.where(delta_gt_eps, mean_to_mean_nbr, eps)

    # set pv fraction to 1 if division by zero
    pat1d_pv[~delta_gt_eps] = 1.0
    # set pv fraction to 1 for voxels that have no valid nbr
    pat1d_pv[pat1d_none_valid] = 1.0
    pat1d_pv[pat1d_pv > 1.0] = 1.0
    pat1d_pv[pat1d_pv < 0.0] = 0.0

    pat1d_inv_pv = 1. - pat1d_pv

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
        labels: Sequence[int],
        norm: npt.NDArray,
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
):
    """
    Calculate the neighbor statistics of labels for a specific patch.

    The patch is defined by slicer_large_patch, slicer_path, slicer_large_to_small,
    slicer_small_to_path, and slicer_large_to_patch.

    Parameters
    ----------
    labels : Sequence[int]
        A sequence of all labels that we want to compute the PV for.
    norm : numpy.typing.NDArray
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
        tuple of slice-objects to extract the small patch (patch plus small filter
        window) from the large patch (patch plus large filter window).
    slicer_small_to_patch : SlicingTuple
        tuple of slice-objects to extract the patch from the patch padded by the small
        filter size.
    slicer_large_to_patch : SlicingTuple
        tuple of slice-objects to extract the patch from the patch padded by the large
        filter size.
    eps : float, default=1e-6
        epsilon for considering a voxel being in the neighborhood.
    legacy_freesurfer : bool, default=False
        Whether to use the legacy freesurfer mri_segstats formula or the corrected
        formula.

    Returns
    -------
    pat_is_border : np.ndarray
        Array indicating whether each label is on the patch border.
    pat_is_nbr : np.ndarray
        Array indicating whether each label is a neighbor in the patch.
    pat_label_counts : np.ndarray
        Array containing label counts in the patch.
    pat_label_sums : np.ndarray
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
                    True,
                    cmp_dtype="int8",
                )
                pat_is_border[i] = pat7_is_border[slicer_small_to_patch].astype(bool)

            pat_is_nbr[i] = tmp_nbr_label_counts[slicer_small_to_patch] > eps
            # as float (*filter_size**3)
            pat_label_counts[i] = uniform_filter(
                same_label_large_patch,
                FILTER_SIZES[1],
                fillvalue_binary_label,
                slicer_patch=slicer_large_to_patch,
            )
            pat_large_filtered_norm = norm[slicer_large_patch] * same_label_large_patch
            pat_label_sums[i] = uniform_filter(
                pat_large_filtered_norm,
                FILTER_SIZES[1],
                0,
                slicer_patch=slicer_large_to_patch,
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
    sys.exit(main(args.parse_args()))
