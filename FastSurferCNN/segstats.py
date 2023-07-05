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
import multiprocessing
from functools import partial, reduce
from itertools import product
from numbers import Number
from typing import Sequence, Tuple, Union, Optional, Dict, overload, cast, TypeVar, List, Iterable, Callable

import nibabel as nib
import numpy as np
import pandas as pd
from numpy import typing as npt

from FastSurferCNN.utils.parser_defaults import add_arguments
from FastSurferCNN.utils.arg_types import (int_gt_zero as patch_size, int_ge_zero as id_type,
                                           float_gt_zero_and_le_one as robust_threshold)

USAGE = "python seg_stats.py  -norm <input_norm> -i <input_seg> -o <output_seg_stats> [optional arguments]"
DESCRIPTION = "Script to calculate partial volumes and other segmentation statistics of a segmentation file."

HELPTEXT = """/keeplinebreaks/
Dependencies:

    Python 3.8

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

_NumberType = TypeVar('_NumberType', bound=Number)
_IntType = TypeVar("_IntType", bound=np.integer)
_DType = TypeVar('_DType', bound=np.dtype)
_ArrayType = TypeVar("_ArrayType", bound=np.ndarray)
PVStats = Dict[str, Union[int, float]]
VirtualLabel = Dict[int, Sequence[int]]

FILTER_SIZES = (3, 15)

UNITS = {"Volume_mm3": "mm^3", "normMean": "MR", "normStdDev": "MR", "normMin": "MR", "normMax": "MR",
         "normRange": "MR"}
FIELDS = {"Index": "Index", "SegId": "Segmentation Id", "NVoxels": "Number of Voxels", "Volume_mm3": "Volume",
          "StructName": "Structure Name", "normMean": "Intensity normMean", "normStdDev": "Intensity normStdDev",
          "normMin": "Intensity normMin", "normMax": "Intensity normMax", "normRange": "Intensity normRange"}
FORMATS = {"Index": "d", "SegId": "d", "NVoxels": "d", "Volume_mm3": ".3f", "StructName": "s", "normMean": ".4f",
           "normStdDev": ".4f", "normMin": ".4f", "normMax": ".4f", "normRange": ".4f"}


class HelpFormatter(argparse.HelpFormatter):
    """Help formatter that keeps line breaks for texts that start with '/keeplinebreaks/'."""

    def _fill_text(self, text, width, indent):
        klb_str = '/keeplinebreaks/'
        if text.startswith(klb_str):
            # the following line is from argparse.RawDescriptionHelpFormatter
            return ''.join(indent + line for line in text[len(klb_str):].splitlines(keepends=True))
        else:
            return super()._fill_text(text, width, indent)


def make_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(usage=USAGE, epilog=HELPTEXT, description=DESCRIPTION,
                                     formatter_class=HelpFormatter)
    parser.add_argument('-norm', '--normfile', type=str, required=True, dest='normfile',
                        help="Biasfield-corrected image in the same image space as segmentation (required).")
    parser.add_argument('-i', '--segfile', type=str, dest='segfile', required=True,
                        help="Segmentation file to read and use for evaluation (required).")
    parser.add_argument('-o', '--segstatsfile', type=str, required=True, dest='segstatsfile',
                        help="Path to output segstats file.")

    parser.add_argument('--excludeid', type=id_type, nargs="*", default=[0],
                        help="List of segmentation ids (integers) to exclude in analysis, "
                             "e.g. `--excludeid 0 1 10` (default: 0).")
    parser.add_argument('--ids', type=id_type, nargs="*",
                        help="List of exclusive segmentation ids (integers) to use "
                             "(default: all ids in --lut or all ids in image).")
    parser.add_argument('--merged_label', type=id_type, nargs="+", dest='merged_labels', default=[], action='append',
                        help="Add a 'virtual' label (first value) that is the combination of all following values, "
                             "e.g. `--merged_label 100 3 4 8` will compute the statistics for label 100 by aggregating "
                             "labels 3, 4 and 8.")
    parser.add_argument('--robust', type=robust_threshold, dest='robust', default=None,
                        help="Whether to calculate robust segmentation metrics. This parameter "
                             "expects the fraction of values to keep, e.g. `--robust 0.95` will "
                             "ignore the 2.5%% smallest and the 2.5%% largest values in the "
                             "segmentation when calculating the statistics (default: no robust "
                             "statistics == `--robust 1.0`).")
    advanced = parser.add_argument_group(title="Advanced options")
    advanced.add_argument('--threads', dest='threads', default=multiprocessing.cpu_count(), type=int,
                          help=f"Number of threads to use (defaults to number of hardware threads: "
                               f"{multiprocessing.cpu_count()})")
    advanced.add_argument('--patch_size', type=patch_size, dest='patch_size', default=32,
                          help="Patch size to use in calculating the partial volumes (default: 32).")
    advanced.add_argument('--empty', action='store_true', dest='empty',
                          help="Keep ids for the table that do not exist in the segmentation (default: drop).")
    advanced = add_arguments(advanced, ['device', 'lut', 'sid', 'in_dir', 'allow_root'])
    advanced.add_argument('--legacy_freesurfer', action='store_true', dest='legacy_freesurfer',
                          help="Reproduce FreeSurfer mri_segstats numbers (default: off). \n"
                               "Please note, that exact agreement of numbers cannot be guaranteed, because the "
                               "condition number of FreeSurfers algorithm (mri_segstats) combined with the fact that "
                               "mri_segstats uses 'float' to measure the partial volume corrected volume. This yields "
                               "differences of more than 60mm3 or 0.1%% in large structures. This uniquely impacts "
                               "highres images with more voxels (on the boundry) and smaller voxel sizes (volume per "
                               "voxel).")
    # Additional info:
    # Changing the data type in mri_segstats to double can reduce this difference to nearly zero.
    # mri_segstats has two operations affecting a bad condition number:
    # 1. pv = (val - mean_nbr) / (mean_label - mean_nbr)
    # 2. volume += vox_vol * pv
    #    This is further affected by the small vox_vol (volume per voxel) of highres images (0.7iso -> 0.343)
    # Their effects stack and can result in differences of more than 60mm3 or 0.1% in a comparison between double and
    # single-precision evaluations.
    advanced.add_argument('--mixing_coeff', type=str, dest='mix_coeff', default='',
                          help="Save the mixing coefficients (default: off).")
    advanced.add_argument('--alternate_labels', type=str, dest='nbr', default='',
                          help="Save the alternate labels (default: off).")
    advanced.add_argument('--alternate_mixing_coeff', type=str, dest='nbr_mix_coeff', default='',
                          help="Save the alternate labels' mixing coefficients (default: off).")
    advanced.add_argument('--seg_means', type=str, dest='seg_means', default='',
                          help="Save the segmentation labels' means (default: off).")
    advanced.add_argument('--alternate_means', type=str, dest='nbr_means', default='',
                          help="Save the alternate labels' means (default: off).")
    advanced.add_argument('--volume_precision', type=id_type, dest='volume_precision', default=None,
                          help="Number of digits after dot in summary stats file (default: 3). Note, "
                               "--legacy_freesurfer sets this to 1.")
    return parser


def loadfile_full(file: str, name: str) \
        -> Tuple[nib.analyze.SpatialImage, np.ndarray]:
    try:
        img = nib.load(file)
    except (IOError, FileNotFoundError) as e:
        raise IOError(f"Failed loading the {name} '{file}' with error: {e.args[0]}") from e
    data = np.asarray(img.dataobj)
    return img, data


def main(args):
    import os
    import time
    start = time.perf_counter_ns()
    from FastSurferCNN.utils.common import assert_no_root
    getattr(args, "allow_root", False) or assert_no_root()

    if not hasattr(args, 'segfile') or not os.path.exists(args.segfile):
        return "No segfile was passed or it does not exist."
    if not hasattr(args, 'normfile') or not os.path.exists(args.normfile):
        return "No normfile was passed or it does not exist."
    if not hasattr(args, 'segstatsfile'):
        return "No segstats file was passed"

    threads = args.threads
    if threads <= 0:
        threads = multiprocessing.cpu_count()

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(threads) as tpe:
        # load these files in different threads to avoid waiting on IO (not parallel due to GIL though)
        seg_future = tpe.submit(loadfile_full, args.segfile, 'segfile')
        norm_future = tpe.submit(loadfile_full, args.normfile, 'normfile')

        if hasattr(args, 'lut') and args.lut is not None:
            try:
                lut = read_classes_from_lut(args.lut)
            except FileNotFoundError as e:
                return f"Could not find the ColorLUT in {args.lut}, please make sure the --lut argument is valid."
        else:
            lut = None
        try:
            seg, seg_data = seg_future.result()  # type: nib.analyze.SpatialImage, Union[np.ndarray, torch.IntTensor]
            norm, norm_data = norm_future.result()  # type: nib.analyze.SpatialImage, Union[np.ndarray, torch.Tensor]

            if seg_data.shape != norm_data.shape or not np.allclose(seg.affine, norm.affine):
                return "The shapes or affines of the segmentation and the norm image are not similar, both must be " \
                       "the same!"

        except IOError as e:
            return e.args[0]
    explicit_ids = False
    if hasattr(args, 'ids') and args.ids is not None and len(args.ids) > 0:
        labels = np.asarray(args.ids)
        explicit_ids = True
    elif lut is not None:
        labels = lut['ID']  # the column ID contains all ids
    else:
        labels = np.unique(seg_data)

    if hasattr(args, 'excludeid') and args.excludeid is not None and len(args.excludeid) > 0:
        exclude_id = list(args.excludeid)
        if explicit_ids:
            excluded_expl_ids = np.asarray(list(filter(lambda x: x in exclude_id, labels)))
            if excluded_expl_ids.size > 0:
                return "Some IDs explicitly passed via --ids are also in the list of ids to exclude (--excludeid)"
        labels = np.asarray(list(filter(lambda x: x not in exclude_id, labels)))
    else:
        exclude_id = []

    kwargs = {
        "vox_vol": np.prod(seg.header.get_zooms()).item(),
        "robust_percentage": getattr(args, 'robust', None),
        "threads": threads,
        "legacy_freesurfer": bool(getattr(args, 'legacy_freesurfer', False)),
        "patch_size": args.patch_size
    }

    if getattr(args, 'volume_precision', None) is not None:
        FORMATS['Volume_mm3'] = f'.{getattr(args, "volume_precision"):d}f'
    elif kwargs["legacy_freesurfer"]:
        FORMATS['Volume_mm3'] = f'.1f'

    if args.merged_labels is not None and len(args.merged_labels) > 0:
        kwargs["merged_labels"] = {lab: vals for lab, *vals in args.merged_labels}

    names = ['nbr', 'nbr_means', 'seg_means', 'mix_coeff', 'nbr_mix_coeff']
    var_names = ['nbr', 'nbrmean', 'segmean', 'pv', 'ipv']
    dtypes = [np.int16] + [np.float32] * 4
    if any(getattr(args, n, '') != '' for n in names):
        table, maps = pv_calc(seg_data, norm_data, labels, return_maps=True, **kwargs)

        for n, v, dtype in zip(names, var_names, dtypes):
            file = getattr(args, n, '')
            if file == '':
                continue
            try:
                print(f'Saving {n} to {file}')
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
            elif "merged_labels" in kwargs and table[i]["SegId"] in kwargs["merged_labels"].keys():
                # noinspection PyTypeChecker
                table[i]["StructName"] = "Merged-Label-" + str(table[i]["SegId"])
            else:
                # make the label unknown
                table[i]["StructName"] = "Unknown-Label"
        lut_idx = {i: lut["ID"] == i for i in exclude_id}
        exclude = {i: lut[lut_idx[i]]["LabelName"].item() if lut_idx[i].any() else "" for i in exclude_id}
    else:
        exclude = {i: "" for i in exclude_id}
    dataframe = pd.DataFrame(table, index=np.arange(len(table)))
    if not bool(getattr(args, "empty", False)):
        dataframe = dataframe[dataframe["NVoxels"] != 0]
    dataframe = dataframe.sort_values("SegId")
    dataframe.index = np.arange(1, len(dataframe) + 1)
    lines = []
    if getattr(args, 'in_dir', None):
        lines.append(f'SUBJECTS_DIR {getattr(args, "in_dir")}')
    if getattr(args, 'sid', None):
        lines.append(f'subjectname {getattr(args, "sid")}')
    lines.append("compatibility with freesurfer's mri_segstats: " +
                 ("legacy" if kwargs["legacy_freesurfer"] else "fixed"))

    write_statsfile(args.segstatsfile, dataframe,
                    exclude=exclude, vox_vol=kwargs["vox_vol"], segfile=args.segfile,
                    normfile=args.normfile, lut=getattr(args, "lut", None), extra_header=lines)
    print(f"Partial volume stats for {dataframe.shape[0]} labels written to {args.segstatsfile}.")
    duration = (time.perf_counter_ns() - start) / 1e9
    print(f"Calculation took {duration:.2f} seconds using up to {threads} threads.")
    return 0


def write_statsfile(segstatsfile: str, dataframe: pd.DataFrame, vox_vol: float, exclude: Optional[Dict[int, str]] = None,
                    segfile: str = None, normfile: str = None, lut: str = None, extra_header: Sequence[str] = ()):
    """Write a segstatsfile very similar and compatible with mri_segstats output.

    Parameters
    ----------
    segstatsfile : str
        path to the output file
    dataframe : pd.DataFrame
        data to write into the file
    vox_vol : float
        voxel volume for the header
    exclude : Optional[Dict[int, str]]
        dictionary of ids and class names that were excluded from the pv analysis (default: None)
    segfile : str
        path to the segmentation file (default: empty)
    normfile : str
        path to the bias-field corrected image (default: empty)
    lut : str
        path to the lookup table to find class names for label ids (default: empty)
    extra_header : Sequence[str]
        sequence of additional lines to add to the header. The initial # and newline characters will be
        added. Should not include newline characters (expect at the end of strings). (default: empty sequence)

    """

    import sys
    import os
    import datetime

    def file_annotation(_fp, name: str, file: Optional[str]) -> None:
        if file is not None:
            _fp.write(f"# {name} {file}\n")
            stat = os.stat(file)
            if stat.st_mtime:
                mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
                _fp.write(f"# {name}Timestamp {mtime:%Y/%m/%d %H:%M:%S}\n")

    os.makedirs(os.path.dirname(segstatsfile), exist_ok=True)
    with open(segstatsfile, "w") as fp:
        fp.write("# Title Segmentation Statistics\n#\n"
                 "# generating_program segstats.py\n"
                 "# cmdline " + " ".join(sys.argv) + "\n")
        if os.name == 'posix':
            fp.write(f"# sysname  {os.uname().sysname}\n"
                     f"# hostname {os.uname().nodename}\n"
                     f"# machine  {os.uname().machine}\n")
        else:
            from socket import gethostname
            fp.write(f"# platform {sys.platform}\n"
                     f"# hostname {gethostname()}\n")
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
                fp.write(f"# Excluding {', '.join(filter(lambda x: len(x) > 0, exclude.values()))}\n")
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
                warn_msg_sent or warn(f"extra_header[{i}] includes embedded newline characters. "
                                      "Replacing all newline characters with <space>.")
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
            for v, name in zip((col, FIELDS.get(col, "Unknown Column"), UNITS.get(col, "NA")),
                               ("ColHeader", "FieldName", "Units    ")):
                fp.write(f"# TableCol {i+1: 2d} {name} {v}\n")
        fp.write(f"# NRows {len(dataframe)}\n"
                 f"# NTableCols {len(dataframe.columns)}\n")
        fp.write("# ColHeaders  " + " ".join(dataframe.columns) + "\n")
        max_index = int(np.ceil(np.log10(np.max(dataframe.index))))

        def fmt_field(code: str, data) -> str:
            is_s, is_f, is_d = code[-1] == "s", code[-1] == "f", code[-1] == "d"
            filler = "<" if is_s else " >"
            prec = int(data.dropna().map(len).max() if is_s else np.ceil(np.log10(data.max())))
            if is_f:
                prec += int(code[-2]) + 1
            return filler + str(prec) + code

        fmts = ("{:" + fmt_field(FORMATS[k], dataframe[k]) + "}" for k in dataframe.columns)
        fmt = " ".join(fmts) + "\n"
        for index, row in dataframe.iterrows():
            data = [row[k] for k in dataframe.columns]
            fp.write(fmt.format(*data))


# Label mapping functions (to aparc (eval) and to label (train))
def read_classes_from_lut(lut_file):
    """This function is modified from datautils to allow support for FreeSurfer-distributed ColorLUTs
    
    Function to read in **FreeSurfer-like** LUT table

    Parameters
    ----------
    lut_file :
        path and name of FreeSurfer-style LUT file with classes of interest
        Example entry:
        ID LabelName  R   G   B   A
        0   Unknown   0   0   0   0
        1   Left-Cerebral-Exterior 70  130 180 0
    Returns
    -------
    DataFrame with ids present, name of ids, color for plotting
    
    """
    if lut_file.endswith(".tsv"):
        return pd.read_csv(lut_file, sep="\t")

    # Read in file
    names = {
        "ID": "int",
        "LabelName": "str",
        "Red": "int",
        "Green": "int",
        "Blue": "int",
        "Alpha": "int"
    }
    return pd.read_csv(lut_file, delim_whitespace=True, index_col=False, skip_blank_lines=True,
                       comment="#", header=None, names=names.keys(), dtype=names)


def seg_borders(_array: _ArrayType, label: Union[np.integer, bool],
                out: Optional[_ArrayType] = None, cmp_dtype: npt.DTypeLike = "int8") -> _ArrayType:
    """Handle to fast 6-connected border computation."""
    # binarize
    bin_array = _array if np.issubdtype(_array.dtype, bool) else _array == label
    # scipy laplace is about 20% faster than skimage laplace on cpu
    from scipy.ndimage import laplace

    def _laplace(data):
        """

        Parameters
        ----------
        data :
            

        Returns
        -------

        
        """
        return laplace(data.astype(cmp_dtype)) != np.asarray(0., dtype=cmp_dtype)
    # laplace
    if out is not None:
        out[:] = _laplace(bin_array)
        return out
    else:
        return _laplace(bin_array)


def borders(_array: _ArrayType, labels: Union[Iterable[np.integer], bool], max_label: Optional[np.integer] = None,
            six_connected: bool = True, out: Optional[_ArrayType] = None) -> _ArrayType:
    """Handle to fast border computation."""

    dim = _array.ndim
    array_alloc = partial(np.full, dtype=_array.dtype)
    _shape_plus2 = [s + 2 for s in _array.shape]

    if labels is True:  # already binarized
        if not np.issubdtype(_array, bool):
            raise ValueError("If labels is True, the array should be boolean.")
        cmp = np.logical_xor
    else:
        if np.issubdtype(_array, bool):
            raise ValueError("If labels is a list/iterable, the array should not be boolean.")

        def cmp(a, b):
            return a == b

        if max_label is None:
            max_label = _array.max().item()
        lookup = array_alloc((max_label + 1,), fill_value=0)
        # filter labels from labels that are bigger than max_label
        labels = list(filter(lambda x: x <= max_label, labels))
        if 0 not in labels:
            labels = [0] + labels
        lookup[labels] = np.arange(len(labels), dtype=lookup.dtype)
        _array = lookup[_array]
    logical_or = np.logical_or
    __array = array_alloc(_shape_plus2, fill_value=0)
    __array[(slice(1, -1),) * dim] = _array

    mid = (slice(1, -1),) * dim
    if six_connected:
        def ii(axis: int, off: int, is_mid: bool) -> Tuple[slice, ...]:
            other_slices = mid[:1] if is_mid else (slice(None),)
            return other_slices * axis + (slice(off, -1 if off == 0 else None),) + other_slices * (
                    dim - axis - 1)

        nbr_same = [cmp(__array[ii(i, 0, True)], __array[ii(i, 1, True)]) for i in range(dim)]
        nbr_same = [logical_or(_ns[ii(i, 0, False)], _ns[ii(i, 1, False)]) for i, _ns in enumerate(nbr_same)]
    else:
        def ii(off: Iterable[int]) -> Tuple[slice, ...]:
            return tuple(slice(o, None if o == 2 else o - 3) for o in off)

        nbr_same = [cmp(__array[mid], __array[ii(i - 1)]) for i in np.ndindex((3,) * dim) if np.all(i != 1)]
    return np.logical_or.reduce(nbr_same, out=out)


def unsqueeze(matrix, axis: Union[int, Sequence[int]] = -1):
    """Allows insertions of axis into the data/tensor, see numpy.expand_dims. This expands the torch.unsqueeze
    syntax to allow unsqueezing multiple axis at the same time. """
    if isinstance(matrix, np.ndarray):
        return np.expand_dims(matrix, axis=axis)


def grow_patch(patch: Sequence[slice], whalf: int, img_size: Union[np.ndarray, Sequence[float]]) -> Tuple[
    Tuple[slice, ...], Tuple[slice, ...]]:
    """Create two slicing tuples for indexing ndarrays/tensors that 'grow' and re-'ungrow' the patch `patch` by `whalf` (also considering the image shape)."""
    # patch start/stop
    _patch = np.asarray([(s.start, s.stop) for s in patch])
    start, stop = _patch.T

    # grown patch start/stop
    _start, _stop = np.maximum(0, start - whalf), np.minimum(stop + whalf, img_size)

    # make grown patch and grown patch to patch
    grown_patch = tuple(slice(s.item(), e.item()) for s, e in zip(_start, _stop))
    ungrow_patch = tuple(
        slice(s.item(), None if e.item() == 0 else e.item()) for s, e in zip(start - _start, stop - _stop))
    return grown_patch, ungrow_patch


def uniform_filter(arr: _ArrayType, filter_size: int, fillval: float,
                   patch: Optional[Tuple[slice, ...]] = None, out: Optional[_ArrayType] = None) -> _ArrayType:
    """Apply a uniform filter (with kernel size `filter_size`) to `input`. The uniform filter is normalized
    (weights add to one)."""
    _patch = (slice(None),) if patch is None else patch
    arr = arr.astype(float)
    from scipy.ndimage import uniform_filter

    def _uniform_filter(_arr, out=None):
        return uniform_filter(_arr, size=filter_size, mode='constant', cval=fillval, output=out)[_patch]

    if out is not None:
        _uniform_filter(arr, out)
        return out
    return _uniform_filter(arr)


@overload
def pv_calc(seg: npt.NDArray[_IntType], norm: np.ndarray, labels: Sequence[_IntType], patch_size: int = 32,
            vox_vol: float = 1.0, eps: float = 1e-6, robust_percentage: Optional[float] = None,
            merged_labels: Optional[VirtualLabel] = None, threads: int = -1, return_maps: False = False,
            legacy_freesurfer: bool = False) -> List[PVStats]:
    ...


@overload
def pv_calc(seg: npt.NDArray[_IntType], norm: np.ndarray, labels: Sequence[_IntType],
            patch_size: int = 32, vox_vol: float = 1.0, eps: float = 1e-6, robust_percentage: Optional[float] = None,
            merged_labels: Optional[VirtualLabel] = None, threads: int = -1, return_maps: True = True,
            legacy_freesurfer: bool = False) \
        -> Tuple[List[PVStats], Dict[str, Dict[int, np.ndarray]]]:
    ...


def pv_calc(seg: npt.NDArray[_IntType], norm: np.ndarray, labels: Sequence[_IntType],
            patch_size: int = 32, vox_vol: float = 1.0, eps: float = 1e-6, robust_percentage: Optional[float] = None,
            merged_labels: Optional[VirtualLabel] = None, threads: int = -1, return_maps: bool = False,
            legacy_freesurfer: bool = False) \
        -> Union[List[PVStats], Tuple[List[PVStats], Dict[str, np.ndarray]]]:
    """Function to compute volume effects.

    Parameters
    ----------
    seg : npt.NDArray[_IntType]
        Segmentation array with segmentation labels
    norm : np.ndarray
        bias
    labels : Sequence[_IntType]
        Which labels are of interest
    patch_size : int
        Size of patches (Default value = 32)
    vox_vol : float
        volume per voxel (Default value = 1.0)
    eps : float
        threshold for computation of equality (Default value = 1e-6)
    robust_percentage : Optional[float]
        fraction for robust calculation of statistics (Default value = None)
    merged_labels : Optional[VirtualLabel]
        defines labels to compute statistics for that are (Default value = None)
    threads : int
        Number of parallel threads to use in calculation (Default value = -1)
    return_maps : bool
        returns a dictionary containing the computed maps (Default value = False)
    legacy_freesurfer : bool
        whether to use a freesurfer legacy compatibility mode to exactly replicate freesurfer (Default value = False)

    Returns
    -------
    Union[List[PVStats],Tuple[List[PVStats],Dict[str,np.ndarray]]]
        Table (list of dicts) with keys SegId, NVoxels, Volume_mm3, StructName, normMean, normStdDev,
        normMin, normMax, and normRange. (Note: StructName is unfilled)
        if return_maps: a dictionary with the 5 meta-information pv-maps:
        nbr: An image of alternative labels that were considered instead of the voxel's label
        nbrmean: The local mean intensity of the label nbr at the specific voxel
        segmean: The local mean intensity of the primary label at the specific voxel
        pv: The partial volume of the primary label at the location
        ipv: The partial volume of the alternative (nbr) label at the location

    
    """

    if not isinstance(seg, np.ndarray) and np.issubdtype(seg.dtype, np.integer):
        raise TypeError("The seg object is not a numpy.ndarray of int type.")
    if not isinstance(norm, np.ndarray) and np.issubdtype(seg.dtype, np.numeric):
        raise TypeError("The norm object is not a numpy.ndarray of numeric type.")
    if not isinstance(labels, Sequence) and all(isinstance(lab, int) for lab in labels):
        raise TypeError("The labels list is not a sequence of ints.")

    if seg.shape != norm.shape:
        raise RuntimeError(f"The shape of the segmentation and the norm must be identical, but shapes are {seg.shape} "
                           f"and {norm.shape}!")

    mins, maxes, voxel_counts, __voxel_counts, sums, sums_2, volumes = [{} for _ in range(7)]
    loc_border = {}

    if merged_labels is not None:
        all_labels = set(labels)
        all_labels = all_labels | reduce(lambda i, j: i | j, (set(s) for s in merged_labels.values()))
    else:
        all_labels = labels

    # initialize global_crop with the full image
    global_crop: Tuple[slice, ...] = tuple(slice(0, _shape) for _shape in seg.shape)
    # ignore all regions of the image that are background only
    if 0 not in all_labels:
        # crop global_crop to the data (plus one extra voxel)
        any_in_global, global_crop = crop_patch_to_mask(seg != 0, sub_patch=global_crop)
        # grow global_crop by one, so all border voxels are included
        global_crop = grow_patch(global_crop, 1, seg.shape)[0]
        if not any_in_global:
            raise RuntimeError("Segmentation map only consists of background")

    global_stats_filled = partial(global_stats,
                                  norm=norm[global_crop], seg=seg[global_crop],
                                  robust_percentage=robust_percentage)
    from multiprocessing import cpu_count
    map_kwargs = {"chunksize": np.ceil(len(labels) / cpu_count())}
    if threads < 0:
        threads = cpu_count()
    elif threads == 0:
        raise ValueError("Zero is not a valid number of threads.")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(threads) as pool:

        global_stats_future = pool.map(global_stats_filled, all_labels, **map_kwargs)

        if return_maps:
            _ndarray_alloc = np.full
            full_nbr_label = _ndarray_alloc(seg.shape, fill_value=0, dtype=seg.dtype)
            full_nbr_mean = _ndarray_alloc(norm.shape, fill_value=0, dtype=np.dtype(float))
            full_seg_mean = _ndarray_alloc(norm.shape, fill_value=0, dtype=np.dtype(float))
            full_pv = _ndarray_alloc(norm.shape, fill_value=1, dtype=np.dtype(float))
            full_ipv = _ndarray_alloc(norm.shape, fill_value=0, dtype=np.dtype(float))
        else:
            full_nbr_label, full_seg_mean, full_nbr_mean, full_pv, full_ipv = [None] * 5

        for lab, *data in global_stats_future:
            if data[0] != 0:
                voxel_counts[lab], __voxel_counts[lab] = data[:2]
                mins[lab], maxes[lab], sums[lab], sums_2[lab] = data[2:-2]
                volumes[lab], loc_border[lab] = data[-2] * vox_vol, data[-1]

        # un_global_crop border here
        _border = np.any(list(loc_border.values()), axis=0)
        border = np.pad(_border, tuple((slc.start, shp - slc.stop) for slc, shp in zip(global_crop, seg.shape)))
        if not np.array_equal(border.shape, seg.shape):
            raise RuntimeError("border and seg_array do not have same shape.")

        # iterate through patches of the image
        patch_iters = [range(slice_.start, slice_.stop, patch_size) for slice_ in global_crop]  # for 3D

        map_kwargs["chunksize"] = int(np.ceil(len(voxel_counts) / cpu_count() / 4))  # 4 chunks per core
        _patches = pool.map(partial(patch_filter, mask=border, global_crop=global_crop, patch_size=patch_size),
                            product(*patch_iters), **map_kwargs)
        patches = (patch for has_pv_vox, patch in _patches if has_pv_vox)

        for vols in pool.map(partial(pv_calc_patch, global_crop=global_crop, loc_border=loc_border, border=border,
                                     seg=seg, norm=norm, full_nbr_label=full_nbr_label, full_seg_mean=full_seg_mean,
                                     full_pv=full_pv, full_ipv=full_ipv, full_nbr_mean=full_nbr_mean, eps=eps,
                                     legacy_freesurfer=legacy_freesurfer),
                             patches, **map_kwargs):
            for lab in volumes.keys():
                volumes[lab] += vols.get(lab, 0.) * vox_vol

    means = {lab: s / __voxel_counts[lab] for lab, s in sums.items() if __voxel_counts.get(lab, 0) > eps}
    # *std = sqrt((sum * (*mean) - 2 * (*mean) * sum + sum2) / (nvoxels - 1));
    stds = {lab: np.sqrt((sums_2[lab] - means[lab] * sums[lab]) / (nvox - 1)) for lab, nvox in
            __voxel_counts.items() if nvox > 1}
    # ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
    table = [{"SegId": lab, "NVoxels": voxel_counts.get(lab, 0), "Volume_mm3": volumes.get(lab, 0.),
              "StructName": "", "normMean": means.get(lab, 0.), "normStdDev": stds.get(lab, 0.),
              "normMin": mins.get(lab, 0.), "normMax": maxes.get(lab, 0.),
              "normRange": maxes.get(lab, 0.) - mins.get(lab, 0.)} for lab in labels]
    if merged_labels is not None:
        def agg(f: Callable[..., np.ndarray], source: Dict[int, _NumberType], merge_labels: Iterable[int]) -> _NumberType:
            return f([source.get(l, 0) for l in merge_labels if __voxel_counts.get(l) is not None]).item()

        for lab, merge in merged_labels.items():
            if all(__voxel_counts.get(l) is None for l in merge):
                logging.getLogger(__name__).warning(f"None of the labels {merge} for merged label {lab} exist in the "
                                                    f"segmentation.")
                continue

            nvoxels, _min, _max = agg(np.sum, voxel_counts, merge), agg(np.min, mins, merge), agg(np.max, maxes, merge)
            _sums = [(l, sums.get(l, 0)) for l in merge]
            _std_tmp = np.sum([s * s / __voxel_counts.get(l, 0) for l, s in _sums if __voxel_counts.get(l, 0) > 0])
            _std = np.sqrt((agg(np.sum, sums_2, merge) - _std_tmp) / (nvoxels - 1)).item()
            merge_row = {"SegId": lab,  "NVoxels": nvoxels, "Volume_mm3": agg(np.sum, volumes, merge),
                         "StructName": "", "normMean": agg(np.sum, sums, merge) / nvoxels, "normStdDev": _std,
                         "normMin": _min, "normMax": _max, "normRange": _max - _min}
            table.append(merge_row)

    if return_maps:
        return table, {"nbr": full_nbr_label, "segmean": full_seg_mean, "nbrmean": full_nbr_mean, "pv": full_pv,
                       "ipv": full_ipv}
    return table


def global_stats(lab: _IntType, norm: npt.NDArray[_NumberType], seg: npt.NDArray[_IntType],
                 out: Optional[npt.NDArray[bool]] = None, robust_percentage: Optional[float] = None) \
        -> Union[Tuple[_IntType, int],
                 Tuple[_IntType, int, int, _NumberType, _NumberType, float, float, float, npt.NDArray[bool]]]:
    """Computes Label, Number of voxels, 'robust' number of voxels, norm minimum, maximum, sum, sum of squares and
    6-connected border of label lab (out references the border)."""
    bin_array = cast(npt.NDArray[bool], seg == lab)
    data = norm[bin_array].astype(int if np.issubdtype(norm.dtype, np.integer) else float)
    nvoxels: int = data.shape[0]
    # if lab is not in the image at all
    if nvoxels == 0:
        return lab, 0
    # compute/update the border
    if out is None:
        out = seg_borders(bin_array, True, cmp_dtype="int8").astype(bool)
    else:
        out[:] = seg_borders(bin_array, True, cmp_dtype="int").astype(bool)

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
    volume: float = np.sum(np.logical_and(bin_array, np.logical_not(out))).astype(float).item()
    return lab, nvoxels, __voxel_count, _min, _max, _sum, sum_2, volume, out


def patch_filter(pos: Tuple[int, int, int], mask: npt.NDArray[bool],
                 global_crop: Tuple[slice, ...], patch_size: int = 32) \
        -> Tuple[bool, Sequence[slice]]:
    """Returns, whether there are mask-True voxels in the patch starting at pos with size patch_size and the resulting
    patch shrunk to mask-True regions."""
    # create slices for current patch context (constrained by the global_crop)
    patch = [slice(p, min(p + patch_size, slice_.stop)) for p, slice_ in zip(pos, global_crop)]
    # crop patch context to the image content
    return crop_patch_to_mask(mask, sub_patch=patch)


def crop_patch_to_mask(mask: npt.NDArray[_NumberType],
                       sub_patch: Optional[Sequence[slice]] = None) \
        -> Tuple[bool, Sequence[slice]]:
    """Crop the patch to regions of the mask that are non-zero. Assumes mask is always positive. Returns whether there
    is any mask>0 in the patch and a patch shrunk to mask>0 regions. The optional subpatch constrains this operation to
    the sub-region defined by a sequence of slicing operations.

    Parameters
    ----------
    mask : npt.NDArray[_NumberType]
        to crop to
    sub_patch : Optional[Sequence[slice]]
        subregion of mask to only consider (default: full mask)

    Note
    ----
    This function requires device synchronization.

    """

    _patch = []
    patch = tuple([slice(0, s) for s in mask.shape] if sub_patch is None else sub_patch)
    patch_in_patch_coords = tuple([slice(0, slice_.stop - slice_.start) for slice_ in patch])
    in_mask = True
    _mask = mask[patch].sum(axis=2)
    for i, pat in enumerate(patch_in_patch_coords):
        p = pat.start
        if in_mask:
            if i == 2:
                _mask = mask[patch][tuple(_patch)].sum(axis=0)
            # can we shrink the patch context in i-th axis?
            pat_has_mask_in_axis = _mask[tuple(_patch[1:] if i != 2 else [])].sum(axis=int(i == 0)) > 0
            # modify both the _patch_size and the coordinate p to shrink the patch
            _pat_mask = np.argwhere(pat_has_mask_in_axis)
            if _pat_mask.shape[0] == 0:
                _patch_size = 0
                in_mask = False
            else:
                offset = _pat_mask[0].item()
                p += offset
                _patch_size = _pat_mask[-1].item() - offset + 1
        else:
            _patch_size = 0
        _patch.append(slice(p, p + _patch_size))

    out_patch = [slice(_p.start + p.start, p.start + _p.stop) for _p, p in zip(_patch, patch)]
    return _patch[0].start != _patch[0].stop, out_patch


def pv_calc_patch(patch: Tuple[slice, ...], global_crop: Tuple[slice, ...],
                  loc_border: Dict[_IntType, npt.NDArray[bool]],
                  seg: npt.NDArray[_IntType], norm: np.ndarray, border: npt.NDArray[bool],
                  full_pv: Optional[npt.NDArray[float]] = None, full_ipv: Optional[npt.NDArray[float]] = None,
                  full_nbr_label: Optional[npt.NDArray[_IntType]] = None,
                  full_seg_mean: Optional[npt.NDArray[float]] = None,
                  full_nbr_mean: Optional[npt.NDArray[float]] = None, eps: float = 1e-6,
                  legacy_freesurfer: bool = False) \
        -> Dict[_IntType, float]:
    """Calculates PV for patch. If full* keyword arguments are passed, also fills, per voxel results for the respective
    voxels in the patch."""

    log_eps = -int(np.log10(eps))

    patch = tuple(patch)
    patch_grow1, ungrow1_patch = grow_patch(patch, (FILTER_SIZES[0]-1)//2, seg.shape)
    patch_grow7, ungrow7_patch = grow_patch(patch, (FILTER_SIZES[1]-1)//2, seg.shape)
    patch_shrink6 = tuple(
        slice(ug7.start - ug1.start, None if ug7.stop == ug1.stop else ug7.stop - ug1.stop) for ug1, ug7 in
        zip(ungrow1_patch, ungrow7_patch))
    patch_in_gc = tuple(slice(p.start - gc.start, p.stop - gc.start) for p, gc in zip(patch, global_crop))

    label_lookup = np.unique(seg[patch_grow1])
    maxlabels = label_lookup[-1] + 1
    if maxlabels > 100_000:
        raise RuntimeError("Maximum number of labels above 100000!")
    # create a view for the current patch border
    pat_border = border[patch]
    pat_is_border, pat_is_nbr, pat_label_counts, pat_label_sums \
        = patch_neighbors(label_lookup, norm, seg, pat_border, loc_border,
                          patch_grow7, patch_in_gc, patch_shrink6, ungrow1_patch, ungrow7_patch,
                          ndarray_alloc=np.full, eps=eps, legacy_freesurfer=legacy_freesurfer)

    # both counts and sums are "normalized" by the local neighborhood size (15**3)
    label_lookup_fwd = np.zeros((maxlabels,), dtype="int")
    label_lookup_fwd[label_lookup] = np.arange(label_lookup.shape[0])

    # shrink 3d patch to 1d list of border voxels
    pat1d_norm, pat1d_seg = norm[patch][pat_border], seg[patch][pat_border]
    pat1d_label_counts = pat_label_counts[:, pat_border]
    # both sums and counts are normalized by n-hood-size**3, so the output is not anymore
    pat1d_label_means = (pat_label_sums[:, pat_border] / np.maximum(pat1d_label_counts, eps * 0.0003)).round(log_eps + 4)  # float

    # get the mean label intensity of the "local label"
    mean_label = np.take_along_axis(pat1d_label_means, unsqueeze(label_lookup_fwd[pat1d_seg], 0), axis=0)[0]
    # get the index of the "alternative label"
    pat1d_is_this_6border = pat_is_border[:, pat_border]
    # calculate which classes to consider:
    is_valid = np.all(
        # 1. considered (mean of) alternative label must be on the other side of norm as the (mean of) the segmentation
        # label of the current voxel
        [np.logical_xor(pat1d_label_means > unsqueeze(pat1d_norm, 0), unsqueeze(mean_label > pat1d_norm, 0)),
         # 2. considered (mean of) alternative label must be different to norm of voxel
         pat1d_label_means != unsqueeze(pat1d_norm, 0),
         # 3. (mean of) segmentation label must be different to norm of voxel
         np.broadcast_to(unsqueeze(np.abs(mean_label - pat1d_norm) > eps, 0), pat1d_label_means.shape),
         # 4. label must be a neighbor
         pat_is_nbr[:, pat_border],
         # 3. label must not be the segmentation
         pat1d_seg[np.newaxis] != label_lookup[:, np.newaxis]], axis=0)

    none_valid = ~is_valid.any(axis=0, keepdims=False)
    # select the label, that is valid or not valid but also exists and is not the current label
    max_counts_index = np.round(pat1d_label_counts * is_valid, log_eps).argmax(axis=0, keepdims=False)

    nbr_label = label_lookup[max_counts_index]  # label with max_counts
    nbr_label[none_valid] = 0

    # get the mean label intensity of the "alternative label"
    mean_nbr = np.take_along_axis(pat1d_label_means, unsqueeze(label_lookup_fwd[nbr_label], 0), axis=0)[0]

    # interpolate between the "local" and "alternative label"
    mean_to_mean_nbr = mean_label - mean_nbr
    delta_gt_eps = np.abs(mean_to_mean_nbr) > eps
    pat1d_pv = (pat1d_norm - mean_nbr) / np.where(delta_gt_eps, mean_to_mean_nbr, eps)  # make sure no division by zero

    pat1d_pv[~delta_gt_eps] = 1.  # set pv fraction to 1 if division by zero
    pat1d_pv[none_valid] = 1.  # set pv fraction to 1 for voxels that have no 'valid' nbr
    pat1d_pv[pat1d_pv > 1.] = 1.
    pat1d_pv[pat1d_pv < 0.] = 0.

    pat1d_inv_pv = 1. - pat1d_pv

    if legacy_freesurfer:
        # re-create the "supposed" freesurfer inconsistency that does not count vertex neighbors, if the voxel label
        # is not of question
        mask_by_6border = np.take_along_axis(pat1d_is_this_6border, unsqueeze(label_lookup_fwd[nbr_label], 0), axis=0)[0]
        pat1d_inv_pv = pat1d_inv_pv * mask_by_6border

    if full_pv is not None:
        full_pv[patch][pat_border] = pat1d_pv
    if full_nbr_label is not None:
        full_nbr_label[patch][pat_border] = nbr_label
    if full_ipv is not None:
        full_ipv[patch][pat_border] = pat1d_inv_pv
    if full_nbr_mean is not None:
        full_nbr_mean[patch][pat_border] = mean_nbr
    if full_seg_mean is not None:
        full_seg_mean[patch][pat_border] = mean_label

    return {lab: (pat1d_pv.sum(where=pat1d_seg == lab) + pat1d_inv_pv.sum(where=nbr_label == lab)).item() for lab in
            label_lookup}


def patch_neighbors(labels, norm, seg, pat_border, loc_border, patch_grow7, patch_in_gc, patch_shrink6,
                        ungrow1_patch, ungrow7_patch, ndarray_alloc, eps, legacy_freesurfer = False):
    """Helper function to calculate the neighbor statistics of labels, etc."""
    loc_shape = (len(labels),) + pat_border.shape

    pat_label_counts, pat_label_sums = ndarray_alloc((2,) + loc_shape, fill_value=0., dtype=float)
    pat_is_nbr, pat_is_border = ndarray_alloc((2,) + loc_shape, fill_value=False, dtype=bool)
    for i, lab in enumerate(labels):
        # in legacy freesurfer mode, we want to fill the binary labels with True if we are looking at the background
        fill_binary_label = float(legacy_freesurfer and lab == 0)

        pat7_bin_array = cast(npt.NDArray[bool], seg[patch_grow7] == lab)
        # implicitly also a border detection: is lab a neighbor of the "current voxel"
        tmp_nbr_label_counts = uniform_filter(pat7_bin_array[patch_shrink6], FILTER_SIZES[0], fill_binary_label)  # as float (*filter_size**3)
        if tmp_nbr_label_counts.sum() > eps:
            # lab is at least once a nbr in the patch (grown by one)
            if lab in loc_border:
                pat_is_border[i] = loc_border[lab][patch_in_gc]
            else:
                pat7_is_border = seg_borders(pat7_bin_array[patch_shrink6], True, cmp_dtype="int8")
                pat_is_border[i] = pat7_is_border[ungrow1_patch].astype(bool)

            pat_is_nbr[i] = tmp_nbr_label_counts[ungrow1_patch] > eps
            pat_label_counts[i] = uniform_filter(pat7_bin_array, FILTER_SIZES[1], fill_binary_label)[ungrow7_patch]  # as float (*filter_size**3)
            pat7_filtered_norm = norm[patch_grow7] * pat7_bin_array
            pat_label_sums[i] = uniform_filter(pat7_filtered_norm, FILTER_SIZES[1], 0)[ungrow7_patch]
        # else: lab is not present in the patch
    return pat_is_border, pat_is_nbr, pat_label_counts, pat_label_sums


if __name__ == "__main__":
    # full timeit cmd arg
    # python -m timeit "from FastSurferCNN.segstats import main, make_arguments; \
    # main(make_arguments().parse_args('-norm $TSUB/mri/norm.mgz -i $TSUB/mri/wmparc.DKTatlas.mapped.mgz \
    # -o $TSUB/stats/wmparc.DKTatlas.mapped.pyvstats --lut $FREESURFER/WMParcStatsLUT.txt'.split(' ')))"
    import sys

    args = make_arguments()
    sys.exit(main(args.parse_args()))
