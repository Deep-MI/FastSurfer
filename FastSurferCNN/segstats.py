# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases(DZNE), Bonn
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
import multiprocessing
from functools import partial, wraps, reduce
from itertools import product, repeat
from numbers import Number
from packaging import version as _version
from typing import Sequence, Tuple, Union, Optional, Dict, overload, cast, TypeVar, List, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from numpy import typing as npt

from FastSurferCNN.utils.parser_defaults import add_arguments
from FastSurferCNN.utils.arg_types import (int_gt_zero as patch_size, int_ge_zero as id_type,
                                           float_gt_zero_and_le_one as robust_threshold)

USAGE = "seg_stats  -norm <input_norm> -i <input_seg> -o <output_seg_stats> [optional arguments]"
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
"""

_NumberType = TypeVar('_NumberType', bound=Number)
_IntType = TypeVar("_IntType", bound=np.integer)
_DType = TypeVar('_DType', bound=np.dtype)
_ArrayType = TypeVar("_ArrayType", bound=np.ndarray)
PVStats = Dict[str, Union[int, float]]

UNITS = {"Volume_mm3": "mm^3", "normMean": "MR", "normStdDev": "MR", "normMin": "MR", "normMax": "MR",
         "normRange": "MR"}
FIELDS = {"Index": "Index", "SegId": "Segmentation Id", "NVoxels": "Number of Voxels", "Volume_mm3": "Volume",
          "StructName": "Structure Name", "normMean": "Intensity normMean", "normStdDev": "Intensity normStdDev",
          "normMin": "Intensity normMin", "normMax": "Intensity normMax", "normRange": "Intensity normRange"}
FORMATS = {"SegId": "d", "NVoxels": "d", "Volume_mm3": ".1f", "StructName": "s", "normMean": ".4f",
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
                        help="Biasfield corrected image in the same image space as segmentation (required).")
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
    parser = add_arguments(parser, ['device', 'lut', 'allow_root'])
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
    # TODO: Testing of allow_root should a shared FastSurfer function.
    if os.getuid() == 0 and hasattr(args, 'allow_root') and getattr(args, 'allow_root') is not True:
        return "Trying to run script as root without passing --allow_root."

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
        except IOError as e:
            return e.args[0]
    if hasattr(args, 'ids') and args.ids is not None and len(args.ids) > 0:
        labels = np.asarray(args.ids)
    elif lut is not None:
        labels = lut['ID']  # the column ID contains all ids
    else:
        labels = np.unique(seg_data)

    if hasattr(args, 'excludeid') and args.excludeid is not None and len(args.excludeid) > 0:
        exclude_id = list(args.excludeid)
        labels = np.asarray(list(filter(lambda x: x not in exclude_id, labels)))
    else:
        exclude_id = []

    kwargs = {
        "vox_vol": np.prod(seg.header.get_zooms()).item(),
        "robust_percentage": args.robust if hasattr(args, 'robust') else None,
        "threads": threads
    }

    table = pv_calc(seg_data, norm_data, labels, patch_size=args.patch_size, **kwargs)

    if lut is not None:
        for i in range(len(table)):
            table[i]["StructName"] = lut[lut["ID"] == table[i]["SegId"]]["LabelName"].item()
    dataframe = pd.DataFrame(table, index=np.arange(len(table)))
    dataframe = dataframe[dataframe["NVoxels"] != 0].sort_values("SegId")
    dataframe.index = np.arange(1, len(dataframe) + 1)
    args.strict = True
    write_statsfile(args.segstatsfile, dataframe, exclude_id, vox_vol=kwargs["vox_vol"], args=args)
    return 0


def write_statsfile(segstatsfile: str, dataframe: pd.DataFrame, exclude_id: Iterable[int], vox_vol: float, args):
    import sys
    import os
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
        fp.write(f"# user       {getuser()}\n"
                 f"# anatomy_type volume\n#\n"
                 f"# SegVolFile {args.segfile}\n")
        if hasattr(args, "lut") and args.lut is not None:
            fp.write(f"# ColorTable {args.lut}\n")
        fp.write(f"# InVolFile {args.normfile}\n"
                 "".join([f"# ExcludeId {id}\n" for id in exclude_id]))
        fp.write("# Only reporting non-empty segmentations\n"
                 f"# VoxelVolume_mm3 {vox_vol}\n")
        for i, col in enumerate(dataframe.columns):
            for v, name in zip((col, FIELDS.get(col, "Unknown Column"), UNITS.get(col, "NA")),
                               ("ColHeader", "FieldName", "Units    ")):
                fp.write(f"# {i: 2d} {name} {v}\n")
        fp.write(f"# NRows {len(dataframe)}\n"
                 f"# NTableCols {len(dataframe.columns)}\n")
        fp.write("# ColHeaders  " + " ".join(dataframe.columns) + "\n")
        max_index = int(np.ceil(np.log10(np.max(dataframe.index))))

        def fmt_field(code: str, data) -> str:
            is_s, is_f, is_d = code[-1] == "s", code[-1] == "f", code[-1] == "d"
            filler = "<" if is_s else " >"
            prec = int(data.map(len).max() if is_s else np.ceil(np.log10(data.max())))
            if is_f:
                prec += int(code[-2]) + 1
            return filler + str(prec) + code

        fmts = ("{:" + fmt_field(FORMATS[k], dataframe[k]) + "}" for k in dataframe.columns)
        fmt = "{:>" + str(max_index) + "d} " + " ".join(fmts) + "\n"
        for index, row in dataframe.iterrows():
            data = [row[k] for k in dataframe.columns]
            fp.write(fmt.format(index, *data))


# Label mapping functions (to aparc (eval) and to label (train))
def read_classes_from_lut(lut_file):
    """This function is modified from datautils to allow support for FreeSurfer-distributed ColorLUTs

    Function to read in **FreeSurfer-like** LUT table
    :param str lut_file: path and name of FreeSurfer-style LUT file with classes of interest
                         Example entry:
                         ID LabelName  R   G   B   A
                         0   Unknown   0   0   0   0
                         1   Left-Cerebral-Exterior 70  130 180 0
    :return pd.Dataframe: DataFrame with ids present, name of ids, color for plotting
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


def uniform_filter(arr: _ArrayType, filter_size: int,
                   patch: Optional[Tuple[slice, ...]] = None, out: Optional[_ArrayType] = None) -> _ArrayType:
    """Apply a uniform filter (with kernel size `filter_size`) to `input`. The uniform filter is normalized
    (weights add to one)."""
    _patch = (slice(None),) if patch is None else patch
    arr = arr.astype(float)
    from scipy.ndimage import uniform_filter

    def _uniform_filter(_arr, out=None):
        return uniform_filter(_arr, size=filter_size, mode='constant', cval=0, output=out)[_patch]

    if out is not None:
        _uniform_filter(arr, out)
        return out
    return _uniform_filter(arr)


@overload
def pv_calc(seg: npt.NDArray[_IntType], norm: np.ndarray, labels: Sequence[_IntType], patch_size: int = 32,
            vox_vol: float = 1.0, eps: float = 1e-6, robust_percentage: Optional[float] = None,
            return_maps: False = False) -> List[PVStats]:
    ...


@overload
def pv_calc(seg: npt.NDArray[_IntType], norm: np.ndarray, labels: Sequence[_IntType],
            patch_size: int = 32, vox_vol: float = 1.0, eps: float = 1e-6, robust_percentage: Optional[float] = None,
            return_maps: True = True) \
        -> Tuple[List[PVStats], Dict[str, Dict[int, np.ndarray]]]:
    ...


def pv_calc(seg: npt.NDArray[_IntType], norm: np.ndarray, labels: Sequence[_IntType],
            patch_size: int = 32, vox_vol: float = 1.0, eps: float = 1e-6, robust_percentage: Optional[float] = None,
            return_maps: bool = False, threads: int = -1) \
        -> Union[List[PVStats], Tuple[List[PVStats], Dict[str, np.ndarray]]]:
    """Function to compute volume effects.

    Args:
        seg: Segmentation array with segmentation labels
        norm: bias-field corrected image
        labels: Which labels are of interest
        patch_size: Size of patches (default: 32)
        vox_vol: volume per voxel (default: 1.0)
        eps: threshold for computation of equality (default: 1e-6)
        robust_percentage: fraction for robust calculation of statistics (e.g. 0.95 drops both the 2.5%
            lowest and highest values per region) (default: None/1. == off)
        threads: Number of parallel threads to use in calculation (default: -1, one per hardware thread; 0 deactivates
            parallelism).
        return_maps: returns a dictionary containing the computed maps.

    Returns:
        Table (list of dicts) with keys SegId, NVoxels, Volume_mm3, StructName, normMean, normStdDev,
            normMin, normMax, and normRange. (Note: StructName is unfilled)
        if return_maps: a dictionary with the 5 meta-information pv-maps:
                nbr: An image of alternative labels that were considered instead of the voxel's label
                nbrmean: The local mean intensity of the label nbr at the specific voxel
                segmean: The local mean intensity of the primary label at the specific voxel
                pv: The partial volume of the primary label at the location
                ipv: The partial volume of the alternative (nbr) label at the location
    """

    mins, maxes, voxel_counts, __voxel_counts, sums, sums_2, volumes = [{} for _ in range(7)]
    loc_border = {}

    # initialize global_crop with the full image
    global_crop: Tuple[slice, ...] = tuple(slice(0, _shape) for _shape in seg.shape)
    # ignore all regions of the image that are background only
    if 0 not in labels:
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

        global_stats_future = pool.map(global_stats_filled, labels, **map_kwargs)

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

        map_kwargs["chunksize"] = np.ceil(len(voxel_counts) / cpu_count() / 4)  # 4 chunks per core
        _patches = pool.map(partial(patch_filter, mask=border, global_crop=global_crop, patch_size=patch_size),
                            product(*patch_iters), **map_kwargs)
        patches = (patch for has_pv_vox, patch in _patches if has_pv_vox)

        for vols in pool.map(partial(pv_calc_patch, global_crop=global_crop, loc_border=loc_border, border=border,
                                     seg=seg, norm=norm, full_nbr_label=full_nbr_label, full_seg_mean=full_seg_mean,
                                     full_pv=full_pv, full_ipv=full_ipv, full_nbr_mean=full_nbr_mean, eps=eps),
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
    if return_maps:
        return table, {"nbr": full_nbr_label, "segmean": full_seg_mean, "nbrmean": full_nbr_mean, "pv": full_pv,
                       "ipv": full_ipv}
    return table


def global_stats(lab: _IntType, norm: npt.NDArray[_NumberType], seg: npt.NDArray[_IntType],
                 out: Optional[npt.NDArray[bool]] = None, robust_percentage: Optional[float] = None) \
        -> Union[Tuple[_IntType, int],
                 Tuple[_IntType, int, int, _NumberType, _NumberType, float, float, float, npt.NDArray[bool]]]:
    """Computes Label, Number of voxels, 'robust' number of voxels, norm minimum, maximum, sum, sum of squares and
    6-connected border of label lab."""
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

    Args:
        mask: to crop to
        sub_patch: subregion of mask to only consider (default: full mask)

    Note:
        This function requires device synchronization."""

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
                  full_nbr_mean: Optional[npt.NDArray[float]] = None, eps: float = 1e-6) \
        -> Dict[_IntType, float]:
    """Calculates PV for patch. If full* keyword arguments are passed, also fills, per voxel results for the respective
    voxels in the patch."""

    log_eps = -int(np.log10(eps))

    patch = tuple(patch)
    patch_grow1, ungrow1_patch = grow_patch(patch, 1, seg.shape)
    patch_grow7, ungrow7_patch = grow_patch(patch, 7, seg.shape)
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
                          ndarray_alloc=np.full, eps=eps)

    # both counts and sums are "normalized" by the local neighborhood size (15**3)
    label_lookup_fwd = np.zeros((maxlabels,), dtype="int")
    label_lookup_fwd[label_lookup] = np.arange(label_lookup.shape[0])

    # shrink 3d patch to 1d list of border voxels
    pat1d_norm, pat1d_seg = norm[patch][pat_border], seg[patch][pat_border]
    pat1d_label_counts = pat_label_counts[:, pat_border]
    # both sums and counts are normalized by n-hood-size**3, so the output is not anymore
    pat1d_label_means = (pat_label_sums[:, pat_border] / np.maximum(pat1d_label_counts, eps)).round(log_eps)  # float

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
         np.broadcast_to(unsqueeze(mean_label != pat1d_norm, 0), pat1d_label_means.shape),
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

    # re-create the "supposed" freesurfer inconsistency that does not count vertex neighbors, if the voxel label
    # is not of question
    mask_by_6border = np.take_along_axis(pat1d_is_this_6border, unsqueeze(label_lookup_fwd[nbr_label], 0), axis=0)[0]
    pat1d_inv_pv = (1. - pat1d_pv) * mask_by_6border

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
                        ungrow1_patch, ungrow7_patch, ndarray_alloc, eps):
    """Helper function to calculate the neighbor statistics of labels, etc."""
    loc_shape = (len(labels),) + pat_border.shape

    pat_label_counts, pat_label_sums = ndarray_alloc((2,) + loc_shape, fill_value=0., dtype=float)
    pat_is_nbr, pat_is_border = ndarray_alloc((2,) + loc_shape, fill_value=False, dtype=bool)
    for i, lab in enumerate(labels):
        pat7_bin_array = cast(npt.NDArray[bool], seg[patch_grow7] == lab)
        # implicitly also a border detection: is lab a neighbor of the "current voxel"
        tmp_nbr_label_counts = uniform_filter(pat7_bin_array[patch_shrink6], 3)  # as float (*filter_size**3)
        if tmp_nbr_label_counts.sum() > eps:
            # lab is at least once a nbr in the patch (grown by one)
            if lab in loc_border:
                pat_is_border[i] = loc_border[lab][patch_in_gc]
            else:
                pat7_is_border = seg_borders(pat7_bin_array[patch_shrink6], True, cmp_dtype="int8")
                pat_is_border[i] = pat7_is_border[ungrow1_patch].astype(bool)

            pat_is_nbr[i] = tmp_nbr_label_counts[ungrow1_patch] > eps
            pat_label_counts[i] = uniform_filter(pat7_bin_array, 15)[ungrow7_patch]  # as float (*filter_size**3)
            pat7_filtered_norm = norm[patch_grow7] * pat7_bin_array
            pat_label_sums[i] = uniform_filter(pat7_filtered_norm, 15)[ungrow7_patch]
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
