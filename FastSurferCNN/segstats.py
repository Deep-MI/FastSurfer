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
from functools import partial, wraps, reduce
from itertools import product, repeat
from numbers import Number
from packaging import version as _version
from typing import Sequence, Tuple, Union, Optional, Dict, overload, cast, TypeVar, List, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from numpy import typing as npt

try:
    import torch
    from torch.nn import functional as _F

    _HAS_TORCH = True
except ImportError:
    # ensure Tensor, etc. are defined for typing
    torch = NotImplemented
    _F = NotImplemented
    _HAS_TORCH = False
try:
    import numba
    from numba.np import numpy_support

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    numba = NotImplemented
    numpy_support = NotImplemented

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

_Conv = {1: _F.conv1d, 2: _F.conv2d, 3: _F.conv3d}

_NumberType = TypeVar('_NumberType', bound=Number)
_IntType = TypeVar("_IntType", bound=np.integer)
_DType = TypeVar('_DType', bound=np.dtype)
_ArrayType = TypeVar("_ArrayType", np.ndarray, 'torch.Tensor')
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
    advanced.add_argument('--legacy', dest='legacy', action='store_true',
                          help="Use the lagacy FreeSurfer algorithm.")
    advanced.add_argument('--patch_size', type=patch_size, dest='patch_size', default=32,
                          help="Patch size to use in calculating the partial volumes (default: 32).")
    parser = add_arguments(parser, ['device', 'lut', 'allow_root'])
    return parser


def loadfile_full(file: str, name: str, device: Union[str, 'torch.device'] = 'cpu') \
        -> Tuple[nib.analyze.SpatialImage, Union[np.ndarray, 'torch.Tensor']]:
    try:
        img = nib.load(file)
    except (IOError, FileNotFoundError) as e:
        raise IOError(f"Failed loading the {name} '{file}' with error: {e.args[0]}") from e
    data = np.asarray(img.dataobj)
    if device != 'cpu':
        if data.dtype.byteorder != "=":
            data = data.astype(data.dtype.newbyteorder("native"))
        data = torch.as_tensor(data, device=device)
    return img, data


def main(args):
    import os
    if os.getuid() == 0 and hasattr(args, 'allow_root') and getattr(args, 'allow_root') is not True:
        return "Trying to run script as root without passing --allow_root."

    if not hasattr(args, 'segfile') or not os.path.exists(args.segfile):
        return "No segfile was passed or it does not exist."
    if not hasattr(args, 'normfile') or not os.path.exists(args.normfile):
        return "No normfile was passed or it does not exist."
    if not hasattr(args, 'segstatsfile'):
        return "No segstats file was passed"

    device = args.device if _HAS_TORCH and hasattr(args, 'device') and args.device is not None else 'cpu'
    if device == 'auto':
        device = 'cpu'  # 'cuda' if _HAS_TORCH and torch.cuda.is_available() else 'cpu'
    if device != 'cpu':
        raise NotImplementedError("Only cpu is currently supported.")
    if _HAS_TORCH and device != "cpu":
        device = torch.device(device)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as tpe:
        # load these files in different threads to avoid waiting on IO (not parallel due to GIL though)
        seg_future = tpe.submit(loadfile_full, args.segfile, 'segfile', device=device)
        norm_future = tpe.submit(loadfile_full, args.normfile, 'normfile', device=device)

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
            if device == 'cpu':
                labels = np.unique(seg_data)
            else:
                labels = torch.unique(seg_data).cpu().numpy()

        if hasattr(args, 'excludeid') and args.excludeid is not None and len(args.excludeid) > 0:
            exclude_id = list(args.excludeid)
            labels = np.asarray(list(filter(lambda x: x not in exclude_id, labels)))
        else:
            exclude_id = []

        kwargs = {
            "vox_vol": np.prod(seg.header.get_zooms()).item(),
            "robust_percentage": args.robust if hasattr(args, 'robust') else None
        }

        if device == "cpu":
            if hasattr(args, 'legacy') and args.legacy:
                if kwargs.get("robust") is not None:
                    return "robust statistics are not supported in --legacy mode"
                if _HAS_NUMBA:
                    def make_pool():
                        return tpe
                else:
                    print("WARNING: Partial volume calculation in legacy mode without numba is VERY SLOW.")
                    from contextlib import contextmanager
                    from concurrent.futures import ProcessPoolExecutor
                    @contextmanager
                    def make_pool():
                        with ProcessPoolExecutor() as ppe:
                            yield ppe
                        return

                from multiprocessing import cpu_count
                all_borders = tpe.map(seg_borders, repeat(seg_data), labels,
                                      chunksize=np.ceil(len(labels) / cpu_count()))
                border: npt.NDArray[bool] = np.logical_or.reduce(list(all_borders), axis=0)
                with make_pool() as pool:
                    table = list(pool.map(partial(legacy_pv_calc, vox_vol=kwargs["vox_vol"]),
                                          repeat(border), repeat(seg_data.astype(np.int32)),
                                          repeat(norm_data.astype(np.float)), labels,
                                          chunksize=np.ceil(len(labels) / cpu_count())))
            else:
                table = pv_calc(seg_data, norm_data, labels, patch_size=args.patch_size, **kwargs)
        else:
            table = pv_calc_torch(seg_data, norm_data.to(torch.float), labels, **kwargs)

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
    if _HAS_TORCH and isinstance(_array, torch.Tensor):
        bin_array = _array if _array.dtype is torch.bool else _array == label

        def _laplace(data):
            raise NotImplementedError("laplace for Tensor not implemented")
    else:
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
    is_tensor = _HAS_TORCH and isinstance(_array, torch.Tensor)
    array_alloc = _array.new_full if is_tensor else partial(np.full, dtype=_array.dtype)
    _shape_plus2 = [s + 2 for s in _array.shape]

    if labels is True:  # already binarized
        if not (is_tensor and _array.dtype is torch.bool) and not (not is_tensor and np.issubdtype(_array, bool)):
            raise ValueError("If labels is True, the array should be boolean.")
        cmp = torch.logical_xor if is_tensor else np.logical_xor
    else:
        if (is_tensor and _array.dtype is torch.bool) or (not is_tensor and np.issubdtype(_array, bool)):
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
        arange = partial(torch.arange, device=_array.device) if is_tensor else np.arange
        lookup[labels] = arange(len(labels), dtype=lookup.dtype)
        _array = lookup[_array.to(torch.long)]
    logical_or = torch.logical_or if is_tensor else np.logical_or
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
    if is_tensor:
        for axis in range(dim - 1, 2, -1):
            nbr_same[axis - 1].logical_or_(nbr_same[axis])
        return torch.logical_or(nbr_same[0], nbr_same[1], out=out)
    else:
        return np.logical_or.reduce(nbr_same, out=out)


@overload
def legacy_pv_calc(border: npt.NDArray[bool], seg_array: npt.NDArray[_IntType], norm_array: npt.NDArray[np.uint8],
                   label: _IntType, vox_vol: float = 1.0, return_maps: False = False, maxlabels: int = 20_000) \
        -> PVStats:
    ...


@overload
def legacy_pv_calc(border: npt.NDArray[bool], seg_array: npt.NDArray[_IntType], norm_array: npt.NDArray[np.uint8],
                   label: _IntType, vox_vol: float = 1.0, return_maps: True = True, maxlabels: int = 20_000) \
        -> Tuple[PVStats, Dict[str, np.ndarray]]:
    ...


if _HAS_NUMBA:
    def numba_auto_cast_types(func):
        """helper function to cast the types of the call."""

        def auto_cast(var):
            if isinstance(var, np.ndarray):
                try:
                    numpy_support.from_dtype(var.dtype)
                except numba.errors.NumbaNotImplementedError:
                    if np.issubdtype(var.dtype, np.integer):
                        return var.astype('int')
                    if np.issubdtype(var.dtype, float):
                        return var.astype('float')
            return var

        @wraps(func)
        def wrapper_func(*args, **kwargs):
            args = [auto_cast(a) for a in args]
            kwargs = {k: auto_cast(a) for k, a in kwargs.items()}

            return func(*args, **kwargs)

        return wrapper_func


    def __pv_calc_signatures():
        """Numba Signatures for the numba_pv_calc function."""
        nbt = numba.types
        _nb_Bool3d, _nb_Float3d, _nb_Int3d = nbt.boolean[:, :, :], nbt.double[:, :, :], nbt.int32[:, :, :]
        _nb_ReturnType = nbt.Tuple([nbt.int32, nbt.int_, nbt.double, nbt.double, nbt.double, nbt.double, nbt.double,
                                    _nb_Int3d, _nb_Float3d, _nb_Float3d, _nb_Float3d])
        return _nb_ReturnType(_nb_Bool3d, _nb_Int3d, _nb_Float3d, nbt.int_, nbt.double, nbt.boolean, nbt.int_)


    def __nbhd_signatures():
        """Numa Signatures for the mri_compute_label_nbhd function."""
        nbt = numba.types
        _nb_ReturnType = nbt.Tuple([nbt.int_[:], nbt.double[:]])
        return _nb_ReturnType(nbt.int32[:, :, :], nbt.double[:, :, :], nbt.int_, nbt.int_, nbt.int_, nbt.int_, nbt.int_)


    def __nbhd_nomean_signatures():
        """Numa Signatures for the mri_compute_label_nbhd_no_mean function."""
        nbt = numba.types
        return nbt.int_[:](nbt.int32[:, :, :], nbt.int_, nbt.int_, nbt.int_, nbt.int_, nbt.int_)


    @numba.njit(cache=True)
    def numba_unique_with_counts(array):
        """Similar to `numpy.unique(array, return_counts=True)`."""
        # https://github.com/numba/numba/pull/2959
        b = np.sort(array.ravel())
        unique = list(b[:1])
        counts = [1 for _ in unique]
        for x in b[1:]:
            if x != unique[-1]:
                unique.append(x)
                counts.append(1)
            else:
                counts[-1] += 1
        return np.asarray(unique), np.asarray(counts)


    @numba.njit(cache=True)
    def numba_repeat_axis0(array, repeats):
        """Similar to `numpy.repeat(array, repeats, axis=0)`."""
        r = np.empty((repeats,) + array.shape, dtype=array._dtype)
        for i in range(repeats):
            r[i] = array
        return r


    @numba.njit(__nbhd_signatures(), nogil=True, cache=True)
    def mri_compute_label_nbhd(seg_data: npt.NDArray[_IntType], norm_data: Optional[np.ndarray],
                               x: int, y: int, z: int, whalf: int = 1, maxlabels: int = 20_000) \
            -> Tuple[npt.NDArray[int], Optional[npt.NDArray[float]]]:
        """Numba-compiled version of `mri_compute_label_nbhd(seg_data, norm_data, x, y, z, whalf, maxlabels)`."""
        label_counts = np.zeros(maxlabels, dtype='int')
        sub_array = seg_data[x - whalf:x + whalf + 1, y - whalf:y + whalf + 1, z - whalf:z + whalf + 1]
        elems, counts = numba_unique_with_counts(sub_array)
        label_counts[elems] = counts

        if norm_data is not None:
            label_means = np.zeros(maxlabels, dtype='float')
            norm_sub_array = norm_data[x - whalf:x + whalf + 1, y - whalf:y + whalf + 1, z - whalf:z + whalf + 1]
            for e in elems:
                label_means[e] = np.sum(norm_sub_array * (sub_array == e)) / label_counts[e]
        else:
            label_means = None
        return label_counts, label_means


    @numba.njit(__nbhd_nomean_signatures(), nogil=True, cache=True)
    def mri_compute_label_nbhd_no_mean(seg_data: npt.NDArray[_IntType],
                                       x: int, y: int, z: int, whalf: int = 1, maxlabels: int = 20_000) \
            -> npt.NDArray[int]:
        """Numba-compiled version of `mri_compute_label_nbhd(seg_data, norm_data, x, y, z, whalf, maxlabels)`."""
        label_counts = np.zeros(maxlabels, dtype='int')
        sub_array = seg_data[x - whalf:x + whalf + 1, y - whalf:y + whalf + 1, z - whalf:z + whalf + 1]
        elems, counts = numba_unique_with_counts(sub_array)
        label_counts[elems] = counts
        return label_counts


    @numba_auto_cast_types
    def legacy_pv_calc(border: npt.NDArray[bool], seg: npt.NDArray[np.int_], norm: npt.NDArray[np.uint8],
                       label: int, vox_vol: float = 1.0, return_maps: bool = False, maxlabels: int = 20_000) \
            -> Union[PVStats, Tuple[PVStats, Dict[str, np.ndarray]]]:
        """Calculate PV effects and volume statistics compatible with FreSurfer's mri_seg_stats for a specific label.
        It optionally returns the partial volume maps (see below).

        This function not only produces the same results as FreeSurfer's mri_seg_stats, but also does it using the same
        algorithm (i.e. it is just a port of the algorithm).

        Args:
            border: The global border of the image (which voxels should be considered for PV effects).
            seg: The segmentation map of the primary class per voxel.
            norm: The bias-field corrected T1w image (intensity image).
            label: The class (from seg) to analyze for PV effects.
            vox_vol: The volume per voxel in mm (default: 1.0).
            return_maps: Whether partial volume maps should be returned (default: False).
            maxlabels: The biggest allowed label index (in seg) (default: 20000)

        Returns:
            Segmentation Statistics: A Dictionary of:
                SegId: The label/class
                NVoxels: The number of voxels for label in seg
                Volume_mm3: The volume of label after partial volume correction
                StructName: (empty) -- never populated, use the lut to populate, see for example main()
                normMean: The average intensity in the bias-field corrected image of voxels labeled as label
                normMin: The minimum intensity in the bias-field corrected image of voxels labeled as label
                normMax: The maximum intensity in the bias-field corrected image of voxels labeled as label
                normStdDev: The standard deviation of the intensities in the bias-field corrected image of voxels labeled as label
                normRange: The range of intensities in the bias-field corrected image of voxels labeled as label
            If return_maps is True, also returns: Partial Volume Maps: A Dictionary of:
                nbr: An image of alternative labels that were considered instead of the voxel's label
                nbrmean: The local mean intensity of the label nbr at the specific voxel
                segmean: The local mean intensity of the current label at the specific voxel
                pv: The partial volume map of the current label

        Note:
            This function is just a wrapper to numba-compiled function to harmonize data types and provide a consistent
            interface. """
        label, _voxel_count, volumes, _mean, _std, _min, _max, full_nbr_label, full_nbr_mean, full_seg_mean, full_pv = \
            __legacy_pv_calc_impl(border, seg, norm, label,
                                  vox_vol=vox_vol, return_maps=return_maps, maxlabels=maxlabels)

        result = {"SegId": int(label), "NVoxels": int(_voxel_count), "Volume_mm3": float(volumes),
                  "StructName": "", "normMean": _mean, "normStdDev": _std, "normMin": _min, "normMax": _max,
                  "normRange": _max - _min}
        if return_maps:
            maps = {"nbr": full_nbr_label, "nbrmean": full_nbr_mean, "segmean": full_seg_mean, "pv": full_pv}
            return result, maps

        return result


    @numba.njit(__pv_calc_signatures(), parallel=True, nogil=True, cache=True)
    def __legacy_pv_calc_impl(border: npt.NDArray[bool], seg_array: npt.NDArray[np.int_],
                              norm_array: npt.NDArray[_NumberType],
                              label: int, vox_vol: float, return_maps: bool, maxlabels: int) \
            -> Tuple[int, int, float, float, float, float, float, npt.NDArray[np.int_], npt.NDArray[float], npt.NDArray[
                float], npt.NDArray[float]]:

        label, _min, _max = np.int32(label), np.infty, -np.infty
        _voxel_count, _sum, _sum_2, volumes = 0, 0., 0., 0.
        if return_maps:
            full_nbr_label = np.zeros(seg_array.shape, dtype=seg_array.dtype)
            full_nbr_mean = np.zeros(norm_array.shape, dtype='float')
            full_seg_mean = np.zeros(norm_array.shape, dtype='float')
            full_pv = np.zeros(norm_array.shape, dtype='float')

        # interestingly, explicitly parallelization of the second loop outperforms other versions (pndindex etc.)
        # by clearly ~20%
        # for x, y in pndindex(seg_array.shape[:2]):
        for x in range(seg_array.shape[0]):
            for y in numba.prange(seg_array.shape[1]):
                for z in range(seg_array.shape[2]):
                    pos = (x, y, z)
                    vox_label = seg_array[pos]
                    border_val = border[pos]

                    # Addition for other stats:
                    val = norm_array[pos]
                    if val < _min:
                        _min = val
                    if val > _max:
                        _max = val
                    _sum += val
                    _sum_2 += val * val
                    _voxel_count += 1

                    if border_val == 0:
                        # not a border voxel ...
                        if vox_label == label:
                            # ... and this voxel is `label`
                            volumes += vox_vol
                        if return_maps:
                            if vox_label != label:
                                # ... and this voxel is not `label`
                                full_nbr_label[pos] = -1
                            else:
                                # ... and this voxel is `label`
                                full_pv[pos] = 1.
                                full_nbr_label[pos] = -2
                    else:
                        # a border voxel
                        nbr_label_counts = mri_compute_label_nbhd_no_mean(seg_array, *pos, 1, maxlabels)
                        label_counts, label_means = mri_compute_label_nbhd(seg_array, norm_array, *pos, 7,
                                                                           maxlabels)

                        mean_label = label_means[vox_label]

                        dist_array = (label_means - val) * (mean_label - val)
                        label_counts[nbr_label_counts == 0] = 0
                        label_counts[dist_array >= 0] = 0
                        label_counts[vox_label] = 0

                        nbr_label = label_counts.argmax()
                        max_count = label_counts[nbr_label]

                        if return_maps:
                            full_seg_mean[pos] = mean_label
                            full_nbr_mean[pos] = label_means[nbr_label]
                            full_nbr_label[pos] = nbr_label
                        if vox_label != label and nbr_label != label:
                            # border voxel, but neither this nor the neighbor are `label`
                            continue

                        if max_count == 0:
                            # border voxel and there are no other "options"
                            volumes += vox_vol
                            if return_maps:
                                full_pv[pos] = 1.
                                full_seg_mean[pos] = mean_label
                                full_nbr_label[pos] = -4
                        else:
                            # border and there are options
                            mean_nbr = label_means[nbr_label]
                            if return_maps:
                                full_nbr_label[pos] = nbr_label
                                full_nbr_mean[pos] = mean_nbr

                            pv = (val - mean_nbr) / (mean_label - mean_nbr)
                            pv = pv if pv <= 1.0 else 1.0

                            if pv < 0:
                                continue

                            if vox_label != label:
                                pv = 1 - pv
                            if return_maps:
                                full_pv[pos] = pv

                            volumes += vox_vol * pv

        _mean = _sum / _voxel_count
        _std = np.sqrt((_mean * _mean * (_voxel_count - 2 / _voxel_count) + _sum_2) / (_voxel_count - 1))
        if not return_maps:
            full_nbr_label = np.zeros_like(seg_array[:1, :1, :1])
            full_nbr_mean = full_pv = full_seg_mean = np.zeros_like(seg_array[:1, :1, :1], dtype='float')

        return label, _voxel_count, volumes, _mean, _std, _min, _max, full_nbr_label, full_nbr_mean, full_seg_mean, full_pv

else:

    def mri_compute_label_nbhd(seg_data: npt.NDArray[_IntType], norm_data: Optional[np.ndarray],
                               x: int, y: int, z: int, whalf: int = 1, maxlabels: int = 20_000) \
            -> Tuple[npt.NDArray[int], npt.NDArray[float]]:
        """Port of the c++ function mri_compute_label_nbhd from mri.cpp of FreeSurfer."""
        # Almost 1-to-1 port of the function
        # def mri_compute_label_nbhd(mri, mri_vals, x, y, z, label_counts, label_means, whalf=1, maxlabels=20000):
        #     label_counts = np.zeros_like(label_counts)
        #     label_means = np.zeros_like(label_means)
        #     for xi in range(x - whalf, x + whalf + 1):
        #         for yi in range(y - whalf, y + whalf + 1):
        #             for zi in range(z - whalf, z + whalf + 1):
        #                 if (not 0 <= xi <= 255) or (not 0 <= yi <= 255) or (not 0 <= zi <= 255):
        #                     continue
        #                 label = mri[xi, yi, zi]
        #                 if label < 0 or label >= maxlabels:
        #                     continue
        #                 label_counts[label] += 1
        #                 if mri_vals is not None:
        #                     val = mri_vals[xi, yi, zi]
        #                     label_means[label] += val
        #     label_means = np.nan_to_num(label_means / label_counts)
        #     return label_counts, label_means
        # Optimized vectorized implementation:
        label_counts = np.zeros(maxlabels, dtype=int)
        label_means = np.zeros(maxlabels, dtype=float)
        sub_array = seg_data[x - whalf:x + whalf + 1, y - whalf:y + whalf + 1, z - whalf:z + whalf + 1]
        elems, counts = np.unique(sub_array, return_counts=True)
        label_counts[elems] = counts
        if norm_data is not None:
            norm_sub_array = norm_data[x - whalf:x + whalf + 1, y - whalf:y + whalf + 1, z - whalf:z + whalf + 1]
            # both sum and where with np.newaxis are considerable faster than mean and expand_dims
            label_means[elems] = np.sum(np.repeat(norm_sub_array[np.newaxis], len(elems), axis=0),
                                        where=sub_array[np.newaxis] == np.asarray(elems)[:, np.newaxis, np.newaxis,
                                                                       np.newaxis],
                                        axis=(1, 2, 3)) / label_counts[elems]

        return label_counts, label_means


    def legacy_pv_calc(border: npt.NDArray[bool], seg_array: npt.NDArray[_IntType], norm_array: np.ndarray,
                       label: _IntType, vox_vol: float = 1.0, return_maps: bool = False, maxlabels: int = 20_000) \
            -> Union[PVStats, Tuple[PVStats, Dict[str, np.ndarray]]]:
        """mri_seg_stats from FreeSurfer equivalent."""

        label, _min, _max = np.int32(label), np.infty, -np.infty
        _voxel_count, _sum, _sum_2, volumes = 0, 0., 0., 0.
        if return_maps:
            full_nbr_label = np.zeros_like(seg_array)
            full_nbr_mean = np.zeros_like(norm_array, dtype=float)
            full_seg_mean = np.zeros_like(norm_array, dtype=float)
            full_pv = np.zeros_like(norm_array, dtype=float)

        for x in range(seg_array.shape[0]):
            for y in range(seg_array.shape[1]):
                for z in range(seg_array.shape[2]):

                    pos = (x, y, z)
                    vox_label = seg_array[x, y, z]
                    border_val = border[x, y, z]

                    # Addition for other stats:
                    val = norm_array[x, y, z]
                    if val < _min:
                        _min = val
                    if val > _max:
                        _max = val
                    _sum += val
                    _sum_2 += (val ** 2)
                    _voxel_count += 1

                    if border_val == 0:
                        # not a border voxel ...
                        if vox_label != label:
                            # ... and this voxel is not `label`
                            if return_maps:
                                full_nbr_label[pos] = -1
                        else:
                            # ... and this voxel is `label`
                            volumes += vox_vol
                            if return_maps:
                                full_pv[pos] = 1.
                                full_nbr_label[pos] = -2
                    else:
                        # a border voxel
                        nbr_label_counts, _ = mri_compute_label_nbhd(seg_array, None, x, y, z, 1, maxlabels)
                        label_counts, label_means = mri_compute_label_nbhd(seg_array, norm_array, x, y, z, 7, maxlabels)

                        mean_label = label_means[vox_label]

                        dist_array = (label_means - val) * (mean_label - val)
                        label_counts[nbr_label_counts == 0] = 0
                        label_counts[dist_array >= 0] = 0
                        label_counts[vox_label] = 0

                        nbr_label = label_counts.argmax().astype('int32')
                        max_count = label_counts[nbr_label]

                        if return_maps:
                            full_seg_mean[pos] = mean_label
                            full_nbr_mean[pos] = label_means[nbr_label]
                            full_nbr_label[pos] = nbr_label
                        if vox_label != label and nbr_label != label:
                            # border voxel, but neither this nor the neighbor are `label`
                            continue

                        if max_count == 0:
                            # border voxel and there are no other "options"
                            volumes += vox_vol
                            if return_maps:
                                full_pv[pos] = 1.
                                full_seg_mean[pos] = mean_label
                                full_nbr_label[pos] = -4
                        else:
                            # border and there are options
                            mean_nbr = label_means[nbr_label]
                            if return_maps:
                                full_nbr_label[pos] = nbr_label
                                full_nbr_mean[pos] = mean_nbr
                            pv = min((val - mean_nbr) / (mean_label - mean_nbr), 1.0)
                            if pv < 0:
                                continue

                            if vox_label != label:
                                pv = 1 - pv
                            if return_maps:
                                full_pv[pos] = pv

                            volumes += vox_vol * pv
        _mean = _sum / _voxel_count
        _std = np.sqrt((_mean * _mean * (_voxel_count - 2 / _voxel_count) + _sum_2) / (_voxel_count - 1))
        result = {"SegId": int(label), "NVoxels": int(_voxel_count), "Volume_mm3": float(volumes),
                  "StructName": "", "normMean": _mean, "normStdDev": _std, "normMin": _min, "normMax": _max,
                  "normRange": _max - _min}
        if return_maps:
            maps = {"nbr": full_nbr_label, "nbrmean": full_nbr_mean, "segmean": full_seg_mean, "pv": full_pv}
            return result, maps
        return result


def unsqueeze(matrix, axis: Union[int, Sequence[int]] = -1):
    """Allows insertions of axis into the data/tensor, see numpy.expand_dims. This expands the torch.unsqueeze
    syntax to allow unsqueezing multiple axis at the same time. """
    if isinstance(matrix, np.ndarray):
        return np.expand_dims(matrix, axis=axis)
    elif _HAS_TORCH and torch.is_tensor(matrix):
        _axis = axis if isinstance(axis, Sequence) else (axis,)
        for dim in _axis:
            matrix = matrix.unsqueeze(dim)
        return matrix


if _HAS_TORCH and _version.parse(torch.__version__) >= _version.parse("1.11"):
    def _round(_arr: _ArrayType, decimals: int = 0) -> _ArrayType:
        return _arr.round(decimals=decimals)
else:
    def _round(_arr: _ArrayType, decimals: int = 0) -> _ArrayType:
        if _HAS_TORCH and isinstance(_arr, torch.Tensor):
            return torch.round(_arr / 10 ** decimals) * 10 ** decimals
        else:
            return np.round(_arr, decimals=decimals)


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
    if isinstance(arr, np.ndarray):
        _patch = (slice(None),) if patch is None else patch
        arr = arr.astype(float)
        from scipy.ndimage import uniform_filter

        def _uniform_filter(_arr):
            return uniform_filter(_arr, size=filter_size, mode='constant', cval=0)[_patch]
    else:
        weight = torch.full((1,1) + (filter_size,) * arr.ndim, 1 / (filter_size ** arr.ndim), device=arr.device)
        arr = arr.to(torch.float)

        def _uniform_filter(_arr, out = None):
            kw = {} if out is None else {'out': unsqueeze(out, [0, 0])}
            return _Conv[arr.ndim](unsqueeze(_arr, [0, 0]),
                                   weight=weight, stride=[1]*_arr.ndim, padding='same',
                                   **kw).squeeze(0).squeeze(0)
    if out is not None:
        _uniform_filter(arr, out)
        return out
    return _uniform_filter(arr)


def torch_pad_crop_for_conv(arr, filter_size, patch):
    """Helper function to pad and crop an array for a torch convolution."""
    whalf = int((filter_size - 1) / 2)
    if patch is None:
        _patch = (slice(None),)
        pad = (whalf,) * (2 * arr.ndim)
    else:
        _patch, pad = [], []
        for s, shape in zip(patch, arr.shape):
            _start0 = 0 if s.start is None else s.start
            if _start0 < 0:
                raise NotImplementedError('uniform_filter does not support negative offsets for the slice start')
            if s.step not in (1, None):
                raise NotImplementedError('uniform_filter does not support slices steps unequal to 1')
            _start = max(_start0 - whalf, 0)
            _stop = shape if s.stop is None else shape
            if _stop >= 0:
                _stop = min(_stop + whalf, shape)
                _pad = whalf - shape + _stop
            else:
                delta = min(_stop + whalf, 0)
                _stop = shape + delta
                _pad = max(0, whalf - delta)
            _patch.append(slice(_start, _stop))
            pad.extend([whalf - _start + _start0, _pad])
        _patch = tuple(_patch)
    if any(p > 0 for p in pad):
        arr = _F.pad(unsqueeze(arr[_patch], (0,) * (arr.ndim + 2 - len(_patch))), pad, mode="constant", value=0)
    return arr


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
            return_maps: bool = False) \
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

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as pool:

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


def patch_filter(pos: Tuple[int, int, int], mask: Union[npt.NDArray[bool], 'torch.BoolTensor'],
                 global_crop: Tuple[slice, ...], patch_size: int = 32) \
        -> Tuple[bool, Sequence[slice]]:
    """Returns, whether there are mask-True voxels in the patch starting at pos with size patch_size and the resulting
    patch shrunk to mask-True regions."""
    # create slices for current patch context (constrained by the global_crop)
    patch = [slice(p, min(p + patch_size, slice_.stop)) for p, slice_ in zip(pos, global_crop)]
    # crop patch context to the image content
    return crop_patch_to_mask(mask, sub_patch=patch)


def crop_patch_to_mask(mask: Union[npt.NDArray[_NumberType], 'torch.BoolTensor'],
                       sub_patch: Optional[Sequence[slice]] = None) \
        -> Tuple[bool, Sequence[slice]]:
    """Crop the patch to regions of the patch that are non-zero. Assumes mask is always positive. Returns whether there
    is any mask>0 in the patch and a patch shrunk to mask>0 regions.

    Args:
        mask: to crop to
        sub_patch: subregion of mask to only consider (default: full mask)

    Note:
        This function requires device synchronization."""

    is_tensor = _HAS_TORCH and torch.is_tensor(mask)

    def axis_kw(axis):
        return {"dim" if is_tensor else "axis": axis}

    _patch = []
    patch = tuple([slice(0, s) for s in mask.shape] if sub_patch is None else sub_patch)
    patch_in_patch_coords = tuple([slice(0, slice_.stop - slice_.start) for slice_ in patch])
    in_mask = True
    _mask = mask[patch].sum(**axis_kw(2))
    for i, pat in enumerate(patch_in_patch_coords):
        p = pat.start
        if in_mask:
            if i == 2:
                _mask = mask[patch][tuple(_patch)].sum(**axis_kw(0))
            # can we shrink the patch context in i-th axis?
            pat_has_mask_in_axis = _mask[tuple(_patch[1:] if i != 2 else [])].sum(**axis_kw(int(i == 0))) > 0
            # modify both the _patch_size and the coordinate p to shrink the patch
            _pat_mask = pat_has_mask_in_axis.nonzero() if is_tensor else np.argwhere(pat_has_mask_in_axis)
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
    max_counts_index = _round(pat1d_label_counts * is_valid, log_eps).argmax(axis=0, keepdims=False)

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


if _HAS_TORCH:
    @overload
    def pv_calc_torch(seg: 'torch.IntTensor', norm: 'torch.Tensor', labels: Sequence[_IntType],
                      vox_vol: float = 1.0, eps: float = 1e-6, robust_percentage: Optional[float] = None,
                      return_maps: False = False) -> List[PVStats]:
        ...


    @overload
    def pv_calc_torch(seg: 'torch.IntTensor', norm: 'torch.Tensor', labels: Sequence[_IntType],
                      vox_vol: float = 1.0, eps: float = 1e-6, robust_percentage: Optional[float] = None,
                      return_maps: True = True) \
            -> Tuple[List[PVStats], Dict[str, Dict[int, np.ndarray]]]:
        ...


    def pv_calc_torch(seg: 'torch.IntTensor', norm: 'torch.Tensor', labels: Sequence[_IntType],
                      vox_vol: float = 1.0, eps: float = 1e-6, robust_percentage: Optional[float] = None,
                      return_maps: bool = False) \
            -> Union[List[PVStats], Tuple[List[PVStats], Dict[str, np.ndarray]]]:
        """torch-port of pv_calc(). As opposed to pv_calc, this function requires Tensors for seg and norm."""

        if not isinstance(seg, torch.Tensor) or not isinstance(norm, torch.Tensor):
            raise ValueError("Either seg or norm are not IntTensors or Tensors, respectively.")

        if seg.device != norm.device:
            raise ValueError("seg and norm are not on the same device.")

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as tpe:
            # executed in different thread as it causes host-device synchronization
            labels_in_seg = tpe.submit(torch.unique, seg)

            log_eps = -int(np.log10(eps))
            teps = seg.new_full((1,), fill_value=eps, dtype=torch.float)
            mins, maxes, voxel_counts, __voxel_counts, sums, sums_2, volumes = [{} for _ in range(7)]

            # initialize global_crop with the full image
            global_crop: Tuple[slice, ...] = tuple(slice(0, _shape) for _shape in seg.shape)
            # ignore all regions of the image that are background only
            if 0 not in labels:
                # crop global_crop to the data
                any_in_global, global_crop = crop_patch_to_mask(cast(torch.BoolTensor, seg != 0), sub_patch=global_crop)
                # grow global_crop by one, so all border voxels are included
                global_crop = grow_patch(global_crop, 1, seg.shape)[0]
                if not any_in_global:
                    raise RuntimeError("Segmentation map only consists background")

            labels_in_seg = labels_in_seg.result()
            relevant_labels = list(sorted(list(set(labels_in_seg.cpu().tolist()).intersection([0] + list(labels)))))
            max_label = labels_in_seg[-1].item()
            border = borders(seg, relevant_labels, max_label=max_label)
            # executed in different thread as it causes host-device synchronization
            norm_of_border = tpe.submit(torch.masked_select, norm, border)
            if return_maps:
                indices_of_border = tpe.submit(torch.nonzero, border.flatten())
            indices_in_flat = border.to(torch.int, copy=True).flatten().cumsum_(0).sub_(1).reshape(border.shape)
            num_border: int = indices_in_flat[tuple(-1 for _ in range(border.ndim))].item() + 1
            norm_of_border = norm_of_border.result()

            # allocate memory for the means and counts (first dim must be `labels_in_seg` for consistent results)
            means_of_border = norm.new_full((labels_in_seg.shape[0], num_border), fill_value=0, dtype=torch.float)
            counts_of_border = norm.new_full((labels_in_seg.shape[0], num_border), fill_value=0, dtype=torch.float)
            mask_by_6border = norm.new_full((labels_in_seg.shape[0], num_border), fill_value=False, dtype=torch.bool_)

            label_stats_filled = partial(label_stats_torch, norm=norm[global_crop], seg=seg[global_crop],
                                         indices=indices_in_flat[global_crop], border=border[global_crop],
                                         robust_keep_fraction=robust_percentage, eps=eps)
            # this map must be run on `labels_in_seg` (including not-asked for labels to make results consistent between
            # different selections of labels, i.e. parameters `labels`)
            # TODO: tpe.map
            label_stats = map(label_stats_filled, labels_in_seg, counts_of_border, means_of_border, mask_by_6border)
            crop2label, this_6border = {}, {}

            for lab, *data in label_stats:
                if data[0] != 0:
                    voxel_counts[lab], __voxel_counts[lab] = data[:2]
                    mins[lab], maxes[lab], sums[lab], sums_2[lab] = data[2:6]
                    volumes[lab], this_6border[lab] = data[6] * vox_vol, data[7]

            seg_of_border = seg[border]
            seg_mean_of_border = torch.gather(means_of_border, dim=0, index=labels_in_seg[seg_of_border])
            if return_maps:
                keys_of_maps = ["nbr", "segmean", "nbrmean", "pv", "ipv"]
                dtypes, defaults = {'nbr': torch.int}, {"pv": 1}
                maps = {key: seg.new_full(seg.shape,
                                          fill_value=defaults.get(key, 0),
                                          dtype=dtypes.get(key, torch.float)) for key in keys_of_maps}
                indices_of_border = indices_of_border.result()
                maps["segmean"][indices_of_border] = seg_mean_of_border

            # 1. considered (mean of) alternative label must be on the other side of norm as the (mean of) the
            # segmentation label of the current voxel
            is_switched_sign = cast(torch.BoolTensor, means_of_border > norm_of_border.unsqueeze(0)) \
                .logical_xor(cast(torch.BoolTensor(seg_mean_of_border > norm_of_border).unsqueeze(0)))

            # 3. (mean of) segmentation label must be different to norm of voxel
            is_valid = is_switched_sign.logical_and(unsqueeze(seg_mean_of_border != norm_of_border, 0))
            none_valid = ~is_valid.any(dim=0, keepdim=False)

            # select the label, that is valid or not valid but also exists and is not the current label
            nbr_label_of_border = _round(counts_of_border * is_valid, log_eps).argmax(dim=0, keepdim=False)
            nbr_label_of_border[none_valid] = 0

            # get the mean label intensity of the "alternative label"
            nbr_mean_of_border = torch.take_along_dim(means_of_border, unsqueeze(nbr_label_of_border, 0), dim=0)[0]

            if return_maps:
                maps["nbr"][indices_of_border] = labels_in_seg[nbr_label_of_border]
                maps["nbrmean"][indices_of_border] = nbr_mean_of_border

            # interpolate between the "local" and "alternative label"
            mean_to_mean_nbr = seg_mean_of_border - nbr_mean_of_border
            delta_gt_eps = mean_to_mean_nbr.abs() > teps
            # make sure no division by zero
            pv_of_border = (norm_of_border - nbr_mean_of_border) / mean_to_mean_nbr.where(delta_gt_eps, teps)

            pv_of_border[~delta_gt_eps] = 1.  # set pv fraction to 1 if division by zero
            pv_of_border[none_valid] = 1.  # set pv fraction to 1 for voxels that have no 'valid' nbr
            pv_of_border[pv_of_border > 1.] = 1.
            pv_of_border[pv_of_border < 0.] = 0.

            # re-create the "supposed" freesurfer inconsistency that does not count vertex neighbors, if the voxel label
            # is not of question
            ipv_of_border = (1. - pv_of_border) * mask_by_6border

            if return_maps:
                maps["pv"][indices_of_border] = pv_of_border
                maps["ipv"][indices_of_border] = ipv_of_border

            for lab in relevant_labels:
                pv_vol = (pv_of_border * (seg_of_border == lab)).sum()
                ipv_vol = (ipv_of_border * (nbr_label_of_border == lab)).sum()
                volumes[lab] += (pv_vol + ipv_vol) * vox_vol

        # numpy algo
        # global_stats_filled = partial(global_stats_torch,
        #                               norm=norm[global_crop], seg=seg[global_crop],
        #                               robust_keep_fraction=robust_keep_fraction)
        # global_stats = map(global_stats_filled, labels)
        #
        # if return_maps:
        #     _ndarray_alloc = seg.new_full
        #     full_nbr_label = _ndarray_alloc(seg.shape, fill_value=0, dtype=seg.dtype)
        #     full_nbr_mean = _ndarray_alloc(norm.shape, fill_value=0, dtype=torch.float)
        #     full_seg_mean = _ndarray_alloc(norm.shape, fill_value=0, dtype=torch.float)
        #     full_pv = _ndarray_alloc(norm.shape, fill_value=1, dtype=torch.float)
        #     full_ipv = _ndarray_alloc(norm.shape, fill_value=0, dtype=torch.float)
        # else:
        #     full_nbr_label, full_seg_mean, full_nbr_mean, full_pv, full_ipv = [None] * 5
        #
        # for lab, *data in global_stats:
        #     if data[0] != 0:
        #         voxel_counts[lab], __voxel_counts[lab] = data[:2]
        #         mins[lab], maxes[lab], sums[lab], sums_2[lab] = data[2:-2]
        #         volumes[lab], loc_border[lab] = data[-2] * vox_vol, data[-1]
        #
        # # un_global_crop border here
        # _border = reduce(torch.logical_or, loc_border.values())
        # border = _F.pad(_border, tuple(chain.from_iterable((slc.start, shp - slc.stop) for slc, shp in zip(global_crop, seg.shape))))
        # if not np.array_equal(border.shape, seg.shape):
        #     raise RuntimeError("border and seg_array do not have same shape.")
        #
        # # iterate through patches of the image
        # patch_iters = [range(slice_.start, slice_.stop, patch_size) for slice_ in global_crop]  # for 3D
        #
        # _patches = map(partial(patch_filter, mask=border, global_crop=global_crop, patch_size=patch_size),
        #                product(*patch_iters))
        # patches = (patch for has_pv_vox, patch in _patches if has_pv_vox)
        #
        # for vols in map(partial(pv_calc_patch_torch, global_crop=global_crop, loc_border=loc_border, border=border,
        #                         seg=seg, norm=norm, full_nbr_label=full_nbr_label, full_seg_mean=full_seg_mean,
        #                         full_pv=full_pv, full_ipv=full_ipv, full_nbr_mean=full_nbr_mean, eps=eps),
        #                 patches):
        #     for lab in volumes.keys():
        #         volumes[lab] += vols.get(lab, 0.) * vox_vol

        def item(_arr: Dict[_IntType], lab: _IntType, default: _NumberType = 0) -> _NumberType:
            v = _arr.get(lab, default)
            if _HAS_TORCH and isinstance(v, torch.Tensor):
                return v.item()
            return v

        means = {lab: s / __voxel_counts[lab] for lab, s in sums.items() if item(__voxel_counts, lab) > eps}
        # *std = sqrt((sum * (*mean) - 2 * (*mean) * sum + sum2) / (nvoxels - 1));
        stds = {lab: ((sums_2[lab] - means[lab] * sums[lab]) / (nvox - 1)).sqrt() for lab, nvox in
                __voxel_counts.items() if nvox > 1}
        # ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
        table = [{"SegId": lab, "NVoxels": item(voxel_counts, lab), "Volume_mm3": item(volumes, lab, 0.),
                  "StructName": "", "normMean": item(means, lab, 0.), "normStdDev": item(stds, lab, 0.),
                  "normMin": item(mins, lab, 0.), "normMax": item(maxes, lab, 0.),
                  "normRange": item(maxes, lab, 0.) - item(mins, lab, 0.)} for lab in labels]
        if return_maps:
            return table, maps
        return table


    def global_stats_torch(lab: _IntType, norm: 'torch.Tensor', seg: 'torch.IntTensor',
                           out: Optional['torch.BoolTensor'] = None, robust_keep_fraction: Optional[float] = None) \
            -> Union[Tuple[_IntType, int],
                     Tuple[_IntType, int, int, _NumberType, _NumberType, float, float, float, 'torch.BoolTensor']]:
        """Computes Label, Number of voxels, 'robust' number of voxels, norm minimum, maximum, sum, sum of squares and
        6-connected border of label lab."""
        bin_array = cast(torch.BoolTensor, seg == lab)
        data = norm[bin_array].to(dtype=torch.int if np.issubdtype(norm.dtype, np.integer) else torch.float)
        nvoxels: int = data.shape[0]
        # if lab is not in the image at all
        if nvoxels == 0:
            return lab, 0
        # compute/update the border
        if out is None:
            out = seg_borders(bin_array, True).to(bool)
        else:
            out[:] = seg_borders(bin_array, True).to(bool)

        if robust_keep_fraction is not None:
            data = torch.sort(data)
            sym_drop_samples = int((1 - robust_keep_fraction / 2) * nvoxels)
            data = data[sym_drop_samples:-sym_drop_samples]
            _min: _NumberType = data[0]
            _max: _NumberType = data[-1]
            __voxel_count = nvoxels - 2 * sym_drop_samples
        else:
            _min = data.min()
            _max = data.max()
            __voxel_count = nvoxels
        _sum: float = data.sum()
        sum_2: float = (data * data).sum()
        # this is independent of the robustness criterium
        volume: float = torch.logical_and(bin_array, torch.logical_not(out)).sum().to(dtype=torch.float)
        return lab, nvoxels, __voxel_count, _min, _max, _sum, sum_2, volume, out


    def pv_calc_patch_torch(patch: Tuple[slice, ...], global_crop: Tuple[slice, ...],
                            loc_border: Dict[_IntType, 'torch.BoolTensor'],
                            seg: 'torch.IntTensor', norm: np.ndarray, border: 'torch.BoolTensor',
                            full_pv: Optional['torch.FloatTensor'] = None,
                            full_ipv: Optional['torch.FloatTensor'] = None,
                            full_nbr_label: Optional['torch.IntTensor'] = None,
                            full_seg_mean: Optional['torch.FloatTensor'] = None,
                            full_nbr_mean: Optional['torch.FloatTensor'] = None, eps: float = 1e-6) \
            -> Dict[_IntType, float]:
        """Calculates PV for patch. If full* keyword arguments are passed, also fills, per voxel results for the respective
        voxels in the patch."""

        log_eps = -int(np.log10(eps))
        teps = seg.new_full((1,), fill_value=eps, dtype=torch.float)

        patch = tuple(patch)
        patch_grow1, ungrow1_patch = grow_patch(patch, 1, seg.shape)
        patch_grow7, ungrow7_patch = grow_patch(patch, 7, seg.shape)
        patch_shrink6 = tuple(
            slice(ug7.start - ug1.start, None if ug7.stop == ug1.stop else ug7.stop - ug1.stop) for ug1, ug7 in
            zip(ungrow1_patch, ungrow7_patch))
        patch_in_gc = tuple(slice(p.start - gc.start, p.stop - gc.start) for p, gc in zip(patch, global_crop))

        label_lookup = torch.unique(seg[patch_grow1])
        maxlabels = label_lookup[-1] + 1
        if maxlabels > 100_000:
            raise RuntimeError("Maximum number of labels above 100000!")
        # create a view for the current patch border
        pat_border = border[patch]
        pat_is_border, pat_is_nbr, pat_label_counts, pat_label_sums \
            = patch_neighbors(label_lookup, norm, seg, pat_border, loc_border,
                              patch_grow7, patch_in_gc, patch_shrink6, ungrow1_patch, ungrow7_patch,
                              ndarray_alloc=seg.new_full, eps=eps)

        # both counts and sums are "normalized" by the local neighborhood size (15**3)
        label_lookup_fwd = seg.new_full((maxlabels,), fill_value=0, dtype="int")
        label_lookup_fwd[label_lookup] = torch.arange(label_lookup.shape[0])

        # shrink 3d patch to 1d list of border voxels
        pat1d_norm, pat1d_seg = norm[patch][pat_border], seg[patch][pat_border]
        pat1d_label_counts = pat_label_counts[:, pat_border]
        # both sums and counts are normalized by n-hood-size**3, so the output is not anymore
        pat1d_label_means = _round(pat_label_sums[:, pat_border] / torch.maximum(pat1d_label_counts, teps), log_eps)  # float

        # get the mean label intensity of the "local label"
        mean_label = torch.take_along_dim(pat1d_label_means, unsqueeze(label_lookup_fwd[pat1d_seg], 0), dim=0)[0]
        # get the index of the "alternative label"
        pat1d_is_this_6border = pat_is_border[:, pat_border]
        # calculate which classes to consider:
        is_valid = reduce(torch.logical_and,
                          # 1. considered (mean of) alternative label must be on the other side of norm as the
                          # (mean of) the segmentation label of the current voxel
                          [torch.logical_xor(cast(torch.BoolTensor, pat1d_label_means > pat1d_norm.unsqueeze(0)),
                                             cast(torch.BoolTensor, mean_label > pat1d_norm).unsqueeze(0)),
                           # 2. considered (mean of) alternative label must be different to norm of voxel
                           pat1d_label_means != pat1d_norm.unsqueeze(0),
                           # 3. (mean of) segmentation label must be different to norm of voxel
                           torch.broadcast_to(unsqueeze(mean_label != pat1d_norm, 0), pat1d_label_means.shape),
                           # 4. label must be a neighbor
                           pat_is_nbr[:, pat_border],
                           # 5. label must not be the segmentation
                           pat1d_seg.unsqueeze(0) != label_lookup.unsqueeze(1)])

        none_valid = ~is_valid.any(dim=0, keepdim=False)
        # select the label, that is valid or not valid but also exists and is not the current label
        max_counts_index = _round(pat1d_label_counts * is_valid, log_eps).argmax(dim=0, keepdim=False)

        nbr_label = label_lookup[max_counts_index]  # label with max_counts
        nbr_label[none_valid] = 0

        # get the mean label intensity of the "alternative label"
        mean_nbr = torch.take_along_dim(pat1d_label_means, unsqueeze(label_lookup_fwd[nbr_label], 0), dim=0)[0]

        # interpolate between the "local" and "alternative label"
        mean_to_mean_nbr = mean_label - mean_nbr
        delta_gt_eps = mean_to_mean_nbr.abs() > eps
        pat1d_pv = (pat1d_norm - mean_nbr) / mean_to_mean_nbr.where(delta_gt_eps, teps)  # make sure no division by zero

        pat1d_pv[~delta_gt_eps] = 1.  # set pv fraction to 1 if division by zero
        pat1d_pv[none_valid] = 1.  # set pv fraction to 1 for voxels that have no 'valid' nbr
        pat1d_pv[pat1d_pv > 1.] = 1.
        pat1d_pv[pat1d_pv < 0.] = 0.

        # re-create the "supposed" freesurfer inconsistency that does not count vertex neighbors, if the voxel label
        # is not of question
        mask_by_6border = torch.take_along_dim(pat1d_is_this_6border, unsqueeze(label_lookup_fwd[nbr_label], 0), dim=0)[
            0]
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

        return {lab: (pat1d_pv * (pat1d_seg == lab)).sum() + (pat1d_inv_pv * (nbr_label == lab)).sum() for lab in
                label_lookup}


    def label_stats_torch(label: _IntType, out_counts: 'torch.FloatTensor', out_means: 'torch.FloatTensor',
                          out_6border: Optional['torch.BoolTensor'],  seg: 'torch.IntTensor', norm: 'torch.Tensor',
                          indices: 'torch.LongTensor', border: 'torch.BoolTensor',
                          robust_keep_fraction: Optional[float] = None, eps: float = 1e-6):
        """Calculates PV for patch. If full* keyword arguments are passed, also fills, per voxel results for the respective
        voxels in the patch.

        Note:
            This function has internal device synchronization points (e.g. indexing by bin_lab)"""

        # function constants
        log_eps = -int(np.log10(eps))
        teps = seg.new_full((1,), fill_value=eps, dtype=torch.float)

        bin_lab = cast(torch.BoolTensor, seg == label)
        any_label, crop2label = crop_patch_to_mask(bin_lab)
        # if lab is not in the image at all, if called from main(), this should never happen
        if not any_label:
            return label, 0
        crop2label, _ = grow_patch(crop2label, 1, seg.shape)
        border_cropped = border[crop2label]
        if out_6border is not None:
            _6border = borders(bin_lab[crop2label], True)
        intensities_of_label = norm[crop2label][bin_lab[crop2label]]
        nvoxels: int = intensities_of_label.shape[0]

        # compute/update the border
        if robust_keep_fraction is not None:
            intensities_of_label = torch.sort(intensities_of_label)
            sym_drop_samples = int((1 - robust_keep_fraction / 2) * nvoxels)
            intensities_of_label = intensities_of_label[sym_drop_samples:-sym_drop_samples]
            _min: _NumberType = intensities_of_label[0]
            _max: _NumberType = intensities_of_label[-1]
            __voxel_count = nvoxels - 2 * sym_drop_samples
        else:
            _min = intensities_of_label.min()
            _max = intensities_of_label.max()
            __voxel_count = nvoxels
        _sum: float = intensities_of_label.sum()
        sum_2: float = (intensities_of_label * intensities_of_label).sum()
        # this is independent of the robustness criterium
        volume: float = torch.logical_and(bin_lab[crop2label], border_cropped.logical_not()).sum().to(dtype=torch.float)

        # implicitly also a border detection: is lab a neighbor of the "current voxel"
        # lab is at least once a nbr in the patch (grown by one)
        is_nbr = uniform_filter(bin_lab[crop2label], 3) > teps  # as float (*filter_size**3)
        label_counts = uniform_filter(bin_lab[crop2label], 15)  # as float (*filter_size**3)
        filtered_norm = norm[crop2label] * bin_lab[crop2label]

        label_means = _round(uniform_filter(filtered_norm, 15) / label_counts, log_eps)

        # conditions:
        # 4. label must be a neighbor (implicitly also border)
        alt_mask = is_nbr.logical_and(border_cropped)
        # 2. considered (mean of) alternative label must be different to norm of voxel
        alt_mask = alt_mask.logical_and(label_means != norm[crop2label])
        # 5. label must not be the segmentation (bin_lab is where label is the segmentation)
        alt_mask = alt_mask.logical_and(bin_lab[crop2label].logical_not())
        # for counts, we want all voxels that fulfill criteria 2., 4., 5.
        out_counts[indices[crop2label][alt_mask]] = label_counts[alt_mask]

        # for the means, we want all border voxels to be set
        indices_border = indices[crop2label][border_cropped]
        out_means[indices_border] = label_means[border_cropped]
        if out_6border is not None:
            out_6border[indices_border] = _6border[border_cropped]

        return label, nvoxels, __voxel_count, _min, _max, _sum, sum_2, volume, out_counts, out_means, out_6border


    def patch_neighbors(labels, norm, seg, pat_border, loc_border, patch_grow7, patch_in_gc, patch_shrink6,
                        ungrow1_patch, ungrow7_patch, ndarray_alloc, eps):
        """Helper function to calculate the neighbor statistics of labels, etc."""
        loc_shape = (len(labels),) + pat_border.shape

        pat_label_counts, pat_label_sums = ndarray_alloc((2,) + loc_shape, fill_value=0., dtype=float)
        pat_is_nbr, pat_is_border = ndarray_alloc((2,) + loc_shape, fill_value=False, dtype=bool)
        for i, lab in enumerate(labels):
            pat7_bin_array = cast(torch.BoolTensor, seg[patch_grow7] == lab)
            # implicitly also a border detection: is lab a neighbor of the "current voxel"
            tmp_nbr_label_counts = uniform_filter(pat7_bin_array[patch_shrink6], 3)  # as float (*filter_size**3)
            if _HAS_TORCH and isinstance(seg,
                                         torch.Tensor) or tmp_nbr_label_counts.sum() > eps:  # TODO: check if this actually faster for torch than doing the if
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
