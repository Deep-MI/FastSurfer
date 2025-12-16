#!/usr/bin/env python3

# Copyright 2021 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

from pathlib import Path
from typing import TypedDict

import numpy.typing as npt

from FastSurferCNN.utils import AffineMatrix4x4


class LTADict(TypedDict):
    """
    Typed dictionary containing all the information from an LTA transform file.

    Attributes
    ----------
    type: int
    nxforms: int
    mean: list[float]
    sigma: float
    lta: AffineMatrix4x4
    src_valid: int
    src_filename: str
    src_volume: list[int]
    src_voxelsize: list[float]
    src_xras: list[float]
    src_yras: list[float]
    src_zras: list[float]
    src_cras: list[float]
    dst_valid: int
    dst_filename: str
    dst_volume: list[int]
    dst_voxelsize: list[float]
    dst_xras: list[float]
    dst_yras: list[float]
    dst_zras: list[float]
    dst_cras: list[float]
    src: AffineMatrix4x4
    dst: AffineMatrix4x4
    """
    type: int
    nxforms: int
    mean: list[float]
    sigma: float
    lta: AffineMatrix4x4
    src_valid: int
    src_filename: str
    src_volume: list[int]
    src_voxelsize: list[float]
    src_xras: list[float]
    src_yras: list[float]
    src_zras: list[float]
    src_cras: list[float]
    dst_valid: int
    dst_filename: str
    dst_volume: list[int]
    dst_voxelsize: list[float]
    dst_xras: list[float]
    dst_yras: list[float]
    dst_zras: list[float]
    dst_cras: list[float]
    src: AffineMatrix4x4
    dst: AffineMatrix4x4


# Collection of functions related to FreeSurfer's LTA (linear transform array) files:
def write_lta(
        filename: Path | str,
        affine: npt.ArrayLike,
        src_fname: Path | str,
        src_header: dict,
        dst_fname: Path | str,
        dst_header: dict
) -> None:
    """
    Write linear transform array info to an .lta file.

    Parameters
    ----------
    filename : Path, str
        File to write on.
    affine : npt.ArrayLike
        Linear transform array to be saved.
    src_fname : Path, str
        Source filename.
    src_header : Dict
        Source header.
    dst_fname : Path, str
        Destination filename.
    dst_header : Dict
        Destination header.

    Raises
    ------
    ValueError
        Header format missing field (Source or Destination).
    """
    import getpass
    from datetime import datetime

    fields = ("dims", "delta", "Mdc", "Pxyz_c")
    for field in fields:
        if field not in src_header:
            raise ValueError(f"write_lta Error: src_header format missing field: {field}")
        if field not in dst_header:
            raise ValueError(f"write_lta Error: dst_header format missing field: {field}")

    src_dims = str(src_header["dims"][0:3])
    src_vsize = str(src_header["delta"][0:3])
    src_v2r = src_header["Mdc"]
    src_c = src_header["Pxyz_c"]

    dst_dims = str(dst_header["dims"][0:3])
    dst_vsize = str(dst_header["delta"][0:3])
    dst_v2r = dst_header["Mdc"]
    dst_c = dst_header["Pxyz_c"]

    with open(filename, "w") as f:
        f.write(
            (f"# transform file {filename}\n"
            f"# created by {getpass.getuser()} on {datetime.now().ctime()}\n\n"
            "type      = 1 # LINEAR_RAS_TO_RAS\n"
            "nxforms   = 1\n"
            "mean      = 0.0 0.0 0.0\n"
            "sigma     = 1.0\n"
            "1 4 4\n"
            f"{affine}\n"
            "src volume info\n"
            "valid = 1  # volume info valid\n"
            f"filename = {src_fname}\n"
            f"volume = {src_dims}\n"
            f"voxelsize = {src_vsize}\n"
            f"xras   = {src_v2r[0, :]}\n"
            f"yras   = {src_v2r[1, :]}\n"
            f"zras   = {src_v2r[2, :]}\n"
            f"cras   = {src_c}\n"
            "dst volume info\n"
            "valid = 1  # volume info valid\n"
            f"filename = {dst_fname}\n"
            f"volume = {dst_dims}\n"
            f"voxelsize = {dst_vsize}\n"
            f"xras   = {dst_v2r[0, :]}\n"
            f"yras   = {dst_v2r[1, :]}\n"
            f"zras   = {dst_v2r[2, :]}\n"
            f"cras   = {dst_c}\n").replace("[", "").replace("]", "")
        )


def read_lta(file: Path | str) -> LTADict:
    """Read the LTA info."""
    import re
    from functools import partial

    import numpy as np
    parameter_pattern = re.compile("^\\s*([^=]+)\\s*=\\s*([^#]*)\\s*(#.*)")
    vol_info_pattern = re.compile("^(.*) volume info$")
    shape_pattern = re.compile("^(\\s*\\d+)+$")
    matrix_pattern = re.compile("^(-?\\d+\\.\\S+\\s+)+$")

    def _vector(_a: str, dtype: npt.DTypeLike = float, count: int = -1) -> npt.DTypeLike:
        return np.fromstring(_a, dtype=dtype, count=count, sep=" ").tolist()

    parameters = {
        "type": int,
        "nxforms": int,
        "mean": partial(_vector, dtype=float, count=3),
        "sigma": float,
        "subject": str,
        "fscale": float,
    }
    vol_info_par = {
        "valid": int,
        "filename": str,
        "volume": partial(_vector, dtype=int, count=3),
        "voxelsize": partial(_vector, dtype=float, count=3),
        **{f"{c}ras": partial(_vector, dtype=float) for c in "xyzc"}
    }

    with open(file) as f:
        lines = f.readlines()

    items = []
    shape_lines = []
    matrix_lines = []
    section = ""
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        if hits := parameter_pattern.match(line):
            name = hits.group(1)
            if section and name in vol_info_par:
                items.append((f"{section}_{name}", vol_info_par[name](hits.group(2))))
            elif name in parameters:
                section = ""
                items.append((name, parameters[name](hits.group(2))))
            else:
                raise NotImplementedError(f"Unrecognized type string in lta-file "
                                          f"{file}:{i+1}: '{name}'")
        elif hits := vol_info_pattern.match(line):
            section = hits.group(1)
            # not a parameter line
        elif shape_pattern.search(line):
            shape_lines.append(np.fromstring(line, dtype=int, count=-1, sep=" "))
        elif matrix_pattern.search(line):
            matrix_lines.append(np.fromstring(line, dtype=float, count=-1, sep=" "))

    shape_lines = list(map(tuple, shape_lines))
    lta = dict(items)
    if lta["nxforms"] != len(shape_lines):
        raise OSError("Inconsistent lta format: nxforms inconsistent with shapes.")
    if len(shape_lines) > 1 and np.any(np.not_equal([shape_lines[0]], shape_lines[1:])):
        raise OSError(f"Inconsistent lta format: shapes inconsistent {shape_lines}")
    lta_matrix = np.asarray(matrix_lines).reshape((-1,) + shape_lines[0])
    lta["lta"] = lta_matrix
    return lta
