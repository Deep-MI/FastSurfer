# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

"""
Gray and white matter intensity modes for surface placement, without rounding to integers.

mris_autodet_gwstats samples the intensity 1 mm inside and outside the initial white surface and
takes the highest bin of an integer histogram as the white and gray mode. When two bins are nearly
equal, a small change of the surface moves a mode by a whole unit, which changes cortical thickness
by about 1 %. Here the samples are the same, but the mode is the maximum of a histogram smoothed
with a Gaussian kernel. The standard deviations, which come from wm.mgz alone, are kept, and all
values that follow from the modes are recomputed with the formulas of AutoDetGWStats.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

# MRIclipBrightWM: voxels labelled white matter (WM_MIN_VAL) are clipped to DEFAULT_DESIRED_WHITE_MATTER_VALUE
WM_MIN_VAL = 5
WHITE_CLIP = 110
# width of the Gaussian kernel in intensity units
BANDWIDTH = 2.0
# default of AutoDetGWStats, not written to the stats file
MIN_GRAY_AT_WHITE_BORDER = 70.0

# keys of the stats file, in the order AutoDetGWStats::ReadStream reads them
KEYS = (
    "hemicode", "white_border_hi", "white_border_low", "white_outside_low", "white_inside_hi", "white_outside_hi",
    "pial_border_hi", "pial_border_low", "pial_outside_low", "pial_inside_hi", "pial_outside_hi", "use_mode",
    "variablesigma", "std_scale", "adWHITE_MATTER_MEAN", "MAX_WHITE", "MIN_BORDER_WHITE", "MAX_BORDER_WHITE",
    "MAX_GRAY", "MID_GRAY", "MIN_GRAY_AT_CSF_BORDER", "MAX_GRAY_AT_CSF_BORDER", "MIN_CSF", "adMAX_CSF",
    "white_mean", "white_mode", "white_std", "gray_mean", "gray_mode", "gray_std", "min_border_white",
    "max_border_white", "min_gray_at_white_border", "max_gray", "min_gray_at_csf_border", "max_gray_at_csf_border",
    "min_csf", "max_csf", "max_gray_scale", "max_scale_down",
)
INTEGER_KEYS = ("hemicode", "use_mode")


def vertex_normals(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Normalised sum of the unit normals of the triangles at each vertex, as FreeSurfer computes them."""
    n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, t[:, k], n)
    return vn / np.linalg.norm(vn, axis=1, keepdims=True)


def sample_white_gray(volume: Path | str, wm: Path | str, surface: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """
    Intensities 1 mm inside and 1 mm outside each vertex of the surface, as MRIScomputeClassModes samples them.

    Returns
    -------
    white, gray : np.ndarray
        Trilinearly interpolated intensity of the volume, with bright white matter clipped as in
        MRIclipBrightWM, inside and outside each vertex.
    """
    img = nib.load(str(volume))
    data = np.asarray(img.dataobj, dtype=np.float64)
    data[(np.asarray(nib.load(str(wm)).dataobj) >= WM_MIN_VAL) & (data > WHITE_CLIP)] = WHITE_CLIP
    v, t = nib.freesurfer.read_geometry(str(surface))[:2]
    v = v.astype(np.float64)
    n = vertex_normals(v, t)
    ras2vox = np.linalg.inv(img.header.get_vox2ras_tkr())
    white, gray = (
        map_coordinates(data, ((v + sign * n) @ ras2vox[:3, :3].T + ras2vox[:3, 3]).T, order=1, mode="constant")
        for sign in (-1.0, 1.0)
    )
    return white, gray


def smooth_mode(x: np.ndarray, bandwidth: float = BANDWIDTH, step: float = 0.01) -> float:
    """
    Maximum of the histogram of ``x`` smoothed with a Gaussian kernel of width ``bandwidth``.

    Samples of intensity 0, which FreeSurfer drops by clearing the zero bin, are left out.
    """
    x = x[x > 0.5]
    lo, hi = np.percentile(x, [1, 99])
    margin = 5 * bandwidth
    counts, edges = np.histogram(x, bins=np.arange(lo - margin, hi + margin + step, step))
    centres = 0.5 * (edges[:-1] + edges[1:])
    offsets = np.arange(-margin, margin + step / 2, step)
    density = np.convolve(counts, np.exp(-0.5 * (offsets / bandwidth) ** 2), mode="same")
    inside = (centres >= lo) & (centres <= hi)
    return float(centres[inside][np.argmax(density[inside])])


def read_stats(path: Path | str) -> dict:
    """Read a stats file written by mris_autodet_gwstats."""
    values = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            values[parts[0]] = float(parts[1])
    missing = [k for k in KEYS if k not in values]
    if missing:
        raise ValueError(f"{path} lacks {', '.join(missing)}")
    return values


def write_stats(stats: dict, path: Path | str) -> None:
    """Write a stats file that mris_place_surface --adgws-in reads."""
    lines = [f"{k} {int(stats[k])}" if k in INTEGER_KEYS else f"{k} {stats[k]:f}" for k in KEYS]
    Path(path).write_text("\n".join(lines) + "\n")


def derived_values(stats: dict, white_mode: float, gray_mode: float) -> dict:
    """
    All values of the stats file that follow from the modes, as AutoDetGWStats::AutoDetectStats computes them.

    The standard deviations and the constants are taken from ``stats``.
    """
    s = stats
    scale = s["max_scale_down"]
    white_std, gray_std = s["white_std"], s["gray_std"]

    def floor(value, constant):
        return float(np.rint(scale * constant)) if value < scale * constant else value

    out = {"white_mode": white_mode, "gray_mode": gray_mode, "white_mean": white_mode, "gray_mean": gray_mode}
    out["min_gray_at_white_border"] = floor(gray_mode - gray_std, MIN_GRAY_AT_WHITE_BORDER)
    out["max_border_white"] = floor(white_mode + white_std, s["MAX_BORDER_WHITE"])
    out["min_border_white"] = floor(gray_mode, s["MIN_BORDER_WHITE"])
    out["max_csf"] = max(gray_mode - max(0.5, s["variablesigma"] - 1.0) * gray_std, scale * s["adMAX_CSF"])
    out["max_gray"] = floor(white_mode - white_std, s["MAX_GRAY"])
    out["max_gray_at_csf_border"] = floor(gray_mode - gray_std, s["MAX_GRAY_AT_CSF_BORDER"])
    out["min_gray_at_csf_border"] = floor(gray_mode - s["variablesigma"] * gray_std, s["MIN_GRAY_AT_CSF_BORDER"])
    out["MID_GRAY"] = (out["max_gray"] + out["min_gray_at_csf_border"]) / 2.0
    out["white_border_hi"] = out["max_border_white"]
    out["white_border_low"] = out["min_border_white"]
    out["white_outside_low"] = out["min_gray_at_white_border"]
    weight = s["max_gray_scale"]
    out["white_outside_hi"] = (out["max_border_white"] + weight * out["max_gray"]) / (weight + 1)
    out["pial_inside_hi"] = out["max_gray"]
    out["pial_border_hi"] = out["max_gray_at_csf_border"]
    out["pial_border_low"] = out["min_gray_at_csf_border"]
    out["pial_outside_hi"] = (out["max_csf"] + out["max_gray_at_csf_border"]) / 2.0
    return out


def refine_stats(stats: dict, white: np.ndarray, gray: np.ndarray) -> dict:
    """
    Replace the integer modes of a stats file by smooth modes of the same samples.

    The gray mode is searched only below the white mode: gray samples that fall on white matter
    clipped to 110 would otherwise form a peak of their own, which can win over the gray peak
    (FreeSurfer issue 1494).

    Raises
    ------
    ValueError
        If the file is not what AutoDetGWStats computes from its own modes, for example because a
        value was set on the command line; such a file is left to the caller.
    """
    if not stats["use_mode"]:
        raise ValueError("the stats file uses means, not modes")
    expected = derived_values(stats, stats["white_mode"], stats["gray_mode"])
    differ = [k for k, value in expected.items() if abs(value - stats[k]) > 1e-4]
    if differ:
        raise ValueError(f"the stats file was not computed from its modes alone: {', '.join(differ)}")
    white_mode = smooth_mode(white)
    gray_mode = smooth_mode(gray[gray < white_mode])
    return {**stats, **derived_values(stats, white_mode, gray_mode)}
