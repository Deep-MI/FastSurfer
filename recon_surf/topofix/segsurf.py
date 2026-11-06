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
White matter surfaces from a FastSurfer segmentation.

The segmentation is reduced to white matter per hemisphere, where white matter also includes the
subcortical structures it encloses, and the surface is the level set of the slightly blurred
binary mask, found with marching cubes.
"""

import numpy as np
from lapy import TriaMesh
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage import measure

# labels that vote for a hemisphere when assigning white matter hypointensities
LEFT_LABELS = (2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 30, 31)
RIGHT_LABELS = (41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63)
# white matter, lateral ventricle, thalamus, caudate, putamen, pallidum, accumbens, ventral DC,
# vessel, choroid plexus
LEFT_WM = (2, 4, 10, 11, 12, 13, 26, 28, 30, 31)
RIGHT_WM = (41, 43, 49, 50, 51, 52, 58, 60, 62, 63)
WM_HYPOINTENSITIES = 77


def white_matter_mask(seg: np.ndarray, hemi: str) -> np.ndarray:
    """
    White matter of one hemisphere, including the subcortical structures it encloses.

    Hippocampus, amygdala, inferior lateral ventricle, cerebellum, brainstem and the third and
    fourth ventricle are not included. White matter hypointensities belong to the hemisphere
    whose labels are more frequent in a 7 voxel neighbourhood.

    Parameters
    ----------
    seg : np.ndarray
        Integer segmentation volume, for example ``aparc.DKTatlas+aseg.deep.mgz``.
    hemi : {"lh", "rh"}
        Hemisphere.

    Returns
    -------
    np.ndarray
        Boolean volume.
    """
    seg = np.asarray(seg)
    own, other = (LEFT_LABELS, RIGHT_LABELS) if hemi == "lh" else (RIGHT_LABELS, LEFT_LABELS)
    mask = np.isin(seg, LEFT_WM if hemi == "lh" else RIGHT_WM)
    hypo = seg == WM_HYPOINTENSITIES
    if hypo.any():
        vote_own = uniform_filter(np.isin(seg, own).astype(np.float32), 7)
        vote_other = uniform_filter(np.isin(seg, other).astype(np.float32), 7)
        mask |= hypo & (vote_own > vote_other)
    return mask


def extract_surface(
        mask: np.ndarray,
        vox2ras: np.ndarray,
        level: float = 0.45,
        sigma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the closed surface around a binary mask.

    The mask is blurred with a Gaussian of ``sigma`` voxels and its level set at ``level`` is
    found with marching cubes. Only the largest connected component is kept. Blurring makes
    voxels that touch only at an edge or a corner either join or separate cleanly, so the result
    is a manifold. A level slightly below one half moves a flat boundary outward by much less
    than a voxel.

    Parameters
    ----------
    mask : np.ndarray
        Boolean volume.
    vox2ras : np.ndarray
        4 x 4 matrix from voxel indices to surface coordinates, for example the tkr RAS matrix.
    level : float, default=0.45
        Level of the blurred mask.
    sigma : float, default=0.5
        Gaussian width in voxels; 0 for no blurring.

    Returns
    -------
    v : np.ndarray
        Vertex array of shape (n_vertices, 3) in surface coordinates.
    t : np.ndarray
        Triangle array of shape (n_triangles, 3), oriented outward.
    """
    volume = np.asarray(mask, dtype=np.float32)
    if sigma > 0:
        volume = gaussian_filter(volume, sigma, mode="nearest")
    v, t, _, _ = measure.marching_cubes(volume, level=level, gradient_direction="descent")
    mesh = TriaMesh(v, t)
    mesh.keep_largest_connected_component_(clean=True)
    v = mesh.v @ np.asarray(vox2ras)[:3, :3].T + np.asarray(vox2ras)[:3, 3]
    t = mesh.t
    if TriaMesh(v, t).volume() < 0:
        t = t[:, ::-1]
    return v, np.ascontiguousarray(t, dtype=np.int64)
