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
Tell whether surgery along a loop cuts a bridge or fills a tunnel.

Surgery along a loop that bounds a disk inside the solid cuts through a bridge; surgery along a
loop that bounds a disk outside the solid fills a tunnel. A loop around a bridge grows when it is
moved outward along the surface normals, while a loop around a tunnel shrinks (Weinrauch et al.
2023).
"""

import numpy as np


def loop_length(v: np.ndarray, loop: np.ndarray) -> float:
    """Length of a closed vertex loop."""
    p = v[loop]
    return float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())


def fan_area(v: np.ndarray, loop: np.ndarray) -> float:
    """Area of the triangle fan from the loop centroid to the loop."""
    p = v[loop]
    c = p.mean(axis=0)
    return 0.5 * float(np.linalg.norm(np.cross(p - c, np.roll(p, -1, axis=0) - c), axis=1).sum())


def local_normals(v: np.ndarray, t: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Area-weighted vertex normals, computed only at the given vertices.

    Returns an array of shape (n_vertices, 3) that is zero away from ``vertices``.
    """
    inside = np.zeros(len(v), dtype=bool)
    inside[vertices] = True
    tt = t[np.any(inside[t], axis=1)]
    tn = np.cross(v[tt[:, 1]] - v[tt[:, 0]], v[tt[:, 2]] - v[tt[:, 0]])
    normals = np.zeros((len(v), 3))
    for k in range(3):
        np.add.at(normals, tt[:, k], tn)
    length = np.linalg.norm(normals, axis=1)
    normals[length > 0] /= length[length > 0, None]
    normals[~inside] = 0.0
    return normals


def offset_ratio(v: np.ndarray, t: np.ndarray, loop: np.ndarray, offset: float | None = None) -> float:
    """
    Length of a loop after moving it outward along the normals, divided by its length.

    Parameters
    ----------
    v : np.ndarray
        Vertex array of shape (n_vertices, 3) with outward oriented triangles.
    t : np.ndarray
        Triangle array of shape (n_triangles, 3).
    loop : np.ndarray
        Vertex indices of a closed loop.
    offset : float, optional
        Distance to move the loop. Defaults to a fifth of the mean loop edge length.

    Returns
    -------
    float
        Above one for a loop around a bridge, below one for a loop around a tunnel.
    """
    length = loop_length(v, loop)
    if offset is None:
        offset = 0.2 * length / len(loop)
    return loop_length(v + offset * local_normals(v, t, loop), loop) / length


def cuts_bridge(v: np.ndarray, t: np.ndarray, loop: np.ndarray) -> bool:
    """Whether surgery along a loop cuts a bridge rather than filling a tunnel, a ``classify`` rule."""
    return offset_ratio(v, t, loop) > 1.0
