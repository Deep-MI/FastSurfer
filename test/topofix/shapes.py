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

"""Small test surfaces and segmentations: a torus, and voxel volumes with a bridge or a protrusion."""

import numpy as np

CORTEX_LH = 1003

# corner offsets of a face square in the two in-plane axes, counter-clockwise around the axis
_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


def segmentation_block(shape=(24, 24, 24), lo=6, hi=18, label=2):
    """Left white matter block inside a cortex shell, inside background."""
    seg = np.zeros(shape, dtype=np.int64)
    seg[lo - 2 : hi + 2, lo - 2 : hi + 2, lo - 2 : hi + 2] = CORTEX_LH
    seg[lo:hi, lo:hi, lo:hi] = label
    return seg


def torus(n=40, m=16, big_r=3.0, small_r=1.0):
    """Quad-grid torus with n segments around the axis and m around the tube."""
    i, j = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    u = 2 * np.pi * i / n
    w = 2 * np.pi * j / m
    v = np.column_stack(
        [
            ((big_r + small_r * np.cos(w)) * np.cos(u)).ravel(),
            ((big_r + small_r * np.cos(w)) * np.sin(u)).ravel(),
            (small_r * np.sin(w)).ravel(),
        ]
    )
    a = (i * m + j).ravel()
    b = (((i + 1) % n) * m + j).ravel()
    c = (((i + 1) % n) * m + (j + 1) % m).ravel()
    d = (i * m + (j + 1) % m).ravel()
    t = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    return v, t


def voxel_surface(mask):
    """
    Tessellate the boundary faces of a binary volume, as mri_tessellate does.

    Every face between a foreground and a background voxel becomes two triangles, with vertices on
    the voxel corners and normals pointing out of the foreground. Voxel centres sit at integer
    coordinates.
    """
    m = np.pad(np.asarray(mask, dtype=np.int8), 1)
    corner_shape = np.array(m.shape) + 1
    quads = []
    for axis in range(3):
        b, c = (axis + 1) % 3, (axis + 2) % 3
        step = np.diff(m, axis=axis)
        for sign in (1, -1):
            # sign -1: foreground before the face, so the normal points along +axis
            idx = np.argwhere(step == sign)
            base = idx.copy()
            base[:, axis] += 1
            corners = np.repeat(base[:, None, :], 4, axis=1)
            corners[:, :, b] += _SQUARE[:, 0]
            corners[:, :, c] += _SQUARE[:, 1]
            if sign == 1:
                corners = corners[:, ::-1]
            quads.append(np.ravel_multi_index(corners.reshape(-1, 3).T, corner_shape).reshape(-1, 4))
    quads = np.concatenate(quads)
    used, inv = np.unique(quads, return_inverse=True)
    quads = inv.reshape(-1, 4)
    t = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    v = np.column_stack(np.unravel_index(used, corner_shape)) - 1.5
    return v.astype(float), t.astype(np.int64)


def mushroom(stalk=2, cap=10):
    """Voxel block with a wide cap on a thin stalk on top, a genus-0 surface with a protrusion."""
    mask = np.zeros((30, 30, 22), dtype=bool)
    mask[2:28, 2:28, 2:10] = True
    c = 15 - stalk // 2
    mask[c : c + stalk, c : c + stalk, 10:14] = True
    o = 15 - cap // 2
    mask[o : o + cap, o : o + cap, 14:17] = True
    return mask


def block_with_bridge():
    """Solid voxel block with a one-voxel-thick arch on top, a genus-1 surface."""
    mask = np.zeros((10, 6, 9), dtype=bool)
    mask[:, :, :5] = True
    mask[2, 2, 5:8] = True
    mask[7, 2, 5:8] = True
    mask[2:8, 2, 7] = True
    return mask
