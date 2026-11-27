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

"""Protrusions on which a spherical map of a genus-0 surface folds over, and their removal."""

import numpy as np
from lapy import TriaMesh
from lapy.diffgeo import flipped_triangles, spherical_spectral_map
from lapy.intersect import self_intersections
from lapy.topology import disk_cap
from scipy import sparse


def sphere_folds(v: np.ndarray, t: np.ndarray, flow_iter: int = 3) -> np.ndarray:
    """Faces on which the spectral map of a closed genus-0 surface onto the sphere folds over."""
    return flipped_triangles(spherical_spectral_map(TriaMesh(v, t), flow_iter=flow_iter), t)


def _has_side(t, a, b):
    """Whether some triangle contains the directed side a -> b."""
    return bool(np.any((t == a) & (np.roll(t, -1, axis=1) == b)))


def close_hole(t: np.ndarray, loop: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Orient the triangles of a disk spanning a boundary loop so that they close the hole in ``t``."""
    return cap[:, ::-1] if _has_side(cap, loop[0], loop[1]) == _has_side(t, loop[0], loop[1]) else cap


def _adjacency(n, t):
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    a = sparse.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return ((a + a.T) > 0).astype(float)


def _dilate(adj, mask, rings):
    for _ in range(rings):
        mask = mask | (adj @ mask.astype(float) > 0)
    return mask


def _boundary_length(v, t, mask):
    """Length of the boundary of the faces whose vertices all lie in ``mask``."""
    inside = np.all(mask[t], axis=1)
    sides = np.concatenate([t[inside][:, [0, 1]], t[inside][:, [1, 2]], t[inside][:, [2, 0]]])
    key = np.minimum(sides[:, 0], sides[:, 1]) * len(v) + np.maximum(sides[:, 0], sides[:, 1])
    u, cnt = np.unique(key, return_counts=True)
    b = u[cnt == 1]
    return np.linalg.norm(v[b // len(v)] - v[b % len(v)], axis=1).sum()


def _fill(adj, mask):
    """Add to ``mask`` every part of its complement except the largest one."""
    rest = np.flatnonzero(~mask)
    n, comp = sparse.csgraph.connected_components(adj[rest][:, rest], directed=False)
    if n > 1:
        keep = np.argmax(np.bincount(comp))
        mask = mask.copy()
        mask[rest[comp != keep]] = True
    return mask


def crop_region(
        v: np.ndarray,
        t: np.ndarray,
        mask: np.ndarray,
        edge_length: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove the faces inside a disk-shaped region and close the hole with a smooth disk.

    Parameters
    ----------
    v : np.ndarray
        Vertex array of shape (n_vertices, 3).
    t : np.ndarray
        Triangle array of shape (n_triangles, 3) of a closed, oriented manifold mesh.
    mask : np.ndarray
        Boolean vertex mask; faces with all three vertices in it are removed.
    edge_length : float, optional
        Target edge length of the cap; default is the mean length of the hole boundary edges.

    Returns
    -------
    v_new, t_new : np.ndarray
        The cropped and capped mesh, without the removed vertices.
    keep : np.ndarray
        Input index of each output vertex, -1 for new cap vertices.

    Raises
    ------
    ValueError
        If the hole has more than one boundary loop, or its boundary touches itself.
    """
    rest = t[~np.all(mask[t], axis=1)]
    remaining = TriaMesh(v, rest)
    # a boundary through a vertex twice is not a simple loop
    if len(remaining.pinch_vertices()) > 0:
        raise ValueError("Region boundary touches itself.")
    loops = remaining.boundary_loops()
    if len(loops) != 1:
        raise ValueError(f"Region boundary has {len(loops)} loops, need one.")
    loop = np.asarray(loops[0])
    pts = v[loop]
    h = edge_length or np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).mean()
    inner, tris = disk_cap(pts, h)
    cap = close_hole(rest, loop, np.concatenate([loop, len(v) + np.arange(len(inner))])[tris])
    allv = np.vstack([v, inner])
    allt = np.vstack([rest, cap])
    used = np.zeros(len(allv), dtype=bool)
    used[allt.ravel()] = True
    new_index = np.cumsum(used) - 1
    keep = np.flatnonzero(used)
    keep_orig = np.where(keep < len(v), keep, -1)
    return allv[used], new_index[allt], keep_orig


def crop_folds(
        v: np.ndarray,
        t: np.ndarray,
        rings: int = 2,
        max_grow: int = 6,
        max_rounds: int = 5,
        flow_iter: int = 3,
        confirm: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Cut off the protrusions where the spectral sphere folds and close them with disks.

    Each round maps the surface to the sphere and takes the vertices of the flipped faces plus
    ``rings`` rings of neighbours. Every connected region is filled to a disk, grown by up to
    ``max_grow`` more rings to where its boundary is shortest, which is usually the neck of the
    protrusion, and replaced by a smooth disk spanning that boundary. A cap that would intersect
    the surface is not applied.

    Parameters
    ----------
    v : np.ndarray
        Vertex array of shape (n_vertices, 3).
    t : np.ndarray
        Triangle array of shape (n_triangles, 3) of a closed genus-0 surface.
    rings : int, default=2
        Neighbour rings added around the flipped faces.
    max_grow : int, default=6
        Further rings tried when looking for the shortest boundary.
    max_rounds : int, default=5
        Maximum number of detect-and-crop rounds.
    flow_iter : int, default=3
        Mean curvature flow steps in :func:`sphere_folds`.
    confirm : bool, default=True
        Map the cropped surface to the sphere again, to check it and crop what still folds. Without
        it, a single round is done.

    Returns
    -------
    v_new, t_new : np.ndarray
        The cropped surface.
    origin : np.ndarray
        Input index of each output vertex, -1 for new cap vertices.
    history : list of int
        Number of flipped faces before each round, and after the last one with ``confirm``.
    """
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=np.int64)
    origin = np.arange(len(v))
    history = []
    for _ in range(max_rounds):
        flip = sphere_folds(v, t, flow_iter)
        history.append(int(flip.sum()))
        if not flip.any():
            return v, t, origin, history
        adj = _adjacency(len(v), t)
        seed = np.zeros(len(v), dtype=bool)
        seed[t[flip].ravel()] = True
        seed = _dilate(adj, seed, rings)
        idx = np.flatnonzero(seed)
        n_comp, comp = sparse.csgraph.connected_components(adj[idx][:, idx], directed=False)
        changed = False
        regions = [idx[comp == c] for c in range(n_comp)]
        for r in range(len(regions)):
            mask = np.zeros(len(v), dtype=bool)
            mask[regions[r]] = True
            candidates = []
            for k in range(max_grow + 1):
                grown = _fill(adj, _dilate(adj, mask, k))
                candidates.append((_boundary_length(v, t, grown), k, grown))
            for _, _, grown in sorted(candidates, key=lambda c: c[0]):
                try:
                    nv, nt, keep = crop_region(v, t, grown)
                except ValueError:
                    continue
                new_faces = np.flatnonzero(np.any(keep[nt] < 0, axis=1))
                if len(self_intersections(TriaMesh(nv, nt), subset=new_faces)) == 0:
                    # carry the remaining regions over to the new vertex indices
                    new_index = np.full(len(v), -1)
                    kept = np.flatnonzero(keep >= 0)
                    new_index[keep[kept]] = kept
                    for q in range(r + 1, len(regions)):
                        regions[q] = new_index[regions[q]][new_index[regions[q]] >= 0]
                    origin = np.where(keep >= 0, origin[np.maximum(keep, 0)], -1)
                    v, t, changed = nv, nt, True
                    adj = _adjacency(len(v), t)
                    break
        if not changed or not confirm:
            return v, t, origin, history
    history.append(int(sphere_folds(v, t, flow_iter).sum()))
    return v, t, origin, history
