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
Joint choice of cut or fill for handles that lie close together.

Each handle can be removed along either loop of its pair. Deciding one handle at a time can go
wrong when several bridges hold the same piece: each bridge alone is cheapest to cut, but cutting
all of them detaches the piece, which then becomes a protrusion that cropping removes. Here the
handles of one group are decided together, on a closed patch around them, by trying their
combinations and keeping the one that changes the least surface after cropping.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
from lapy import TriaMesh
from lapy.topology import disk_cap, find_handles, fix_topology, handle_patches, shorter_loop
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from recon_surf.topofix.classify import fan_area
from recon_surf.topofix.folds import close_hole, crop_folds


def _centre(v, a, b):
    return v[np.concatenate([a, b])].mean(axis=0)


class FlipChooser:
    """
    Choose as ``choose`` does, but take the other loop for the flipped handles.

    A loop pair found during surgery belongs to the nearest handle centre not yet used, if that
    is closer than ``tol``; pairs that match no handle keep the choice of ``choose``.
    """

    def __init__(self, choose, centres, flips, tol=3.0):
        self.choose = choose
        self.centres = np.asarray(centres, dtype=float).reshape(-1, 3)
        self.flips = np.asarray(flips, dtype=bool)
        self.tol = tol
        self.used = np.zeros(len(self.centres), dtype=bool)

    def __call__(self, a, b, length_a, length_b, scores=None, v=None):
        """Return 0 or 1 for the loop to use, as a ``choose`` rule of :func:`lapy.topology.fix_topology`."""
        k = self.choose(a, b, length_a, length_b, scores, v=v)
        if v is None or len(self.centres) == 0:
            return k
        d = np.linalg.norm(self.centres - _centre(v, a, b), axis=1)
        d[self.used] = np.inf
        j = int(np.argmin(d))
        if d[j] < self.tol:
            self.used[j] = True
            if self.flips[j]:
                k = 1 - k
        return k


def closed_patch(
        v: np.ndarray,
        t: np.ndarray,
        seeds: np.ndarray,
        radius: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """
    Cut out the surface within geodesic ``radius`` of ``seeds`` and close it with disks.

    Returns
    -------
    vp, tp : np.ndarray
        Closed patch, or None if it is not a closed, connected, oriented manifold.
    origin : np.ndarray
        Input vertex index of each patch vertex, -1 for cap vertices.
    """
    adj = TriaMesh(v, t).adj_sym.tocoo()
    lengths = np.linalg.norm(v[adj.row] - v[adj.col], axis=1)
    graph = sparse.csr_matrix((lengths, (adj.row, adj.col)), shape=adj.shape)
    inside = np.isfinite(csgraph.dijkstra(graph, indices=seeds, min_only=True, limit=radius))
    tp = t[np.all(inside[t], axis=1)]
    # keep the part connected to the seeds
    _, comp = TriaMesh(v, tp).connected_components()
    tp = tp[comp[tp[:, 0]] == comp[seeds[0]]]
    used = np.unique(tp)
    index = np.full(len(v), -1)
    index[used] = np.arange(len(used))
    patch = TriaMesh(v[used], index[tp])
    source = patch.split_pinch_vertices_()
    vp, tp = patch.v, patch.t
    origin = np.concatenate([used, used[source]])
    caps_v, caps_t = [vp], [tp]
    n = len(vp)
    for loop in patch.boundary_loops():
        loop = np.asarray(loop)
        h = np.linalg.norm(np.roll(vp[loop], -1, axis=0) - vp[loop], axis=1).mean()
        inner, tris = disk_cap(vp[loop], h)
        caps_v.append(inner)
        caps_t.append(close_hole(tp, loop, np.concatenate([loop, n + np.arange(len(inner))])[tris]))
        n += len(inner)
    vp, tp = np.vstack(caps_v), np.vstack(caps_t)
    origin = np.concatenate([origin, np.full(len(vp) - len(origin), -1)])
    closed = TriaMesh(vp, tp)
    _, comp = closed.connected_components()
    if not (closed.is_closed() and closed.is_manifold() and closed.is_oriented()) or len(np.unique(comp)) != 1:
        return None, None, origin
    return vp, tp, origin


def surface_change(v, t, real, choose, classify=None, crop=True):
    """
    Area that the repair changes: surgery disks plus surface removed by cropping.

    ``real`` marks the vertices whose removal counts, so that cap vertices of a patch do not.
    Cropping does a single round, without mapping the result to the sphere again.

    Returns
    -------
    cost : float
        Changed area in mm2.
    removed : np.ndarray
        Boolean mask of the real vertices that cropping removed.
    """
    fixed, o1, surgeries = fix_topology(TriaMesh(v, t), choose=choose, classify=classify)
    disks = sum(fan_area(v, s.loop) for s in surgeries)
    removed = np.zeros(len(v), dtype=bool)
    if crop:
        _, _, o2, _ = crop_folds(fixed.v, fixed.t, confirm=False)
        k = o1[o2[o2 >= 0]]
        removed[:] = True
        removed[k[k >= 0]] = False
        removed &= real
    area = TriaMesh(v, t).vertex_areas()
    return disks + float(area[removed].sum()), removed


@dataclass
class GroupChoice:
    """The combinations tried for one group of handles and their surface change in mm2."""

    handles: np.ndarray
    flips: np.ndarray
    tried: list = field(default_factory=list)


def choose_jointly(
        v: np.ndarray,
        t: np.ndarray,
        choose=shorter_loop,
        classify=None,
        radius: float = 20.0,
        group_radius: float = 1.5,
        max_exhaustive: int = 5,
        min_gain: float = 1.0,
        near: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, list[GroupChoice]]:
    """
    Decide for every handle whether to use its default loop or the other one.

    Handles are grouped as in :func:`lapy.topology.handle_patches`. For each group, a closed
    patch of geodesic ``radius`` around its loops is repaired with the default choices. If that
    crops surface within ``near`` of the group's loops, every combination of flipped handles is
    tried as well (single flips only for groups larger than ``max_exhaustive``), and the
    combination with the least :func:`surface_change` is kept if it saves at least ``min_gain``
    mm2 over the default.

    Parameters
    ----------
    v : np.ndarray
        Vertex array of shape (n_vertices, 3).
    t : np.ndarray
        Triangle array of shape (n_triangles, 3) of a closed, connected, oriented manifold mesh.
    choose, classify : callable
        Default rule and classifier, as in :func:`lapy.topology.fix_topology`.
    radius : float, default=20.0
        Geodesic radius of the patch around each group in mm.
    group_radius : float, default=1.5
        Handles whose loops come this close share a group.
    max_exhaustive : int, default=5
        Largest group for which all combinations are tried.
    min_gain : float, default=1.0
        Smallest saving in mm2 for which flips replace the default.
    near : float, default=3.0
        Distance in mm from the group's loops within which cropping starts the search.

    Returns
    -------
    centres : np.ndarray
        Centre of each handle's loop pair, shape (n_handles, 3).
    flips : np.ndarray
        Whether each handle uses the other loop than ``choose`` picks.
    groups : list of GroupChoice
        The search for each group.
    """
    tria = TriaMesh(v, t)
    handles = find_handles(tria)
    _, group = handle_patches(tria, handles, radius=group_radius)
    centres = np.array([_centre(v, h.loop_a, h.loop_b) for h in handles]).reshape(-1, 3)
    flips = np.zeros(len(handles), dtype=bool)
    groups = []
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        seeds = np.unique(np.concatenate([np.r_[handles[i].loop_a, handles[i].loop_b] for i in idx]))
        vp, tp, origin = closed_patch(v, t, seeds, radius)
        choice = GroupChoice(idx + 1, np.zeros(len(idx), dtype=bool))
        groups.append(choice)
        if vp is None:
            continue
        k = len(idx)
        default = (False,) * k
        cost, removed = surface_change(vp, tp, origin >= 0, FlipChooser(choose, centres[idx], default), classify)
        choice.tried.append((default, cost))
        # without cropping next to the group, flips only change disk areas slightly
        if not removed.any() or cKDTree(v[seeds]).query(vp[removed])[0].min() > near:
            continue
        if k <= max_exhaustive:
            combos = list(itertools.product([False, True], repeat=k))[1:]
        else:
            combos = [tuple(np.eye(k, dtype=bool)[i]) for i in range(k)]
        for combo in combos:
            chooser = FlipChooser(choose, centres[idx], combo)
            choice.tried.append((combo, surface_change(vp, tp, origin >= 0, chooser, classify)[0]))
        best, cost = min(choice.tried, key=lambda c: c[1])
        if cost <= choice.tried[0][1] - min_gain:
            choice.flips = np.array(best, dtype=bool)
            flips[idx] = choice.flips
    return centres, flips, groups
