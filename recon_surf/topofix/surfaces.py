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

"""Create ``?h.orig.premesh`` and ``?h.orig`` of one hemisphere by correcting the topology of ``?h.orig.nofix``."""

import time
from pathlib import Path

import numpy as np
from lapy import TriaMesh
from lapy.intersect import self_intersections
from lapy.remesh import remesh
from lapy.topology import fix_topology

from recon_surf.topofix.classify import cuts_bridge
from recon_surf.topofix.folds import crop_folds


def average_vertex_positions(v: np.ndarray, t: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Replace each vertex by the mean of itself and its neighbours, as MRISaverageVertexPositions.

    mris_fix_topology smooths the whole surface this way, twice, and writes the smoothed positions.
    """
    adj = TriaMesh(v, t).adj_sym.tocsr()
    adj.data[:] = 1.0
    count = np.asarray(adj.sum(axis=1)).ravel() + 1.0
    v = np.asarray(v, dtype=float)
    for _ in range(iterations):
        v = (v + adj @ v) / count[:, None]
    return v


def remove_intersections(v: np.ndarray, t: np.ndarray, max_rounds: int = 20) -> tuple[np.ndarray, int]:
    """
    Smooth the surface around self-intersections until there are none, as mris_remove_intersection.

    Each round averages the vertices of the intersecting triangles and their neighbours with their
    neighbours, see :func:`average_vertex_positions`, and leaves all other vertices in place.

    Returns
    -------
    v : np.ndarray
        Vertex positions without self-intersections.
    n_moved : int
        Number of vertices that moved.

    Raises
    ------
    RuntimeError
        If intersections remain after ``max_rounds`` rounds.
    """
    v = np.asarray(v, dtype=float).copy()
    adj = TriaMesh(v, t).adj_sym.tocsr()
    moved = np.zeros(len(v), dtype=bool)
    for _ in range(max_rounds):
        pairs = self_intersections(TriaMesh(v, t))
        if len(pairs) == 0:
            return v, int(moved.sum())
        near = np.zeros(len(v), dtype=bool)
        near[t[np.unique(np.asarray(pairs).ravel())].ravel()] = True
        near |= adj @ near.astype(float) > 0
        v[near] = average_vertex_positions(v, t, iterations=1)[near]
        moved |= near
    raise RuntimeError(f"{len(pairs)} self-intersections remain after {max_rounds} rounds.")


def correct_orig_surface(surf_dir: Path | str, hemi: str) -> dict:
    """
    Correct the topology of ``<hemi>.orig.nofix``, and write ``<hemi>.orig.premesh`` and ``<hemi>.orig``.

    As with ``mris_fix_topology``, the surface is first smoothed by averaging every vertex with its
    neighbours twice, see :func:`average_vertex_positions`, and away from the handles the result
    keeps these positions. Each handle is removed by surgery along the shorter of its two loops,
    and the protrusions on which a spherical map folds are cut off. As in ``recon-all -fix``, the
    result is then remeshed and freed of self-intersections, see :func:`remove_intersections`.

    Parameters
    ----------
    surf_dir : Path, str
        The subject's ``surf`` folder.
    hemi : {"lh", "rh"}
        Hemisphere.

    Returns
    -------
    dict
        What was done, for the QC record: the genus, every removed handle, the cropping and the
        size of each surface, and the time each step took.
    """
    surf_dir = Path(surf_dir)
    nofix = TriaMesh.read_fssurf(str(surf_dir / f"{hemi}.orig.nofix"))
    qc = {"hemi": hemi, "genus": nofix.genus(), "seconds": {}}
    qc["orig.nofix"] = {"vertices": len(nofix.v), "triangles": len(nofix.t)}

    start = time.perf_counter()
    nofix = TriaMesh(average_vertex_positions(nofix.v, nofix.t), nofix.t, nofix.fsinfo)
    fixed, origin, surgeries = fix_topology(nofix, classify=cuts_bridge)
    qc["handles"] = [
        {
            "operation": s.operation,
            "length": float(s.length),
            "other_length": float(s.other_length),
            "centre": [round(float(c), 2) for c in nofix.v[s.loop].mean(axis=0)],
            "intersections": int(s.n_intersections),
        }
        for s in surgeries
    ]
    qc["seconds"]["fix"] = time.perf_counter() - start

    start = time.perf_counter()
    v, t, kept, history = crop_folds(fixed.v, fixed.t)
    origin = np.where(kept >= 0, origin[np.maximum(kept, 0)], -1)
    removed = np.ones(len(nofix.v), dtype=bool)
    removed[origin[origin >= 0]] = False
    qc["folded_triangles_per_round"] = history
    qc["removed_area_mm2"] = float(nofix.vertex_areas()[removed].sum())
    qc["seconds"]["crop"] = time.perf_counter() - start

    premesh = TriaMesh(v, t, nofix.fsinfo)
    premesh.write_fssurf(str(surf_dir / f"{hemi}.orig.premesh"))
    qc["orig.premesh"] = {"vertices": len(v), "triangles": len(t)}
    qc["genus_corrected"] = premesh.genus()

    # the density of mris_remesh --remesh --iters 3, whose edges end up at 0.95 times the mean
    # edge length of its input
    start = time.perf_counter()
    sides = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    sides = np.unique(np.sort(sides, axis=1), axis=0)
    mean_edge = float(np.linalg.norm(v[sides[:, 0]] - v[sides[:, 1]], axis=1).mean())
    orig = remesh(premesh, target_length=0.95 * mean_edge, iterations=3)
    v_orig, n_moved = remove_intersections(orig.v, orig.t)
    TriaMesh(v_orig, orig.t, nofix.fsinfo).write_fssurf(str(surf_dir / f"{hemi}.orig"))
    qc["orig"] = {"vertices": len(v_orig), "triangles": len(orig.t)}
    qc["vertices_moved_by_intersection_removal"] = n_moved
    qc["seconds"]["remesh"] = time.perf_counter() - start
    return qc
