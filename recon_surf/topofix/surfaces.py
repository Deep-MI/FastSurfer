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

"""Create ``?h.orig.nofix``, ``?h.orig.premesh`` and ``?h.orig`` of one hemisphere from the segmentation."""

import time
from pathlib import Path

import nibabel as nib
import numpy as np
from lapy import TriaMesh
from lapy.intersect import undo_intersections
from lapy.remesh import remesh
from lapy.topology import fix_topology, shorter_loop

from recon_surf.topofix.classify import cuts_bridge
from recon_surf.topofix.folds import crop_folds
from recon_surf.topofix.joint import FlipChooser, choose_jointly
from recon_surf.topofix.segsurf import extract_surface, white_matter_mask

# Target edge lengths of ?h.orig in mm. They give about the vertex density of ?h.orig on the
# standard route, where -hires surfaces are coarser because they are decimated before the fix.
TARGET_LENGTH = 0.82
TARGET_LENGTH_HIRES = 0.93


def read_segmentation(path: Path | str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Read a label volume with its voxel to surface matrix and the volume geometry for surfaces.

    Returns
    -------
    seg : np.ndarray
        Integer label volume.
    vox2ras : np.ndarray
        4 x 4 matrix from voxel indices to surface (tkr RAS) coordinates.
    info : dict
        Volume geometry header for surfaces written with :meth:`lapy.TriaMesh.write_fssurf`.
    """
    img = nib.load(str(path))
    hdr = img.header
    mdc = np.asarray(hdr["Mdc"], dtype=float)
    info = {
        "head": np.array([2, 0, 20], dtype=np.int32),
        "valid": "1  # volume info valid",
        "filename": str(path),
        "volume": np.asarray(hdr["dims"][:3], dtype=np.int64),
        "voxelsize": np.asarray(hdr["delta"], dtype=float),
        "xras": mdc[0],
        "yras": mdc[1],
        "zras": mdc[2],
        "cras": np.asarray(hdr["Pxyz_c"], dtype=float),
    }
    return np.asarray(img.dataobj).astype(np.int64), hdr.get_vox2ras_tkr(), info


def create_orig_surfaces(
        seg_path: Path | str,
        hemi: str,
        surf_dir: Path | str,
        target_length: float = TARGET_LENGTH,
        smooth_iterations: int = 2,
) -> dict:
    """
    Create the uncorrected and the corrected white matter surface of one hemisphere.

    Writes ``<hemi>.orig.nofix``, the surface of the white matter in the segmentation;
    ``<hemi>.orig.premesh``, the same surface with all handles removed, the protrusions on which
    a spherical map folds cut off, and Taubin smoothing; and ``<hemi>.orig``, that surface
    remeshed to ``target_length``.

    Parameters
    ----------
    seg_path : Path, str
        Segmentation, for example ``aparc.DKTatlas+aseg.deep.mgz``.
    hemi : {"lh", "rh"}
        Hemisphere.
    surf_dir : Path, str
        Output folder, the subject's ``surf`` folder.
    target_length : float, default=TARGET_LENGTH
        Edge length of ``<hemi>.orig`` in mm.
    smooth_iterations : int, default=2
        Taubin smoothing iterations; vertices whose smoothing would make the surface intersect
        itself keep their position.

    Returns
    -------
    dict
        What was done, for the QC record: the genus, every removed handle, the cropping and the
        size of each surface, and the time each step took.
    """
    surf_dir = Path(surf_dir)
    qc = {"hemi": hemi, "segmentation": str(seg_path), "seconds": {}}

    def write(name, v, t):
        TriaMesh(v, t, info).write_fssurf(str(surf_dir / f"{hemi}.{name}"))
        qc[name] = {"vertices": len(v), "triangles": len(t)}

    start = time.perf_counter()
    seg, vox2ras, info = read_segmentation(seg_path)
    v, t = extract_surface(white_matter_mask(seg, hemi), vox2ras)
    write("orig.nofix", v, t)
    qc["genus"] = TriaMesh(v, t).genus()
    qc["seconds"]["surface"] = time.perf_counter() - start

    start = time.perf_counter()
    centres, flips, _ = choose_jointly(v, t, classify=cuts_bridge)
    fixed, origin, surgeries = fix_topology(
        TriaMesh(v, t), choose=FlipChooser(shorter_loop, centres, flips), classify=cuts_bridge,
    )
    qc["handles"] = [
        {
            "operation": s.operation,
            "length": float(s.length),
            "other_length": float(s.other_length),
            "centre": [round(float(c), 2) for c in v[s.loop].mean(axis=0)],
            "intersections": int(s.n_intersections),
        }
        for s in surgeries
    ]
    qc["flipped_by_joint_choice"] = int(flips.sum())
    qc["seconds"]["fix"] = time.perf_counter() - start

    start = time.perf_counter()
    v2, t2, kept, history = crop_folds(fixed.v, fixed.t)
    origin = np.where(kept >= 0, origin[np.maximum(kept, 0)], -1)
    qc["folded_triangles_per_round"] = history
    qc["input_vertices_removed"] = len(v) - len(np.unique(origin[origin >= 0]))
    qc["seconds"]["crop"] = time.perf_counter() - start

    start = time.perf_counter()
    surface = TriaMesh(v2, t2)
    v2, n_back = undo_intersections(surface, surface.smooth_taubin(n=smooth_iterations))
    qc["vertices_kept_by_smoothing"] = int(n_back)
    write("orig.premesh", v2, t2)
    qc["genus_corrected"] = TriaMesh(v2, t2).genus()
    orig = remesh(TriaMesh(v2, t2), target_length=target_length)
    write("orig", orig.v, orig.t)
    qc["target_length"] = target_length
    qc["seconds"]["smooth_remesh"] = time.perf_counter() - start
    return qc
