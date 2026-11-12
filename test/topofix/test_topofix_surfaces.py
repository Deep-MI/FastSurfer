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
The step recon-surf.sh runs with --topofix, called as recon-surf.sh calls it.

recon-surf.sh only looks at the exit status: 0 means that orig.nofix, orig.premesh and orig are
written and the latter two are genus 0, anything else starts the standard surface steps instead.
"""

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from lapy import TriaMesh
from lapy.intersect import self_intersections
from scipy.spatial import cKDTree
from shapes import voxel_surface

from recon_surf.topofix.surfaces import average_vertex_positions, remove_intersections

FASTSURFER_HOME = Path(__file__).parent.parent.parent
SCRIPT = FASTSURFER_HOME / "recon_surf" / "topofix_surfaces.py"
# LIA orientation, as conformed volumes
CONFORMED = np.array([[-1.0, 0, 0, 12], [0, 0, 1, -12], [0, -1, 0, 12], [0, 0, 0, 1]])


def run(subjects_dir):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--hemi", "lh", "--sd", str(subjects_dir), "--subject", "subj"],
        capture_output=True,
        text=True,
    )


def test_tunnel_is_removed_and_the_rest_of_the_surface_kept(tmp_path):
    mask = np.zeros((24, 24, 24), dtype=bool)
    mask[6:18, 6:18, 6:18] = True
    mask[11:13, 11:13, 6:18] = False  # a tunnel through the block, along the third voxel axis
    for folder in ("mri", "surf", "scripts"):
        (tmp_path / "subj" / folder).mkdir(parents=True)
    img = nib.MGHImage(np.where(mask, 255, 0).astype(np.uint8), CONFORMED)
    img.to_filename(tmp_path / "subj" / "mri" / "filled-pretess255.mgz")

    done = run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr

    record = json.loads((tmp_path / "subj" / "scripts" / "lh.topofix.json").read_text())
    assert record["status"] == "ok"
    assert record["genus"] == 1 and len(record["handles"]) == 1 and record["genus_corrected"] == 0
    surf = tmp_path / "subj" / "surf"
    nofix = TriaMesh.read_fssurf(str(surf / "lh.orig.nofix"))
    assert nofix.is_closed() and nofix.is_manifold() and nofix.genus() == 1
    # halfway between voxel centres: the block minus the tunnel, with its edges cut off
    assert 0.95 * (12**3 - 48) < nofix.volume() < 12**3 - 48
    premesh = TriaMesh.read_fssurf(str(surf / "lh.orig.premesh"))
    assert premesh.is_closed() and premesh.is_manifold() and premesh.is_oriented()
    assert premesh.genus() == 0 and premesh.volume() > 0
    assert len(self_intersections(premesh)) == 0
    # away from the tunnel, every vertex keeps its position after the averaging mris_fix_topology does
    vox2ras = img.header.get_vox2ras_tkr()
    axis = vox2ras[:3, 2] / np.linalg.norm(vox2ras[:3, 2])
    offset = nofix.v - (vox2ras @ [11.5, 11.5, 12, 1])[:3]
    far = np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1) > 4
    averaged = average_vertex_positions(nofix.v, nofix.t)
    assert np.all(cKDTree(premesh.v).query(averaged[far])[0] < 1e-4)
    orig = TriaMesh.read_fssurf(str(surf / "lh.orig"))
    assert orig.is_closed() and orig.is_manifold() and orig.is_oriented() and orig.genus() == 0
    assert len(self_intersections(orig)) == 0
    assert abs(orig.volume() / premesh.volume() - 1) < 0.02
    # the volume geometry of filled-pretess, which later FreeSurfer steps read
    for surface in (nofix, premesh, orig):
        assert surface.fsinfo is not None and tuple(surface.fsinfo["volume"]) == (24, 24, 24)


def test_intersection_is_smoothed_away_locally():
    mask = np.zeros((8, 8, 3), dtype=bool)
    mask[1:7, 1:7, 1] = True  # a slab one voxel thick
    v, t = voxel_surface(mask)
    top = np.flatnonzero((v[:, 2] == v[:, 2].max()) & (np.abs(v[:, 0] - 4.5) < 0.1) & (np.abs(v[:, 1] - 4.5) < 0.1))
    v[top, 2] -= 1.5  # pushed through the bottom face
    assert len(self_intersections(TriaMesh(v, t))) > 0
    fixed, n_moved = remove_intersections(v, t)
    assert len(self_intersections(TriaMesh(fixed, t))) == 0
    assert 0 < n_moved < len(v) / 2
    far = np.max(np.abs(v[:, :2] - 4.5), axis=1) > 2.5
    assert np.array_equal(fixed[far], v[far])


def test_a_failure_exits_with_1_and_records_the_error(tmp_path):
    (tmp_path / "subj" / "scripts").mkdir(parents=True)
    done = run(tmp_path)
    assert done.returncode == 1
    record = json.loads((tmp_path / "subj" / "scripts" / "lh.topofix.json").read_text())
    assert record["status"] == "failed" and "filled-pretess255.mgz" in record["error"]
