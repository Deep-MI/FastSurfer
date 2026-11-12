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

recon-surf.sh only looks at the exit status: 0 means that orig.premesh is written and genus 0,
anything else starts the standard surface steps instead.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from lapy import TriaMesh
from lapy.intersect import self_intersections
from scipy.spatial import cKDTree
from shapes import voxel_surface

from recon_surf.topofix.surfaces import average_vertex_positions

FASTSURFER_HOME = Path(__file__).parent.parent.parent
SCRIPT = FASTSURFER_HOME / "recon_surf" / "topofix_surfaces.py"
VOLUME_INFO = {
    "head": np.array([2, 0, 20], dtype=np.int32),
    "valid": "1  # volume info valid",
    "filename": "filled.mgz",
    "volume": np.array([24, 24, 24]),
    "voxelsize": np.array([1.0, 1.0, 1.0]),
    "xras": np.array([-1.0, 0.0, 0.0]),
    "yras": np.array([0.0, 0.0, -1.0]),
    "zras": np.array([0.0, 1.0, 0.0]),
    "cras": np.array([0.0, 0.0, 0.0]),
}


def run(subjects_dir):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--hemi", "lh", "--sd", str(subjects_dir), "--subject", "subj"],
        capture_output=True,
        text=True,
    )


def test_tunnel_is_removed_and_the_rest_of_the_surface_kept(tmp_path):
    mask = np.zeros((24, 24, 24), dtype=bool)
    mask[6:18, 6:18, 6:18] = True
    mask[11:13, 11:13, 6:18] = False  # a tunnel through the block
    for folder in ("surf", "scripts"):
        (tmp_path / "subj" / folder).mkdir(parents=True)
    surf = tmp_path / "subj" / "surf"
    v, t = voxel_surface(mask)
    TriaMesh(v, t, VOLUME_INFO).write_fssurf(str(surf / "lh.orig.nofix"))

    done = run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr

    record = json.loads((tmp_path / "subj" / "scripts" / "lh.topofix.json").read_text())
    assert record["status"] == "ok"
    assert record["genus"] == 1 and len(record["handles"]) == 1 and record["genus_corrected"] == 0
    premesh = TriaMesh.read_fssurf(str(surf / "lh.orig.premesh"))
    assert premesh.is_closed() and premesh.is_manifold() and premesh.is_oriented()
    assert premesh.genus() == 0 and premesh.volume() > 0
    assert len(self_intersections(premesh)) == 0
    # away from the tunnel, every vertex keeps its position after the averaging mris_fix_topology does
    far = np.max(np.abs(v[:, :2] - 11.5), axis=1) > 3
    averaged = average_vertex_positions(v, t)
    assert np.all(cKDTree(premesh.v).query(averaged[far])[0] < 1e-4)
    # the volume geometry of orig.nofix, which later FreeSurfer steps read
    assert premesh.fsinfo is not None and tuple(premesh.fsinfo["volume"]) == (24, 24, 24)


def test_a_failure_exits_with_1_and_records_the_error(tmp_path):
    (tmp_path / "subj" / "scripts").mkdir(parents=True)
    done = run(tmp_path)
    assert done.returncode == 1
    record = json.loads((tmp_path / "subj" / "scripts" / "lh.topofix.json").read_text())
    assert record["status"] == "failed" and "orig.nofix" in record["error"]
