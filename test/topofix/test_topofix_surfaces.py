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
written and genus 0, anything else starts the standard surface steps instead.
"""

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from lapy import TriaMesh
from lapy.intersect import self_intersections
from shapes import segmentation_block

FASTSURFER_HOME = Path(__file__).parent.parent.parent
SCRIPT = FASTSURFER_HOME / "recon_surf" / "topofix_surfaces.py"


def run(subjects_dir, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--hemi", "lh", "--sd", str(subjects_dir), "--subject", "subj", *args],
        capture_output=True,
        text=True,
    )


def test_tunnel_through_the_white_matter_is_removed(tmp_path):
    seg = segmentation_block()
    seg[11:13, 11:13, 6:18] = 17  # hippocampus, not white matter, so a tunnel through the block
    for folder in ("mri", "surf", "scripts"):
        (tmp_path / "subj" / folder).mkdir(parents=True)
    seg_path = tmp_path / "subj" / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    nib.MGHImage(seg.astype(np.int32), np.eye(4)).to_filename(seg_path)

    done = run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr

    record = json.loads((tmp_path / "subj" / "scripts" / "lh.topofix.json").read_text())
    assert record["status"] == "ok"
    assert record["genus"] == 1 and len(record["handles"]) == 1 and record["genus_corrected"] == 0
    surf = tmp_path / "subj" / "surf"
    nofix = TriaMesh.read_fssurf(str(surf / "lh.orig.nofix"))
    assert nofix.genus() == 1
    premesh = TriaMesh.read_fssurf(str(surf / "lh.orig.premesh"))
    assert len(self_intersections(premesh)) == 0
    for surface in (premesh, TriaMesh.read_fssurf(str(surf / "lh.orig"))):
        assert surface.is_closed() and surface.is_manifold() and surface.is_oriented()
        assert surface.genus() == 0 and surface.volume() > 0
        # the volume geometry of the segmentation, which later FreeSurfer steps read
        assert surface.fsinfo is not None and tuple(surface.fsinfo["volume"]) == seg.shape


def test_a_failure_exits_with_1_and_records_the_error(tmp_path):
    (tmp_path / "subj" / "scripts").mkdir(parents=True)
    done = run(tmp_path, "--seg", str(tmp_path / "missing.mgz"))
    assert done.returncode == 1
    record = json.loads((tmp_path / "subj" / "scripts" / "lh.topofix.json").read_text())
    assert record["status"] == "failed" and "missing.mgz" in record["error"]
