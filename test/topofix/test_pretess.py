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

"""Joining a mask at edge and corner contacts, against mri_pretess where it is installed."""

import shutil
import subprocess

import nibabel as nib
import numpy as np
import pytest

from recon_surf.topofix.pretess import pretess

MODES = ("fixed", "freesurfer")


def random_volume(seed, shape=(18, 20, 22)):
    """A mask with many edge and corner contacts, and intensities with ties."""
    rng = np.random.default_rng(seed)
    return rng.random(shape) < 0.35, rng.integers(0, 8, shape).astype(np.uint8)


@pytest.mark.parametrize("mode", MODES)
def test_edge_contact_gets_the_brighter_face_neighbour(mode):
    mask = np.zeros((4, 4, 3), dtype=bool)
    mask[1, 1, 1] = mask[2, 2, 1] = True
    val = np.zeros(mask.shape)
    val[1, 2, 1], val[2, 1, 1] = 50, 40
    added = pretess(mask, val, mode=mode) & ~mask
    assert np.argwhere(added).tolist() == [[1, 2, 1]]


@pytest.mark.parametrize("mode", MODES)
def test_corner_contact_gets_the_brightest_path(mode):
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = mask[2, 2, 2] = True
    val = np.full(mask.shape, 10.0)
    val[1, 2, 1], val[1, 2, 2] = 30, 30  # the path over the second and then the third axis
    added = pretess(mask, val, mode=mode) & ~mask
    assert sorted(np.argwhere(added).tolist()) == [[1, 2, 1], [1, 2, 2]]


@pytest.mark.parametrize("mode", MODES)
def test_empty_corner_contact_gets_the_brighter_voxel(mode):
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    mask[1, 1, 1] = mask[2, 2, 2] = False
    val = np.zeros(mask.shape)
    val[1, 1, 1], val[2, 2, 2] = 20, 30
    added = pretess(mask, val, mode=mode) & ~mask
    assert np.argwhere(added).tolist() == [[2, 2, 2]]


@pytest.mark.skipif(shutil.which("mri_pretess") is None, reason="needs FreeSurfer's mri_pretess")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_freesurfer_mode_is_mri_pretess(tmp_path, seed):
    mask, val = random_volume(seed)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    nib.MGHImage(np.where(mask, 255, 0).astype(np.uint8), affine).to_filename(tmp_path / "mask.mgz")
    nib.MGHImage(val, affine).to_filename(tmp_path / "val.mgz")
    subprocess.run(
        ["mri_pretess", str(tmp_path / "mask.mgz"), "255", str(tmp_path / "val.mgz"), str(tmp_path / "out.mgz")],
        check=True, capture_output=True,
    )
    expected = np.asarray(nib.load(tmp_path / "out.mgz").dataobj) == 255
    joined = pretess(mask, val, mode="freesurfer")
    assert (expected & ~mask).sum() > 20
    np.testing.assert_array_equal(joined, expected)


@pytest.mark.parametrize("seed", [0, 1])
def test_fixed_mode_leaves_no_contact_and_ignores_the_voxel_order(seed):
    mask, val = random_volume(seed)
    joined = pretess(mask, val)
    assert (joined & ~mask).sum() > 20
    # nothing left to join
    np.testing.assert_array_equal(pretess(joined, val), joined)
    # the same voxels for mirrored and permuted axes
    for flip in ((0,), (1,), (2,), (0, 1, 2)):
        np.testing.assert_array_equal(np.flip(pretess(np.flip(mask, flip), np.flip(val, flip)), flip), joined)
    perm = (2, 0, 1)
    permuted = pretess(np.transpose(mask, perm), np.transpose(val, perm))
    np.testing.assert_array_equal(np.transpose(permuted, np.argsort(perm)), joined)
