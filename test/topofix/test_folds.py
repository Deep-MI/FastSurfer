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

"""Cropping of protrusions on which a spherical map folds."""

import numpy as np
import pytest
from lapy import TriaMesh
from lapy.intersect import self_intersections
from shapes import mushroom, torus, voxel_surface

from recon_surf.topofix.folds import crop_folds, crop_region, sphere_folds


def test_plain_block_does_not_fold():
    mask = np.zeros((30, 30, 12), dtype=bool)
    mask[2:28, 2:28, 2:10] = True
    v, t = voxel_surface(mask)
    assert not sphere_folds(v, t).any()
    v2, t2, origin, history = crop_folds(v, t)
    assert history == [0]
    assert np.array_equal(v2, v) and np.array_equal(t2, t)


def test_mushroom_cap_is_cut_off_at_the_stalk():
    v, t = voxel_surface(mushroom())
    assert sphere_folds(v, t).sum() > 0
    v2, t2, origin, history = crop_folds(v, t)
    assert history[-1] == 0
    out = TriaMesh(v2, t2)
    assert out.is_closed() and out.is_manifold() and out.is_oriented() and out.genus() == 0
    assert len(self_intersections(out)) == 0
    # the cap starts at z = 14; the body ends at z = 10
    assert v2[:, 2].max() < 14
    body = v[:, 2] <= 10
    kept = origin[origin >= 0]
    assert np.all(np.isin(np.flatnonzero(body), kept))
    assert np.array_equal(v2[origin >= 0], v[kept])


def test_region_whose_boundary_touches_itself_is_rejected():
    # removing two triangles that share only vertex 0 leaves a hole whose
    # boundary passes through vertex 0 twice
    v, t = torus()
    fan = np.flatnonzero(np.any(t == 0, axis=1))

    def region_of(i, j):
        mask = np.zeros(len(v), dtype=bool)
        mask[np.union1d(t[i], t[j])] = True
        return mask

    region = next(
        region_of(i, j)
        for i in fan
        for j in fan
        if len(np.intersect1d(t[i], t[j])) == 1 and np.all(region_of(i, j)[t], axis=1).sum() == 2
    )
    with pytest.raises(ValueError, match="touches itself"):
        crop_region(v, t, region)
