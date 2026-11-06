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

"""Cut or fill decided together for nearby handles, and whether a loop cuts or fills."""

from lapy import TriaMesh
from lapy.topology import find_handles, fix_topology, shorter_loop
from shapes import bar_on_two_pillars, block_with_bridge, voxel_surface

from recon_surf.topofix.classify import cuts_bridge, offset_ratio
from recon_surf.topofix.folds import crop_folds
from recon_surf.topofix.joint import FlipChooser, choose_jointly


def test_bridge_loop_cuts_and_hole_loop_fills():
    v, t = voxel_surface(block_with_bridge())
    (handle,) = find_handles(TriaMesh(v, t))
    short, long = sorted([handle.loop_a, handle.loop_b], key=len)
    assert offset_ratio(v, t, short) > 1 > offset_ratio(v, t, long)
    assert cuts_bridge(v, t, short) and not cuts_bridge(v, t, long)


def test_bar_is_kept_by_filling_between_the_pillars():
    v, t = voxel_surface(bar_on_two_pillars())
    assert TriaMesh(v, t).genus() == 1

    # one handle at a time, the shorter loop cuts one pillar and cropping removes the bar
    fixed, _, _ = fix_topology(TriaMesh(v, t))
    v2, _, _, _ = crop_folds(fixed.v, fixed.t)
    assert v2[:, 2].max() < 14

    centres, flips, groups = choose_jointly(v, t)
    assert flips.tolist() == [True]
    fixed, _, _ = fix_topology(TriaMesh(v, t), choose=FlipChooser(shorter_loop, centres, flips))
    v2, t2, _, _ = crop_folds(fixed.v, fixed.t)
    assert TriaMesh(v2, t2).genus() == 0
    assert v2[:, 2].max() > 16
