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

"""Whether surgery along a loop cuts a bridge or fills a tunnel."""

from lapy import TriaMesh
from lapy.topology import find_handles
from shapes import block_with_bridge, voxel_surface

from recon_surf.topofix.classify import cuts_bridge, offset_ratio


def test_bridge_loop_cuts_and_hole_loop_fills():
    v, t = voxel_surface(block_with_bridge())
    (handle,) = find_handles(TriaMesh(v, t))
    short, long = sorted([handle.loop_a, handle.loop_b], key=len)
    assert offset_ratio(v, t, short) > 1 > offset_ratio(v, t, long)
    assert cuts_bridge(v, t, short) and not cuts_bridge(v, t, long)
