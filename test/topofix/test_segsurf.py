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

"""White matter surfaces from a segmentation."""

import numpy as np
from lapy import TriaMesh
from shapes import segmentation_block

from recon_surf.topofix.segsurf import extract_surface, white_matter_mask


def test_block_gives_closed_outward_surface_in_surface_coordinates():
    vox2ras = np.eye(4)
    vox2ras[:3, 3] = [-100.0, 20.0, 5.0]
    v, t = extract_surface(white_matter_mask(segmentation_block(), "lh"), vox2ras)
    surface = TriaMesh(v, t)
    assert surface.is_closed() and surface.is_manifold() and surface.is_oriented() and surface.genus() == 0
    # outward orientation and roughly the 12 x 12 x 12 voxel block
    assert 0.9 * 12**3 < surface.volume() < 1.1 * 12**3
    np.testing.assert_allclose(v.min(axis=0), np.array([5.5, 5.5, 5.5]) + vox2ras[:3, 3], atol=0.3)
    np.testing.assert_allclose(v.max(axis=0), np.array([17.5, 17.5, 17.5]) + vox2ras[:3, 3], atol=0.3)


def test_mirrored_voxel_axes_keep_outward_orientation():
    # conformed volumes map voxels to surface coordinates with a mirroring
    vox2ras = np.array([[-1.0, 0, 0, 12], [0, 0, 1, -12], [0, -1, 0, 12], [0, 0, 0, 1]])
    v, t = extract_surface(white_matter_mask(segmentation_block(), "lh"), vox2ras)
    assert TriaMesh(v, t).volume() > 0


def test_enclosed_subcortical_structures_count_as_white_matter():
    seg = segmentation_block()
    seg[11:13, 11:13, 6:18] = 10  # thalamus fills a tunnel through the block
    v, t = extract_surface(white_matter_mask(seg, "lh"), np.eye(4))
    assert TriaMesh(v, t).genus() == 0
    seg[11:13, 11:13, 6:18] = 17  # hippocampus does not, so the tunnel stays open
    v, t = extract_surface(white_matter_mask(seg, "lh"), np.eye(4))
    assert TriaMesh(v, t).genus() == 1


def test_hemispheres_and_hypointensities():
    seg = np.zeros((40, 24, 24), dtype=np.int64)
    seg[4:18, 4:20, 4:20] = 2
    seg[18:36, 4:20, 4:20] = 41
    seg[8:12, 10:14, 10:14] = 77  # inside the left white matter
    seg[28:32, 10:14, 10:14] = 77  # inside the right white matter
    left, right = white_matter_mask(seg, "lh"), white_matter_mask(seg, "rh")
    assert left[8:12, 10:14, 10:14].all() and not right[8:12, 10:14, 10:14].any()
    assert right[28:32, 10:14, 10:14].all() and not left[28:32, 10:14, 10:14].any()
    v, _ = extract_surface(left, np.eye(4))
    assert v[:, 0].max() < 18
