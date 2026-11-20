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
Print the number of voxels set to 255 in brain.finalsurfs.manedit.mgz that the pial surface treats as non-brain.

mris_place_surface labels voxels brighter than 125 outside wm.mgz with fewer than 14 of their 26 neighbours in
wm.mgz as bright non-brain (MRIfindBrightNonWM), and zeroes them for the pial surface. Both volumes must be on the
same voxel grid.
"""

import argparse
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manedit", help="path to brain.finalsurfs.manedit.mgz")
    parser.add_argument("wm", help="path to wm.mgz")
    args = parser.parse_args()

    edit_img, wm_img = nib.load(args.manedit), nib.load(args.wm)
    # up to float32 rounding of the affine
    if edit_img.shape[:3] != wm_img.shape[:3] or not np.allclose(edit_img.affine, wm_img.affine, atol=1e-4):
        print(f"{args.manedit} and {args.wm} are not on the same voxel grid.")
        return 1
    wm = np.asanyarray(wm_img.dataobj) >= 5
    wm_neighbours = ndimage.convolve(wm.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant")
    bright = (np.asanyarray(edit_img.dataobj) == 255) & ~wm & (wm_neighbours < 14)
    print(np.count_nonzero(bright))
    return 0


if __name__ == "__main__":
    sys.exit(main())
