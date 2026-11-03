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

"""Print the number of labelled voxels of a segmentation outside a mask, both on the same voxel grid."""

import argparse
import sys

import nibabel as nib
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segmentation", help="path to the segmentation")
    parser.add_argument("mask", help="path to the mask, voxels > 0 are inside")
    args = parser.parse_args()

    seg_img, mask_img = nib.load(args.segmentation), nib.load(args.mask)
    # up to float32 rounding of the affine
    if seg_img.shape[:3] != mask_img.shape[:3] or not np.allclose(seg_img.affine, mask_img.affine, atol=1e-4):
        print(f"{args.mask} and {args.segmentation} are not on the same voxel grid.")
        return 1
    outside = (np.asanyarray(seg_img.dataobj) > 0) & (np.asanyarray(mask_img.dataobj) == 0)
    print(np.count_nonzero(outside))
    return 0


if __name__ == "__main__":
    sys.exit(main())
