# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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


# IMPORTS
import optparse
import sys
import numpy as np
import nibabel as nib

from skimage.morphology import binary_dilation


HELPTEXT = """
Script to perform quick qualtiy checks for the input segmentation to identify gross errors.

USAGE:
quick_qc.py --asegdkt_segfile <aparc+aseg.mgz>


"""

VENT_LABELS = {
    "Left-Lateral-Ventricle": 4,
    "Right-Lateral-Ventricle": 43,
    "Left-choroid-plexus": 31,
    "Right-choroid-plexus": 63,
}
BG_LABEL = 0


def options_parse():
    """Command line option parser"""
    parser = optparse.OptionParser(
        version="$Id: quick_qc,v 1.0 2022/09/28 11:34:08 mreuter Exp $", usage=HELPTEXT
    )
    parser.add_option(
        "--asegdkt_segfile",
        "--aparc_aseg_segfile",
        dest="asegdkt_segfile",
        help="Input aparc+aseg segmentation to be checked",
    )

    (options, args) = parser.parse_args()

    if options.asegdkt_segfile is None:
        sys.exit(
            "ERROR: Please specify input segmentation --asegdkt_segfile <filename>"
        )

    return options


def check_volume(asegdkt_segfile, voxvol, thres=0.70):
    """

    Parameters
    ----------
    asegdkt_segfile :
        
    voxvol :
        
    thres :
         (Default value = 0.70)

    Returns
    -------

    """
    print("Checking total volume ...")
    mask = asegdkt_segfile > 0
    total_vol = np.sum(mask) * voxvol / 1000000
    print("Voxel size in mm3: {}".format(voxvol))
    print("Total segmentation volume in liter: {}".format(np.round(total_vol, 2)))
    if total_vol < thres:
        return False

    return True


def get_region_bg_intersection_mask(
    seg_array, region_labels=VENT_LABELS, bg_label=BG_LABEL
):
    """Returns a mask of the intersection between the voxels of a given region and background voxels.
    
    This is obtained by dilating the region by 1 voxel and computing the intersection with the
    background mask.
    
    The region can be defined by passing in the region_labels dict.

    Parameters
    ----------
    seg_array : numpy.ndarray
        Segmentation array
    region_labels : dict
        dict whose values correspond to the desired region's labels (Default value = VENT_LABELS)
    bg_label :
         (Default value = BG_LABEL)

    Returns
    -------
    bg_intersect : numpy.ndarray
        Region and background intersection mask array

    
    """

    region_array = seg_array.copy()
    conditions = np.all(
        np.array([(region_array != value) for value in region_labels.values()]), axis=0
    )
    region_array[conditions] = 0
    region_array[region_array != 0] = 1

    bg_array = seg_array.copy()
    bg_array[bg_array != bg_label] = -1.0
    bg_array[bg_array == bg_label] = 1
    bg_array[bg_array != 1] = 0

    region_array_dilated = binary_dilation(region_array)

    bg_intersect = np.bitwise_and(
        region_array_dilated.astype(int), bg_array.astype(int)
    )

    return bg_intersect


def get_ventricle_bg_intersection_volume(seg_array, voxvol):
    """Returns a volume estimate for the intersection of ventricle voxels with background voxels.

    Parameters
    ----------
    seg_array : numpy.ndarray
        Segmentation array
    voxvol : float
        Voxel volume

    Returns
    -------
    intersection_volume : float
        Estimated volume of voxels in ventricle and background intersection

    
    """
    bg_intersect_mask = get_region_bg_intersection_mask(seg_array)
    intersection_volume = bg_intersect_mask.sum() * voxvol

    return intersection_volume


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()
    print("Reading in aparc+aseg: {} ...".format(options.asegdkt_segfile))
    inseg = nib.load(options.asegdkt_segfile)
    inseg_data = np.asanyarray(inseg.dataobj)
    inseg_header = inseg.header
    inseg_voxvol = np.product(inseg_header.get_zooms())

    # Ventricle-BG intersection volume check:
    print("Estimating ventricle-background intersection volume...")
    print(
        "Ventricle-background intersection volume in mm3: {:.2f}".format(
            get_ventricle_bg_intersection_volume(inseg_data, inseg_voxvol)
        )
    )

    # Total volume check:
    if not check_volume(inseg_data, inseg_voxvol):
        print(
            "WARNING: Total segmentation volume is very small. Segmentation may be corrupted! Please check."
        )
        sys.exit(0)
    else:
        sys.exit(0)
