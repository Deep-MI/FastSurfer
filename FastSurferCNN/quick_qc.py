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


HELPTEXT = """
Script to perform quick qualtiy checks for the input segmentation to identify gross errors.

USAGE:
quick_qc.py --aparc_aseg_segfile <aparc+aseg.mgz>


"""

def options_parse():
    """
    Command line option parser
    """
    parser = optparse.OptionParser(version='$Id: quick_qc,v 1.0 2022/09/28 11:34:08 mreuter Exp $', usage=HELPTEXT)
    parser.add_option('--aparc_aseg_segfile',  dest='aparc_aseg_segfile', help="Input aparc+aseg segmentation to be checked")

    (options, args) = parser.parse_args()

    if options.aparc_aseg_segfile is None:
        sys.exit('ERROR: Please specify input segmentation --aparc_aseg_segfile <filename>')

    return options


def check_volume(aparc_aseg_segfile, voxvol, thres=0.70):
    print("Checking total volume ...")
    mask = (aparc_aseg_segfile > 0)
    total_vol = np.sum(mask) * voxvol / 1000000
    print("Voxel size in mm3: {}".format(voxvol))
    print("Total segmentation volume in liter: {}".format(np.round(total_vol,2)))
    if (total_vol < thres):
        return False

    return True


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()
    print("Reading in aparc+aseg: {} ...".format(options.aparc_aseg_segfile))
    inseg = nib.load(options.aparc_aseg_segfile)
    inseg_data = np.asanyarray(inseg.dataobj)
    inseg_header = inseg.header
    inseg_voxvol = np.product(inseg_header.get_zooms())

    if not check_volume(inseg_data, inseg_voxvol):
        print('WARNING: Total segmentation volume is very small. Segmentation may be corrupted! Please check.')
        sys.exit(0)
    else:
        sys.exit(0)
