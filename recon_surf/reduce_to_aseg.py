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
import copy

import scipy.ndimage
from skimage.measure import label
from skimage.filters import gaussian

HELPTEXT = """
Script to reduce aparc+aseg to aseg by mapping cortex lables back to left/right GM.

If --outmask is used, it also creates a brainmask by dilating (5) and eroding (4) 
the segmentation, and then selecting the largest component. In that case also the 
segmentation is masked (to remove small components outside the main brain region).

If --flipwm is passed, disconnected WM islands will be checked and potentially
swapped to the other hemisphere. Sometimes these islands carry the wrong label 
and are far from the main body into the other hemisphere. This will cause mri_cc
to become really slow as it needs to cover a large search box. 


USAGE:
reduce_to_aseg  -i <input_seg> -o <output_seg>

    
Dependencies:
    Python 3.8

    Numpy
    http://www.numpy.org

    Nibabel to read and write FreeSurfer data
    http://nipy.org/nibabel/
    
    skimage for erosion, dilation, connected component
    https://scikit-image.org/

Original Author: Martin Reuter
Date: Jul-24-2018

"""

h_input = 'path to input segmentation'
h_output = 'path to ouput segmentation'
h_outmask = 'path to ouput mask'
h_fixwm = 'whether to try to flip lables of disconnected WM island to other hemi'


def options_parse():
    """
    Command line option parser for reduce_to_aseg.py
    """
    parser = optparse.OptionParser(version='$Id: reduce_to_aseg.py,v 1.0 2018/06/24 11:34:08 mreuter Exp $',
                                   usage=HELPTEXT)
    parser.add_option('--input', '-i',  dest='input_seg',   help=h_input)
    parser.add_option('--output', '-o', dest='output_seg',  help=h_output)
    parser.add_option('--outmask',      dest='output_mask', help=h_outmask)
    parser.add_option('--fixwm',        dest='fix_wm',      help=h_fixwm, default=False, action="store_true")
    (options, args) = parser.parse_args()

    if options.input_seg is None or options.output_seg is None:
        sys.exit('ERROR: Please specify input and output segmentations')

    return options


def reduce_to_aseg(data_inseg):
    print ("Reducing to aseg ...")
    # replace 2000... with 42
    data_inseg[data_inseg >= 2000] = 42
    # replace 1000... with 3
    data_inseg[data_inseg >= 1000] = 3
    return data_inseg


def create_mask(aseg_data, dnum, enum):

    print ("Creating dilated mask ...")

    # treat lateral orbital frontal and parsorbitalis special to avoid capturing too much of eye nerve
    lat_orb_front_mask = np.logical_or(aseg_data == 2012, aseg_data == 1012)
    parsorbitalis_mask = np.logical_or(aseg_data == 2019, aseg_data == 1019)
    frontal_mask = np.logical_or(lat_orb_front_mask, parsorbitalis_mask)
    print("Frontal region special treatment: ", format(np.sum(frontal_mask)))

    # reduce to binary
    datab = (aseg_data > 0)
    datab[frontal_mask] = 0
    # dilate and erode
    datab = scipy.ndimage.binary_dilation(datab, np.ones((3, 3, 3)), iterations=dnum)
    datab = scipy.ndimage.binary_erosion(datab, np.ones((3, 3, 3)), iterations=enum)
    #for x in range(dnum):
    #    datab = binary_dilation(datab, np.ones((3, 3, 3)))
    #for x in range(enum):
    #    datab = binary_erosion(datab, np.ones((3, 3, 3)))

    # extract largest component
    labels = label(datab)
    assert (labels.max() != 0)  # assume at least 1 real connected component
    print("  Found {} connected component(s)!".format(labels.max()))

    if labels.max() > 1:
        print("  Selecting largest component!")
        datab = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1)

    # add frontal regions back to mask
    datab[frontal_mask] = 1

    # set mask
    aseg_data[~datab] = 0
    aseg_data[datab] = 1
    return aseg_data


def flip_wm_islands(aseg_data):
# Sometimes WM is far in the other hemisphere, but with a WM label from the other hemi
# These are usually islands, not connected ot the main hemi WM component
# Here we decide to flip assignment based on proximity to other WM and GM labels

    #label ids
    lh_wm = 2
    lh_gm = 3
    rh_wm = 41
    rh_gm = 42

    # for lh get largest component and islands
    mask = (aseg_data == lh_wm)
    labels = label(mask, background=0)
    assert( labels.max() != 0 ) # assume at least 1 connected component
    bc = np.bincount(labels.flat)[1:]
    largestID = np.argmax(bc)+1
    largestCC = (labels == largestID)
    lh_islands = ( ~ largestCC) & (labels > 0)

    # same for rh
    mask = (aseg_data == rh_wm)
    labels = label(mask, background=0)
    assert( labels.max() != 0 ) # assume at least 1 CC
    bc = np.bincount(labels.flat)[1:]
    largestID = np.argmax(bc)+1
    largestCC = labels == largestID
    rh_islands = (labels != largestID ) & (labels > 0)

    # get signed probability for lh and rh (by smoothing joined GM+WM labels)
    lhmask = (aseg_data == lh_wm) | (aseg_data == lh_gm)
    rhmask = (aseg_data == rh_wm) | (aseg_data == rh_gm)
    ii = gaussian(lhmask.astype(float) * (-1) + rhmask.astype(float), sigma=1.5)

    # flip island
    rhswap = rh_islands & (ii < 0.0)
    lhswap = lh_islands & (ii > 0.0)
    flip_data = aseg_data.copy()
    flip_data[rhswap] = lh_wm
    flip_data[lhswap] = rh_wm
    print("FlipWM: rh {} and lh {} flipped.".format(rhswap.sum(),lhswap.sum()))

    return flip_data


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()

    print("Reading in aparc+aseg: {} ...".format(options.input_seg))
    inseg = nib.load(options.input_seg)
    inseg_data = np.asanyarray(inseg.dataobj)
    inseg_header = inseg.header
    inseg_affine = inseg.affine

    # Change datatype to np.uint8
    inseg_header.set_data_dtype(np.uint8)

    # get mask
    if options.output_mask:
        bm = create_mask(copy.deepcopy(inseg_data), 5, 4)
        print ("Outputing mask: {}".format(options.output_mask))
        mask = nib.MGHImage(bm, inseg_affine, inseg_header)
        mask.to_filename(options.output_mask)

    # reduce aparc to aseg and mask regions
    aseg = reduce_to_aseg(inseg_data)

    if options.output_mask:
        # mask aseg also
        aseg[bm == 0] = 0

    if options.fix_wm:
        aseg = flip_wm_islands(aseg)

    print ("Outputing aseg: {}".format(options.output_seg))
    aseg_fin = nib.MGHImage(aseg, inseg_affine, inseg_header)
    aseg_fin.to_filename(options.output_seg)

    sys.exit(0)
