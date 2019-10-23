
# Copyright 2019 Image Analysis Lab, German Center for Neurodegenrative Diseases (DZNE), Bonn
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
import os
import sys
import numpy as np
import nibabel as nib
import copy


HELPTEXT = """
Script to reduce aparc+aseg to aseg by mapping cortex lables back to left/right GM.
If --outmask is used, it also creates a brainmask by dilating (5) and eroding (4) 
the segmentation, and then selecting the largest component. In that case also the 
segmentation is masked (to remove small components outside the main brain region).


USAGE:
reduce_to_aseg  -i <input_seg> -o <output_seg>

    
Dependencies:
    Python 3.5

    Numpy
    http://www.numpy.org

    Nibabel to read and write FreeSurfer data
    http://nipy.org/nibabel/
    
    skimage for erosion, dilation, connected component
    https://scikit-image.org/

Original Author: Martin Reuter
Date: Jul-24-2018

"""

h_input  = 'path to input segmentation'
h_output = 'path to ouput segmentation'
h_outmask = 'path to ouput mask'


def options_parse():
    """
    Command line option parser for reduce_to_aseg.py
    """
    parser = optparse.OptionParser(version='$Id: reduce_to_aseg.py,v 1.0 2018/06/24 11:34:08 mreuter Exp $', usage=HELPTEXT)
    parser.add_option('--input',  '-i', dest='input_seg', help=h_input)
    parser.add_option('--output', '-o', dest='output_seg', help=h_output)
    parser.add_option('--outmask', dest='output_mask', help=h_outmask)
    (options, args) = parser.parse_args()

    if options.input_seg is None or options.output_seg is None:
        sys.exit('ERROR: Please specify input and output segmentations')

    return options



def reduce_to_aseg(inseg):
    print ("Reducing to aseg ...")
    data = inseg.get_data()
    # replace 2000... with 42
    data[data>= 2000] = 42
    # replace 1000... with 3
    data[data>=1000] = 3
    return inseg


def create_mask(aseg, dnum, enum):
    from skimage.morphology import binary_dilation, binary_erosion
    from skimage.measure import label
    print ("Creating dilated mask ...")
    data = aseg.get_data()
    
    # treat lateral orbital frontal and parsorbitalis special to avoid capturing too much of eye nerve
    lat_orb_front_mask = np.logical_or(data==2012, data==1012)
    parsorbitalis_mask = np.logical_or(data==2019, data==1019)
    frontal_mask = np.logical_or(lat_orb_front_mask, parsorbitalis_mask)
    print("Frontal region special treatment: ",format(np.sum(frontal_mask)))
    
    # reduce to binary
    datab = (data>0)
    datab[frontal_mask] = 0
    # dilate and erode
    for x in range(dnum):
        datab = binary_dilation(datab,np.ones((3,3,3)))
    for x in range(enum):
        datab = binary_erosion(datab,np.ones((3,3,3)))
    # extract largest component
    labels = label(datab)
    assert( labels.max() != 0 ) # assume at least 1 real connected component
    print("  Found {} connected component(s)!".format(labels.max()))
    if (labels.max() > 1):
        print("  Selecting largest component!")
        datab = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)
	
    # add frontal regions back to mask
    datab[frontal_mask] = 1
    
    # set mask
    data[~datab] = 0
    data[datab] = 1
    return aseg
    


if __name__=="__main__":
    # Command Line options are error checking done here
    options = options_parse()

    print("Reading in aparc+aseg: {} ...".format(options.input_seg))
    inseg = nib.load(options.input_seg)
    
    # get mask
    if options.output_mask:
        bm = create_mask(copy.deepcopy(inseg),5,4)
    
    aseg = reduce_to_aseg(inseg)
    aseg.set_data_dtype(np.uint8)

    if options.output_mask:
        bm.set_data_dtype(np.uint8)
        print ("Outputing mask: {}".format(options.output_mask))
        nib.save(bm, options.output_mask)
	# mask aseg also
        data = aseg.get_data()
        data[bm.get_data()==0] = 0
	
    print ("Outputing aseg: {}".format(options.output_seg))
    nib.save(aseg, options.output_seg)

    sys.exit(0)


