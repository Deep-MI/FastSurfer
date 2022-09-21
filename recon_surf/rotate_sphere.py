#!/usr/bin/env python3


# Copyright 2021 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
import numpy as np
import sys
import nibabel.freesurfer.io as fs
import align_points as align


HELPTEXT = """

Script to rigidly align two spheres (=rotation) using APARCs

USAGE:
rotate_sphere.py --srcsphere <?h.sphere> --srcaparc <?h.aparc>
                 --trgsphere <?h.sphere> --trgaparc <?h.aparc> 
                 --out <angles.txt>

Dependencies:
    Python 3.8

    SimpleITK https://simpleitk.org/ (v2.1.1)


Description:

For each common segmentation ID in the two aparcs, the centroid coordinate is 
computed (and mapped back to the sphere of radius 100, FS format). The point pairs
are then aligned by finding the rotation that minimizes their SSD. The output file
will contain the angles (alpha,beta,gama) as expected by mris_register for rotation
initialization.


Original Author: Martin Reuter
Date: Jun-8-2022
"""


# In the future, maybe add a way to specify what labels to align as a list or 
# txt file to pass to the routine. 
# Also add output as LTA instead of registration angles.

h_srcsphere = 'path to src ?h.sphere'
h_srcaparc  = 'path to src corresponding cortical parcellation'
h_trgsphere = 'path to trg ?h.sphere'
h_trgaparc  = 'path to trg corresponding cortical parcellation'
h_out       = 'path to output txt files for angles'

def options_parse():
    """
    Command line option parser
    """
    parser = optparse.OptionParser(version='$Id: rotate_sphere.py,v 1.0 2022/03/18 21:22:08 mreuter Exp $', usage=HELPTEXT)
    parser.add_option('--srcsphere', dest='srcsphere', help=h_srcsphere)
    parser.add_option('--srcaparc',  dest='srcaparc',  help=h_srcaparc)
    parser.add_option('--trgsphere', dest='trgsphere', help=h_trgsphere)
    parser.add_option('--trgaparc',  dest='trgaparc',  help=h_trgaparc)
    parser.add_option('--out',       dest='out',       help=h_out)
    (options, args) = parser.parse_args()

    if options.srcsphere is None or options.srcaparc is None or options.trgsphere is None or options.trgaparc is None or options.out is None:
        sys.exit('\nERROR: Please specify src and target sphere and parcellation files as well as output txt file\n   Use --help to see all options.\n')

    return options


def align_aparc_centroids(v_mov, labels_mov, v_dst, labels_dst, label_ids=[]):
# Aligns centroid of aparc parcels on the sphere (Attention mapping back to sphere!)
    # inferiorparietal,inferiortemporal,lateraloccipital,postcentral, posteriorsingulate
    #  precentral, precuneus, superiorfrontal, supramarginal
    #lids=np.array([8,9,11,22,23,24,25,28,31])
    #lids=np.array([8,9,22,24,31])
    # lids=np.array([8,22,24])
    if not label_ids:
        # use all joint labels except -1 and 0:
        lids=np.intersect1d(labels_mov,labels_dst)
        lids=lids[(lids>0)]
    else:
        lids=label_ids
    # compute mean for each label id
    counter=0
    centroids_mov=np.empty([lids.size,3])
    centroids_dst=np.empty([lids.size,3])
    for id in lids:
        centroids_mov[counter] = np.mean(v_mov[(labels_mov==id),:],axis=0)
        centroids_dst[counter] = np.mean(v_dst[(labels_dst==id),:],axis=0)
        counter=counter+1
    # map back to sphere of radius 100
    centroids_mov = (100/np.sqrt(np.sum(centroids_mov*centroids_mov,axis=1)))[:, np.newaxis] * centroids_mov
    centroids_dst = (100/np.sqrt(np.sum(centroids_dst*centroids_dst,axis=1)))[:, np.newaxis] * centroids_dst
    # find rotation
    R = align.find_rotation(centroids_mov, centroids_dst)    
    return R



if __name__ == "__main__":

    # Command Line options are error checking done here
    options = options_parse()

    print()
    print("Rotate Sphere Parameters:")
    print()
    print("- src sphere {}".format(options.srcsphere))
    print("- src aparc: {}".format(options.srcaparc))
    print("- trg sphere {}".format(options.trgsphere))
    print("- trg aparc: {}".format(options.trgaparc))
    print("- out txt {}".format(options.out))
    
    # read image (only nii supported) and convert to float32
    print("\nreading {}".format(options.srcsphere))
    srcsphere = fs.read_geometry(options.srcsphere, read_metadata=True)
    print("reading annotation: {} ...".format(options.srcaparc))
    srcaparc = fs.read_annot(options.srcaparc)
    print("reading {}".format(options.trgsphere))
    trgsphere = fs.read_geometry(options.trgsphere, read_metadata=True)
    print("reading annotation: {} ...".format(options.trgaparc))
    trgaparc = fs.read_annot(options.trgaparc)

    R = align_aparc_centroids(srcsphere[0],srcaparc[0],trgsphere[0],trgaparc[0])
    alpha,beta,gamma = align.rmat2angles(R)
    print("\nalpha {:.1f}   beta {:.1f}   gamma {:.1f}\n".format(alpha,beta,gamma))

    # write angles
    print("writing: {}".format(options.out))
    f = open(options.out, "w")
    f.write("{:.1f} {:.1f} {:.1f}\n".format(alpha,beta,gamma))
    f.close()
    print("...done\n")
    
    sys.exit(0)





