#!/usr/bin/env python3


# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
from sklearn.neighbors import KDTree



HELPTEXT = """

Script to map surface labels across surfaces based on given sphere registrations

map_surf_label.py --srclabel <in.label> --srcsphere <sphere.reg>
                  --trgsphere <sphere.reg> --trgsurf <white> --trgsid <sid>
                  --outlabel <out.label>


Dependencies:
    Python 3.8
    numpy, nibabel, sklearn
    

Description:
Computes correspondence between source and target spheres and maps src-label to target.
Target surf (usually the white) coordinates at label vertices and SID is only written
to output label and not used for any computations. 

Original Author: Martin Reuter
Date: Aug-24-2022
"""


h_srclabel   = 'path to src surface label file'
h_srcsphere  = 'path to src sphere.reg'
h_trgsphere  = 'path to trg sphere.reg'
h_trgsurf    = 'path to trg surf (usually white) to write coordinates into label file'
h_trgsid     = 'target subject id, also written into label file'
h_outlabel   = 'output label file'

def options_parse():
    """
    Command line option parser
    """
    parser = optparse.OptionParser(version='$Id:map_surf_label.py,v 1.0 2022/08/24 21:22:08 mreuter Exp $', usage=HELPTEXT)
    parser.add_option('--srclabel',  dest='srclabel',  help=h_srclabel)
    parser.add_option('--srcsphere', dest='srcsphere', help=h_srcsphere)
    parser.add_option('--trgsphere', dest='trgsphere', help=h_trgsphere)
    parser.add_option('--trgsurf',   dest='trgsurf',   help=h_trgsurf)
    parser.add_option('--trgsid',    dest='trgsid',    help=h_trgsid)
    parser.add_option('--outlabel',  dest='outlabel',  help=h_outlabel)
    (options, args) = parser.parse_args()
    if options.srclabel is None or options.srcsphere is None or options.trgsphere is None \
       or options.trgsurf is None or options.trgsid is None or options.outlabel is None:
        sys.exit('\nERROR: Please specify all parameters!\n   Use --help to see all options.\n')
    return options



def writeSurfLabel(filename, sid, label, values, surf):
    # writes a FS surface label file to filename (e.g. lh.labelname.label)
    # stores sid string in the header, then number of vertices
    # and table of vertex index, RAS wm-surface coords (taken from surf)
    # and values (which can be zero)
    if values is None:
        values = np.zeros(label.shape)
    if values.size != label.size :
        raise ValueError("writeLabel Error: label and values should have same sizes {}!={}".format(label.size,values.size))
    coords = surf[label,:]
    header="#!ascii label  , from subject {} vox2ras=TkReg \n{}".format(sid,label.size)
    data = np.column_stack([label, coords, values])
    np.savetxt(filename, data, fmt=['%d','%.3f','%.3f','%.3f','%.6f'], header=header, comments='')


def getSurfCorrespondence(src_sphere, trg_sphere, tree=None):
    # For each vertex in src_sphere finds the closest vertex in trg_sphere
    # spheres are Nx3 arrays of coordinates on the sphere (usually R=100 FS format)
    # *_sphere can also be a file name of the sphere.reg files, then we load it.
    # Will return indices, distances and the KDtree of the trg surface.
    # The KDtree can be passed in cases where src moves around and trg stays fixed
    #
    # We can also work with file names instead of surface vertices
    if isinstance(src_sphere, str):
        src_sphere = fs.read_geometry(src_sphere, read_metadata=False)[0]
    if isinstance(trg_sphere, str):
        trg_sphere = fs.read_geometry(trg_sphere, read_metadata=False)[0]
    # if someone passed the full output of fs.read_geometry
    if isinstance(src_sphere,tuple):
        src_sphere = src_sphere[0]
    if isinstance(trg_sphere,tuple):
        trg_sphere = trg_sphere[0]
    # create tree if necessary and compute mapping
    if tree is None:
        from sklearn.neighbors import KDTree
        tree = KDTree(trg_sphere)
    distances, mapping = tree.query(src_sphere, 1)
    return mapping, distances, tree


def mapSurfLabel(src_label_name, out_label_name, trg_surf, trg_sid, rev_mapping):
    # maps a label from src surface according to the correspondence
    # in rev_mapping (! a mapping from target to source, listing the 
    # corresponding src vertex for each vertex on the trg surface)
    # trg_surf is passed so that the label file will list the 
    # correct coordinates (usually the white surface), can be vetrices or filename
    # trg_sid is the subject id (str) of the target subject (as 
    # stored in the output label file header)
    print("Mapping label: {} ...".format(src_label_name))
    src_label, src_values = fs.read_label(src_label_name, read_scalars=True)
    smax = max(np.max(src_label),np.max(rev_mapping)) + 1
    tmax = rev_mapping.size
    if isinstance(trg_surf, str):
        print("Reading in surface: {} ...".format(trg_surf))
        trg_surf = fs.read_geometry(trg_surf, read_metadata=False)[0]
    if trg_surf.shape[0] != tmax:
        raise ValueError("mapSurfLabel Error: label and trg vertices should have same sizes {}!={}".format(tmax,trg_surf.shape[0]))
    inside = np.zeros(smax, dtype=bool)
    inside[src_label] = True
    values = np.zeros(smax)
    values[src_label] = src_values
    inside_trg = inside[rev_mapping]
    trg_label = np.nonzero(inside[rev_mapping])[0]
    trg_values = values[rev_mapping[trg_label]]
    #print(trg_values)
    #print(trg_label.size)
    if out_label_name is not None:
        writeSurfLabel(out_label_name,trg_sid,trg_label,trg_values,trg_surf)
    return trg_label, trg_values


if __name__ == "__main__":

    # Command line options and error checking done here
    options = options_parse()

    print()
    print("Map Surface Labels Parameters:")
    print()
    print("- src label {}".format(options.srclabel))
    print("- src sphere {}".format(options.srcsphere))
    print("- trg sphere {}".format(options.trgsphere))
    print("- trg surf {}".format(options.trgsurf))
    print("- trg sid {}".format(options.trgsid))
    print("- out label {}".format(options.outlabel))    

    # for example:
    #src_label_name = "fsaverage/label/lh.BA1_exvivo.label"
    #out_label_name = "lh.BA1_exvivo_my.label"
    #src_sphere_name = "fsaverage/surf/lh.sphere.reg" # identical to fsaverage sphere
    #trg_sphere_name = "OAS1_0111_MR1/surf/lh.sphere72_my4.reg"
    #trg_white_name = "OAS1_0111_MR1/surf/lh.white"
    #trg_sid = "OAS1_0111_MR1"
    
    # ./map_surf_label.py --srclabel fsaverage/label/lh.BA1_exvivo.label --srcsphere fsaverage/surf/lh.sphere.reg --trgsphere OAS1_0111_MR1/surf/lh.sphere72_my4.reg --trgsurf OAS1_0111_MR1/surf/lh.white --trgsid OAS1_0111_MR1 --outlabel lh.BA1_exvivo_my.label
    
    print("Reading in src sphere: {} ...".format(options.srcsphere))
    src_sphere = fs.read_geometry(options.srcsphere, read_metadata=False)[0]
    print("Reading in trg sphere: {} ...".format(options.trgsphere))
    trg_sphere = fs.read_geometry(options.trgsphere, read_metadata=False)[0]
    # get reverse mapping (trg->src) for sampling
    print("Computing reverse mapping ...")
    rev_mapping,_,_ = getSurfCorrespondence(trg_sphere, src_sphere)
    
    # map label from src to target
    print("Mapping src label to trg ...")
    mapSurfLabel(options.srclabel, options.outlabel, options.trgsurf, options.trgsid, rev_mapping)
    print("Output label {} written".format(options.outlabel))

    print("...done\n")
    
    sys.exit(0)

