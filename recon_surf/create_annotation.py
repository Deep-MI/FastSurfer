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
import os.path
import numpy as np
import sys
import nibabel.freesurfer.io as fs
from map_surf_label import mapSurfLabel, getSurfCorrespondence


HELPTEXT = """

Script to map and combine a set of labels into an annotation file

create_annotation.py --hemi <lh,rh> --colortab <table.txt> --labeldir <ldir>
                     --white <hemi.white> --outannot <hemi.annot>
                     --cortex <hemi.cortex.label> --append <extension>

when mapping labels use these additional flags:

                     --srcsphere <hemi.shpere.reg> --trgsphere <hemi.sphere.reg>
                     --trgdir <label-dir> --trgsid <sid>


Dependencies:
    Python 3.8
    numpy, nibabel, sklearn
    

Description:
This script reads a set of labels as listed in the color table file (and hemi) from the
label directory and creates a FreeSurfer annotation file usually on the white surface of
that subject, optionally cropping to the cortex region if a cortex label is passed. 
It therefore is similar to FreeSurfer's mris_label2annot.

Additionally it is possible to take the labels from another subject (usually fsaverage) 
and map those to the current subject. For this you need to pass src and trg sphere.reg 
files as well as the src label directory. If the mapped labels should be output, also
pass the trgdir where to write the labels and trgsid, else it only creates the
annotation after mapping and discards the individual labels. In this case --labeldir is 
understood to be the label directory in the source subject. 

Note, the ordering of the labels in the colortable is relevant as later labels can 
overwrite earlier ones (if their label value is equal, else the one with the higher value
will prevail).

Also note, all filenames require full (or relative) path including hemi and filename, e.g.
--outannot $SUBJECTS_DIR/subjectid/surf/lh.BA_exvivo.annot


Original Author: Martin Reuter
Date: Aug-24-2022
"""

h_hemi       = '"lh" or "rh" for reading labels'
h_colortab   = 'colortab with label ids, names and colors'
h_labeldir   = 'dir where to find the label files (when reading)'
h_white      = 'path/filename of white surface for the annotation'
h_cortex     = 'optional path to hemi.cortex for optional masking of annotation to only cortex'
h_outannot   = 'path to output annotation file'
h_append     = 'optional, e.g. ".thresh" can be appended to label names (I/O) for exvivo FS labels'
# when mapping labels additionally
h_srcsphere  = 'optional, when mapping: path to src sphere.reg'
h_trgsphere  = 'optional, when mapping: path to trg sphere.reg'
h_trgdir     = 'optional: directory where to write mapped label files'
h_trgsid     = 'optional, when storing mapped labels: target subject id, also written into label file'


def options_parse():
    """
    Command line option parser
    """
    parser = optparse.OptionParser(version='$Id:create_annotation.py,v 1.0 2022/08/24 21:22:08 mreuter Exp $', usage=HELPTEXT)
    parser.add_option('--hemi',      dest='hemi',      help=h_hemi)
    parser.add_option('--colortab',  dest='colortab',  help=h_colortab)
    parser.add_option('--labeldir',  dest='labeldir',  help=h_labeldir)
    parser.add_option('--white',     dest='white',     help=h_white)
    parser.add_option('--cortex',    dest='cortex',    help=h_cortex)
    parser.add_option('--append',    dest='append',    help=h_append)
    parser.add_option('--outannot',  dest='outannot',  help=h_outannot)
    parser.add_option('--srcsphere', dest='srcsphere', help=h_srcsphere)
    parser.add_option('--trgsphere', dest='trgsphere', help=h_trgsphere)
    parser.add_option('--trgdir',    dest='trgdir',    help=h_trgdir)
    parser.add_option('--trgsid',    dest='trgsid',    help=h_trgsid)
    (options, args) = parser.parse_args()
    if options.hemi is None or options.colortab is None or options.labeldir is None \
       or options.white is None or options.outannot is None:
        sys.exit('\nERROR: Please specify all parameters!\n   Use --help to see all options.\n')
    if options.trgsphere is not None or options.srcsphere is not None or options.trgdir is not None \
       or options.trgsid is not None:
        if options.trgsphere is None or options.srcsphere is None:
            sys.exit('\nERROR: Please specify at least src and trg sphere when mapping!\n   Use --help to see all options.\n')
    if (options.trgdir is not None and options.trgsid is None) or (options.trgdir is None and options.trgsid is not None):
        sys.exit('\nERROR: Please specify both trgdir and trgsid when outputting mapped labels!\n   Use --help to see all options.\n')

    return options



def map_multiple_labels(hemi, src_dir, src_labels, src_sphere_name, 
                        trg_sphere_name, trg_white_name, trg_sid, 
                        out_dir = None, stop_missing = True):
# function to map a list of labels (just names without hemisphere or path, which are 
#  passed via hemi, src_dir, out_dir) from one surface (e.g. fsavaerage sphere.reg)
#  to another. 
#  stop_missing determines whether to stop on a missing src label file, or continue
#  with a warning. 
#  All mapped labels and their values are returned
#  Also mapped label files are written to the out_dir if specified
    # get reverse mapping (trg->src) for sampling
    rev_mapping,_,_ = getSurfCorrespondence(trg_sphere_name, src_sphere_name)
    all_labels = []
    all_values = []
    # read target surf info (for label writing)
    print("Reading in trg white surface: {} ...".format(trg_white_name))
    trg_white = fs.read_geometry(trg_white_name, read_metadata=False)[0]
    out_label_name = None
    for l_name in src_labels:
        if l_name == "unknown":
            print("unknown label: skipping ...")
            continue
        src_label_name = os.path.join(src_dir,hemi+"."+l_name+".label")
        if out_dir is not None:
            out_label_name = os.path.join(out_dir,hemi+"."+l_name+".label")
        # map label from src to target
        if os.path.exists(src_label_name):
            #print("Mapping label {}.{} ...".format(hemi,l_name))
            l,v = mapSurfLabel(src_label_name, out_label_name, trg_white, trg_sid, rev_mapping)
        else:
            if stop_missing:
                raise ValueError("ERROR: Label file missing {}\n".format(src_label_name))
            else:
                print("\nWARNING: Label file missing {}\n".format(src_label_name))
                l=[]
                v=[]
        all_labels.append(l)
        all_values.append(v)
    return all_labels, all_values


def read_multiple_labels(hemi, input_dir, label_names):
# read multiple label files from input_dir
    all_labels = []
    all_values = []
    for l_name in label_names:
        label_file = os.path.join(input_dir,hemi+"."+l_name+".label")
        if os.path.exists(label_file):
            l, v = fs.read_label(label_file, read_scalars=True)
        else:
            print("\nWARNING: Label file missing {}\n".format(label_file))
            l=[]
            v=[]
        all_labels.append(l)
        all_values.append(v)
    return all_labels, all_values
  


def build_annot(all_labels, all_values, col_ids, trg_white, cortex_label_name=None):
# function to create an annotation from multiple labels. Here we also consider the
# label values and overwrite existing labels if values of current are larger (or equal,
# so the order of the labels matters). 
# annot_ids and values are returned, no output is written.
#
    # create annot from a bunch of labels (and values)
    if isinstance(trg_white, str):
        trg_white= fs.read_geometry(trg_white, read_metadata=False)[0]
    annot_ids  = np.zeros(trg_white.shape[0],dtype="i8")
    annot_vals = np.zeros(trg_white.shape[0])
    counter=0
    #print(col_ids)
    #offset =1 # start with id=1 (as zero is unknown)
    for label in all_labels:
        #print("counter={}".format(counter))
        if len(label) == 0:
            print("\nWARNING: Label with id {} missing, skipping ...\n".format(col_ids[counter]))
            counter=counter+1
            continue
        vals = np.squeeze(all_values[counter])
        label = np.squeeze(label)
        mask = (vals >= annot_vals[label])
        label_masked = label[mask]
        vals_masked = vals[mask]
        #annot_ids[label_masked] = counter + offset
        annot_ids[label_masked] = col_ids[counter]
        annot_vals[label_masked] = vals_masked
        counter=counter+1
    # mask non-cortex if cortex label is passed (mask value is -1)
    if cortex_label_name is not None:
        cortex_label = fs.read_label(cortex_label_name, read_scalars=False)
        mask = np.ones(trg_white.shape[0], dtype=bool)
        mask[cortex_label] = False
        annot_ids[mask] = -1 
    return annot_ids, annot_vals


def read_colortable(colortab_name):
    colortab = np.genfromtxt(colortab_name, dtype="i8",usecols=(0,2,3,4,5))
    ids = colortab[:,0]
    colors = colortab[:,1:]
    names = np.genfromtxt(colortab_name, dtype="S30",usecols=(1))
    names = [x.decode() for x in names]
    return ids, names, colors


def write_annot(annot_ids, label_names, colortab_name, out_annot, append=""):
# This function combines the colortable with the annotations ids to 
# write an annotation file (which contains colortable information)
# Care needs to be taken that the colortable file has the same number
# and order of labels as specified in the label_names list
#
    #colortab_name="colortable_BA.txt"
    col_ids, col_names, col_colors = read_colortable(colortab_name)
    offset = 0
    if col_names[0] == "unknown":
        offset = 1
    for name_tab, name_list in zip(col_names[offset:], label_names):
        if name_tab+append != name_list:
            #print("Name in colortable and in label lists disagree: {} != {}".format(name_tab+append,name_list))
            raise ValueError("Error: name in colortable and in label lists disagree: {} != {}".format(name_tab+append,name_list))
    # fill_ctab computes the last column (R+G*2^8+B*2^16)
    if offset==0:
       # no unknown as 0 label in color table, we need to add it
       col_names = np.concatenate((['unknown'],col_names))
       col_colors = np.vstack([[25,5,25,0],col_colors])
    fs.write_annot(out_annot, annot_ids, col_colors, col_names, fill_ctab=True)

def create_annotation(options, verbose=True):
# main function to map (if required), build  and write annotation
    print()
    print("Map BA Labels Parameters:")
    print()
    if verbose:
        print("- hemi: {}".format(options.hemi))
        print("- color table: {}".format(options.colortab))
        print("- label dir: {}".format(options.labeldir))
        print("- white: {}".format(options.white))
        print("- out annot: {}".format(options.outannot))
        if options.cortex is not None:
            print("- cortex mask: {}".format(options.cortex))
        if options.append is not None:
            if options.append[0] != '.':
                options.append = '.'+options.append
            print("- append {} to label names".format(options.append))
        if options.trgsphere is not None:
            print("Mapping labels from another subject:")
            print("- src sphere: {}".format(options.srcsphere))
            print("- trg sphere: {}".format(options.trgsphere))
            if options.trgdir is not None:
                print("And will write mapped labels:")
                print("- trg dir: {}".format(options.trgdir))
                print("- trg sid: {}".format(options.trgsid))
        print()
    # read label names from color table
    print("Reading in colortable: {} ...".format(options.colortab))
    ids, names, cols = read_colortable(options.colortab)
    if (names[0]=="unknown"):
        ids = ids[1:]
        names = names[1:]
        cols = cols[1:] # although we do not care about color at this stage at all
    if options.append is not None:
            names = [x+options.append for x in names]
    print("Merging these labels into annot:\n{}\n".format(names))
    # if reading multiple label files
    if options.trgsphere is None:
        print("Reading multiple labels from {} ...".format(options.labeldir))
        all_labels, all_values = read_multiple_labels(options.hemi, options.labeldir, names)
    else:
    # if mapping multiple label files
        print("Mapping multiple labels from {} to {} ...".format(options.labeldir, options.trgdir))
        all_labels, all_values = map_multiple_labels(options.hemi, options.labeldir, 
                        names, options.srcsphere, options.trgsphere,
                        options.white, options.trgsid, options.trgdir)
    # merge labels into annot
    print("Creating annotation on {}".format(options.white))
    annot_ids, annot_vals = build_annot(all_labels, all_values, ids, options.white, options.cortex)
    # write annot
    print("Writing annotation to {}".format(options.outannot))
    write_annot(annot_ids, names, options.colortab, options.outannot, options.append)
    print("...done\n")


if __name__ == "__main__":

    # Command line options and error checking done here
    options = options_parse()

    # for example:

    #./create_annotation.py --hemi lh \
    #   --colortab colortable_BA.txt \
    #   --labeldir fsaverage/label \
    #   --white OAS1_0111_MR1/surf/lh.white \
    #   --outannot lh.test.annot \
    #   --cortex OAS1_0111_MR1/label/lh.cortex.label \
    #   --srcsphere fsaverage/surf/lh.sphere.reg \
    #   --trgsphere OAS1_0111_MR1/surf/lh.sphere.reg \
    #   --trgdir test --trgsid OAS1_0111_MR1
    
    #./create_annotation.py --hemi lh \
    #   --colortab colortable_BA.txt \
    #   --labeldir OAS1_0111_MR1/label \
    #   --white OAS1_0111_MR1/surf/lh.white \
    #   --outannot lh.test2.annot \
    #   --cortex OAS1_0111_MR1/label/lh.cortex.label \

    #./create_annotation.py --hemi lh \
    #   --colortab $FREESURFER_HOME/average/colortable_vpnl.txt \
    #   --labeldir fsaverage/label  \
    #   --white OAS1_0111_MR1/surf/lh.white \
    #   --outannot lh.vpnl.annot \
    #   --cortex OAS1_0111_MR1/label/lh.cortex.label \
    #   --srcsphere $FREESURFER_HOME/subjects/fsaverage/surf/lh.sphere.reg \
    #   --trgsphere OAS1_0111_MR1/surf/lh.sphere.reg \
    #   --trgdir test --trgsid OAS1_0111_MR1


    create_annotation(options)
    
    sys.exit(0)


