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
import os
import numpy as np
import sys
import nibabel.freesurfer.io as fs
from create_annotation import map_multiple_labels, read_colortable, build_annot, write_annot

HELPTEXT = """

Script to replicate recon-all --balabels step, to map and merge BA exvivo labels and
Grill-Spector labels into annotations and produce statistics on some.

fs_balabels.py --sid <subject id> --sd <subjects_dir> 

Optional flags:
               --fsaverage <fsaverage dir> --hemi <lh or rh>

Dependencies:
    Python 3.8
    numpy, nibabel, sklearn
    
    Also FreeSurfer v7.3.2 is needed
    
Description:
This script replaces recon-all --balabels with mostly our own python based mapping 
scripts. These are much faster as the surface correspondence is computed one time and
used when mapping all labels from fsaverage to the current case. Precisely, for each
hemisphere it will:
1. map BA_exvivo labels
2. map BA_exvivo.thresh labels
3. map Grill-Spector labels 
4. create annotations for each of those 
5. compute surface stats files for the BA_exvivo labels (currently still using
   FreeSurfer's mris_anatomical_stats via command line execution)
Note, this script needs to be updated if FreeSurfer introduces changes into the -balabel 
block of recon-all. Currently this is based on FreeSurfer 7.3.2.

Original Author: Martin Reuter
Date: Aug-31-2022
"""

h_sid        = 'subject id (name of directory within the subject directory)'
h_sd         = 'subject directory path'
h_hemi       = 'optioal: "lh" or "rh" (default run both hemispheres)'
h_fsaverage  = 'optional: path to fsaverage (default is $FREESURFER_HOME/subjects/fsaverage)'


def options_parse():
    """
    Command line option parser
    """
    parser = optparse.OptionParser(version='$Id:fs_balabels.py,v 1.0 2022/08/24 21:22:08 mreuter Exp $', usage=HELPTEXT)
    parser.add_option('--sid',       dest='sid',       help=h_sid)
    parser.add_option('--sd',        dest='sd',        help=h_sd)
    parser.add_option('--hemi',      dest='hemi',      help=h_hemi)
    parser.add_option('--fsaverage', dest='fsaverage', help=h_fsaverage)
    
    (options, args) = parser.parse_args()

    if options.sid is None or options.sd is None:
        sys.exit('\nERROR: Please specify --sid and --sd !\n   Use --help to see all options.\n')
    if options.hemi is None:
        options.hemi = ["lh","rh"]
    else:
        options.hemi = [options.hemi]
    
    return options


def read_colortables(colnames,colappend,drop_unknown=True):
# reads multiple colortables and appends extensions, 
# drops unknown by default
    pos=0
    all_names = []
    all_ids = []
    all_cols = []
    for coltab in colnames:
        print("Reading in colortable: {} ...".format(coltab))
        ids, names, cols = read_colortable(coltab)
        if drop_unknown and names[0]=="unknown":
            ids = ids[1:]
            names = names[1:]
            cols = cols[1:]
        if colappend[pos]:
            names = [x+colappend[pos] for x in names]
        all_names.append(names)
        all_ids.append(ids)
        all_cols.append(cols)
        pos=pos+1
    return all_ids, all_names, all_cols

if __name__ == "__main__":

    stream = os.popen('date')
    output = stream.read()

    print
    print("#--------------------------------------------")
    print("#@# BA_exvivo Labels "+output)
    print

    # Command line options and error checking done here
    options = options_parse()

    fshome = os.environ.get('FREESURFER_HOME')
    if not fshome:
        sys.exit('\nERROR: FREESURFER_HOME environment variable needs to be specified.\n')
    sdir = os.environ.get('SUBJECTS_DIR')
    if not sdir:
        os.environ['SUBJECTS_DIR'] = options.sd
    else:
        if sdir != options.sd:
            print("WARNING environment $SUBJECTS_DIR is set differently to --sd !")
            os.environ['SUBJECTS_DIR'] = options.sd
    if options.fsaverage is None:
        options.fsaverage=os.path.join(fshome,"subjects","fsaverage")


    # read and stack colortable labels
    ba   = os.path.join(fshome,"average","colortable_BA.txt")
    vpnl = os.path.join(fshome,"average","colortable_vpnl.txt")
    colnames = [ ba, ba, vpnl ]
    colappend = ["",".thresh",".mpm.vpnl"]
    annotnames = [ "BA_exvivo", "BA_exvivo.thresh", "mpm.vpnl"]    
    label_ids, label_names, label_cols = read_colortables(colnames, colappend)

    labeldir = os.path.join(options.fsaverage,"label")
    trgdir = os.path.join(options.sd,options.sid,"label")
    for hemi in options.hemi:
        # map all labels for this hemisphere in one step (one registration)
        srcsphere=os.path.join(options.fsaverage,"surf",hemi+".sphere.reg")
        trgsphere=os.path.join(options.sd,options.sid,"surf",hemi+".sphere.reg")
        white=os.path.join(options.sd,options.sid,"surf",hemi+".white")
        cortex=os.path.join(options.sd,options.sid,"label",hemi+".cortex.label")
        print("Mapping multiple labels from {} to {} for {} ...\n".format(labeldir, trgdir, hemi))
        all_labels, all_values = map_multiple_labels(hemi, labeldir, 
                                 np.concatenate(label_names), srcsphere, trgsphere,
                                 white, options.sid, trgdir)
        # merge labels into annot
        pos = 0  # 0,1,2
        start = 0 # to corresponding blocks from all_labels and all_values
        #print("Debug length all labels: {}".format(len(all_labels)))
        for annot in annotnames:
            #print("Debug length labelids pos {}".format(len(label_ids[pos])))
            stop = start + len(label_ids[pos])
            print("\nCreating {} annotation on {}".format(annot,white))
            #print("Debug info start: {}, stop: {}".format(start,stop))
            annot_ids, annot_vals = build_annot(all_labels[start:stop], all_values[start:stop], label_ids[pos], white, cortex)
            # write annot
            print("Writing BA_exvivo annotation to {}\n".format(annot))
            annotout = os.path.join(options.sd,options.sid,"label",hemi+"."+annot+".annot")
            write_annot(annot_ids, label_names[pos], colnames[pos], annotout, colappend[pos])
            # now call annatomical stats from comand line (onyl for BA_exvivo)
            if pos < 2:
                print("Computing "+hemi+"."+annot+".stats ...")
                stats = os.path.join(options.sd,options.sid,"stats",hemi+"."+annot+".stats")
                ctab  = os.path.join(options.sd,options.sid,"label",annot+".ctab")
                cmd = "mris_anatomical_stats -mgz -f {} -b -a {} -c {} \
                       {} {} white".format(stats,annotout,ctab,options.sid,hemi)
                print("Debug cmd: "+cmd)
                stream = os.popen(cmd)
                print(stream.read())
            start = stop
            pos=pos+1
    
    print("...done\n")

