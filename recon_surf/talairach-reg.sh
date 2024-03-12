#!/bin/bash

# Copyright 2024 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

# This script only runs the FreeSurfer talairach registration pipeline
# The call signature is:
usage="talairach-reg.sh <mri-directory> <3T atlas: true/false> <Logfile>"

if [[ "$#" != "3" ]]
then
  echo "Invalid number of arguments to talairach-reg.sh, must be '$usage'"
  exit 1
fi
if ! [[ -d "$1" ]]
then
  echo "First argument must be the mri-directory: $usage"
  exit 1
fi
mdir="$1"
if [[ "$2" != "true" ]] && [[ "$2" != "false" ]]
then
  echo "Second argument must be true or false: $usage"
  exit 1
fi
atlas3T="$2"
if ! [[ -f "$3" ]]
then
  echo "Third argument must be the logfile (must already exist): $usage"
  exit 1
fi
LF="$3"

if [ -z "$FASTSURFER_HOME" ]
then
  binpath="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/"
else
  binpath="$FASTSURFER_HOME/recon_surf/"
fi

# Load the RunIt and the RunBatchJobs functions
source "$binpath/functions.sh"

# needs <sd>/<sid>/mri
# needs <sd>/<sid>/mri/transforms
mkdir -p $mdir/transforms
mkdir -p $mdir/tmp

pushd "$mdir" || ( echo "Could not change to $mdir!" | tee -a "$LF" && exit 1)

# talairach.xfm: compute talairach full head (25sec)
if [[ "$atlas3T" == "true" ]]
then
  echo "Using the 3T atlas for talairach registration."
  atlas="--atlas 3T18yoSchwartzReactN32_as_orig"
else
  echo "Using the default atlas (1.5T) for talairach registration."
  atlas=""
fi
if [[ ! -f /bin/tcsh ]] ; then
  echo "ERROR: The talairach_avi script requires tcsh, but /bin/tcsh does not exist"
  exit 1
fi
cmd="talairach_avi --i $mdir/orig_nu.mgz --xfm $mdir/transforms/talairach.auto.xfm $atlas"
RunIt "$cmd" $LF
# create copy
cmd="cp $mdir/transforms/talairach.auto.xfm $mdir/transforms/talairach.xfm"
RunIt "$cmd" $LF
# talairach.lta: convert to lta
cmd="lta_convert --src $mdir/orig.mgz --trg $FREESURFER_HOME/average/mni305.cor.mgz --inxfm $mdir/transforms/talairach.xfm --outlta $mdir/transforms/talairach.xfm.lta --subject fsaverage --ltavox2vox"
RunIt "$cmd" $LF

# FS would here create better nu.mgz using talairach transform (finds wm and maps it to approx 110)
#NuIterations="1 --proto-iters 1000 --distance 50"  # default 3T
#FS60 cmd="mri_nu_correct.mni --i $mdir/orig.mgz --o $mdir/nu.mgz --uchar $mdir/transforms/talairach.xfm --n $NuIterations --mask $mdir/mask.mgz"
#FS72 cmd="mri_nu_correct.mni --i $mdir/orig.mgz --o $mdir/nu.mgz --uchar $mdir/transforms/talairach.xfm --n $NuIterations --ants-n4"
# all this is basically useless, as we did a good orig_nu already, including WM normalization

# Since we do not run mri_em_register we sym-link other talairach transform files here
pushd $mdir/transforms || ( echo "ERROR: Could not change to the transforms directory $mdir/transforms!" | tee -a "$LF" && exit 1 )
cmd="ln -sf talairach.xfm.lta talairach_with_skull.lta"
RunIt "$cmd" $LF
cmd="ln -sf talairach.xfm.lta talairach.lta"
RunIt "$cmd" $LF
popd || exit 1

# Add xfm to nu
# (use orig_nu, if nu.mgz does not exist already); by default, it should exist
if [[ -e "$mdir/nu.mgz" ]]; then src_nu_file="$mdir/nu.mgz"
else src_nu_file="$mdir/orig_nu.mgz"
fi
cmd="mri_add_xform_to_header -c $mdir/transforms/talairach.xfm $src_nu_file $mdir/nu.mgz"
RunIt "$cmd" $LF

popd || return