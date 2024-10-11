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

function usage()
{
  echo "talairach-reg.sh <logfile> --dir <mri-directory> --conformed_name <conformed image file> --norm_name <norm name>"
  echo "                 [--edits] [--long <basedir>] [--3T]"
}

function checkdir()
{
  if [[ ! -d "$1" ]] ; then
    echo "ERROR: Argument ($arg) must be a dir, expected usage:"
    usage
    return 0
  else
    return 1
  fi
}

function checkfile()
{
  if [[ ! -e "$1" ]] ; then
    echo "ERROR: Argument ($key) must be a file, expected usage:"
    usage
    return 0
  else
    return 1
  fi
}

long="false"
edits="false"
atlas3T="false"

LF="$1"
shift
export key=1
if checkfile "$LF" ; then exit 1 ; fi

while [[ $# -gt 0 ]]
do
# make key lowercase
key=$(echo "$1" | tr '[:upper:]' '[:lower:]')
shift # past argument

case $key in
  --dir) if checkdir "$1" ; then exit 1 ; fi ; mdir="$1" ; shift ;;
  --conformed_name) if checkfile "$1" ; then exit 1 ; fi ; conformed_name="$1" ; shift ;;
  --norm_name) if checkfile "$1" ; then exit 1 ; fi ; norm_name="$1" ; shift ;;
  --long) if checkdir "$1" ; then exit 1 ; fi ; long="true" ; basedir="$1" ; shift ;;
  --edits) edits="true" ;;
  --3t) atlas3T="true" ;;
  *) echo "ERROR: Unrecognized argument $key!" ; usage ; exit 1 ;;
esac
done

for arg_spec in "dir=$mdir" "conformed_name=$conformed_name" "norm_name=$norm_name"
do
  arg=$(echo "$arg_spec" | cut -d"=" -f1)
  value=$(echo "$arg_spec" | cut -d"=" -f2)
  if [[ -z "$value" ]] ; then
    echo "ERROR: --$arg is a required argument, but no value was passed!"
    exit 1
  fi
done

if [ -z "$FASTSURFER_HOME" ]
then
  binpath="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/"
else
  binpath="$FASTSURFER_HOME/recon_surf/"
fi

# Load the run_it function
source "$binpath/functions.sh"

# needs <sd>/<sid>/mri
# needs <sd>/<sid>/mri/transforms
mkdir -p "$mdir/transforms"
mkdir -p "$mdir/tmp"


pushd "$mdir" > /dev/null || ( echo "Could not change to $mdir!" | tee -a "$LF" && exit 1)

tal_file="$mdir/transforms/talairach"
if [[ "$edits" == "true" ]] && [[ -f "$tal_file.xfm" ]] && { [[ ! -f "$tal_file.auto.xfm" ]] || \
  [[ -f "$tal_file.auto.xfm" ]] && [[ "$(md5sum "$tal_file.xfm")" != "$(md5sum "$tal_file.auto.xfm")" ]] ; }
then
  # Skip talairach registration (edits is true and we have a file)
  {
    echo "INFO: Skipping talairach registration: $tal_file.xfm exists, because edits is true"
    echo "  and $tal_file.auto.xfm does not exist or is different!"
  } | tee -a "$LF"
elif [[ "$edits" != "true" ]] && [[ -f "$tal_file.xfm" ]]
then
  {
    echo "ERROR: Running talairach registration on top of an existing registration file, but edits is false."
    echo "  Either delete $tal_file.xfm or add the --edits flag."
  } | tee -a "$LF"
  exit 1
else
  if [[ "$edits" == "true" ]] && [[ -f "$tal_file.xfm" ]]
  then
    {
      echo "WARNING: $tal_file.xfm exists and edits is true, but we repeat the talairach"
      echo "  because it seems it has not been modified (no change to $tal_file.auto.xfm)."
      echo "  This will also replace $tal_file.xfm!"
    } | tee -a "$LF"
  fi

  if [[ "$long" == "true" ]] ; then
    # longitudinal processing

    # copy all talairach transforms from base (as we are in same space)
    # this also fixes eTIV across time (if FreeSurfer scaling method is used)
    cmd=(cp "$basedir/mri/transforms/talairach.lta" "$tal_file.lta")
    run_it "$LF" "${cmd[@]}"
    cmd=(cp "$basedir/mri/transforms/talairach.auto.xfm" "$tal_file.auto.xfm")
    run_it "$LF" "${cmd[@]}"
  else
    # regular processing (cross and base)

    if [[ ! -f /bin/tcsh ]] ; then
      echo "ERROR: The talairach_avi script requires tcsh, but /bin/tcsh does not exist!"
      exit 1
    fi

    # talairach.xfm: compute talairach full head (25sec)
    cmd=(talairach_avi --i "$norm_name" --xfm "$mdir/transforms/talairach.auto.xfm")
    if [[ "$atlas3T" == "true" ]]
    then
      echo "INFO: Using the 3T atlas for talairach registration."
      cmd+=(--atlas "3T18yoSchwartzReactN32_as_orig")
    else
      echo "INFO: Using the default atlas (1.5T) for talairach registration."
    fi
    run_it "$LF" "${cmd[@]}"
  fi

  # ALWAYS create copy
  cmd=(cp "$tal_file.auto.xfm" "$tal_file.xfm")
  run_it "$LF" "${cmd[@]}"

  if [[ "$long" == "true" ]] ; then
    cmd=(cp "$basedir/mri/transforms/talairach.xfm.lta" "$tal_file.xfm.lta")
    run_it "$LF" "${cmd[@]}"
  else
    # regular processing (cross and base)

    # talairach.lta: convert to lta
    cmd=(lta_convert --src "$conformed_name" --trg "$FREESURFER_HOME/average/mni305.cor.mgz"
         --inxfm "$tal_file.xfm" --outlta "$tal_file.xfm.lta" --subject fsaverage --ltavox2vox)
    run_it "$LF" "${cmd[@]}"

    # FS would here create better nu.mgz using talairach transform (finds wm and maps it to approx 110)
    #NuIterations="1 --proto-iters 1000 --distance 50"  # default 3T
    #FS60 cmd="mri_nu_correct.mni --i $mdir/orig.mgz --o $mdir/nu.mgz --uchar $mdir/transforms/talairach.xfm --n $NuIterations --mask $mdir/mask.mgz"
    #FS72 cmd="mri_nu_correct.mni --i $mdir/orig.mgz --o $mdir/nu.mgz --uchar $mdir/transforms/talairach.xfm --n $NuIterations --ants-n4"
    # all this is basically useless, as we did a good orig_nu already, including WM normalization
  fi
fi

# Since we do not run mri_em_register we sym-link other talairach transform files here
pushd "$mdir/transforms" > /dev/null || ( echo "ERROR: Could not change to the transforms directory $mdir/transforms!" | tee -a "$LF" ; exit 1 )
  {
    if [[ ! -e talairach_with_skull.lta ]] ; then
      cmd=(softlink_or_copy -sf talairach.xfm.lta talairach_with_skull.lta)
      echo_quoted "${cmd[@]}"
      "${cmd[@]}"
    fi
    if [[ ! -e talairach.lta ]] ; then
      cmd=(softlink_or_copy -sf talairach.xfm.lta talairach.lta)
      echo_quoted "${cmd[@]}"
      "${cmd[@]}"
    fi
  } | tee -a "$LF"
popd > /dev/null || exit 1

# Add xfm to nu
# (use orig_nu, if nu.mgz does not exist already) -- usually nu does not exist and therefore will be created here.
# The difference between orig_nu.mgz and nu.mgz then is the xform-header wrt. the talairach space.
if [[ -e "$mdir/nu.mgz" ]]; then src_nu_file="$mdir/nu.mgz"
else src_nu_file="$norm_name"
fi
cmd=(mri_add_xform_to_header -c "$tal_file.xfm" "$src_nu_file" "$mdir/nu.mgz")
run_it "$LF" "${cmd[@]}"

popd > /dev/null || return