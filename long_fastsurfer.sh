#!/bin/bash

# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

###################################################################################################
#
#
#  FastSurfer Longitudinal Processing
# 
#  1. Preapre base inputs (time point co-registration etc)
#  2. Run Base Segmentation (aparcDKT only)
#  3. Run Base Surface creation (skip some steps there using -base flag)
#  4. Run Long Segmentation (can be in parallel with 2 and 3 above)
#  5. Run Long Surface creation (depends on all previous steps)
#
#
###################################################################################################


# Set default values for arguments
if [[ -z "${BASH_SOURCE[0]}" ]]; then
    THIS_SCRIPT="$0"
else
    THIS_SCRIPT="${BASH_SOURCE[0]}"
fi
if [[ -z "$FASTSURFER_HOME" ]]
then
  FASTSURFER_HOME=$(cd "$(dirname "$THIS_SCRIPT")" &> /dev/null && pwd)
  echo "Setting ENV variable FASTSURFER_HOME to script directory ${FASTSURFER_HOME}. "
  echo "Change via environment to location of your choice if this is undesired (export FASTSURFER_HOME=/dir/to/FastSurfer)"
  export FASTSURFER_HOME
fi


# Paths
fastsurfercnndir="$FASTSURFER_HOME/FastSurferCNN"
reconsurfdir="$FASTSURFER_HOME/recon_surf"


# setup variables that are actually passed
tid=""
sd="$SUBJECTS_DIR"
tpids=()
t1s=()
python="python3.10 -s" # avoid user-directory package inclusion


source "${reconsurfdir}/functions.sh"

function usage()
{
cat << EOF

Usage: long_fastsurfer.sh --sid <sid> --sd <sdir> --t1 <t1_input> [OPTIONS]

long_fastsurfer.sh takes a list of T1 full head image and creates:
     (i)   a template subject directory 
     (ii)  directories for each processed time point

FLAGS:

  --tid <templateID>        ID for subject template/base directory inside
                              \$SUBJECTS_DIR to be created"
  --t1s <T1_1> <T1_2> ..    T1 full head inputs for each time point (do not need
                              to be bias corrected). Requires ABSOLUTE paths!
  --tpids <tID1> >tID2> ..  IDs for future time points directories inside
                              \$SUBJECTS_DIR to be created later (during --long)
  --sd  <subjects_dir>      Output directory \$SUBJECTS_DIR (or pass via env var)
  --py <python_cmd>.        Command for python, used in both pipelines.
                              Default: "$python"
                              (-s: do no search for packages in home directory)
  -h --help                 Print Help

EOF
}

# PRINT USAGE if called without params
if [[ $# -eq 0 ]]
then
  usage
  exit
fi

# PARSE Command line
inputargs=("$@")
POSITIONAL=()
i=0
while [[ $# -gt 0 ]]
do
# make key lowercase
key=$(echo "$1" | tr '[:upper:]' '[:lower:]')

shift # past argument
case $key in
  --tid) tid="$1" ; shift ;;
  --tpids)
    while [[ $# -gt 0 ]] && [[ $1 != -* ]] 
    do
      tpids+=("$1")
      shift  # past value
    done
    ;;
  --t1s)
    while [[ $# -gt 0 ]] && [[ $1 != -* ]] 
    do
      t1s+=("$1")
      shift  # past value
    done
    ;;
  --sd) sd="$1" ; export SUBJECTS_DIR="$1" ; shift  ;;
  --py) python="$1" ; shift ;;
  -h|--help) usage ; exit ;;
  *)    # unknown option
    POSITIONAL_FASTSURFER[i]=$KEY
    i=$((i + 1))
    ;;
esac
done




################################### CHECKS ###############################

if [ -z "$t1s" ]
 then
  echo "ERROR: must supply T1 inputs (full head) via --t1s"
  exit 1
fi

if [ -z "$tpids" ]
 then
  echo "ERROR: must supply time points ids via --tpids"
  exit 1
fi

if [ -z "$tid" ]
 then
  echo "ERROR: must supply subject template name via --tid"
  exit 1
fi

# check that t1s list is same length as tpids
if [ "${#tpids[@]}" -ne "${#t1s[@]}" ]
 then
  echo "ERROR: length of tpids must equal t1s"
  exit 1
fi

# check that SUBJECTS_DIR exists
if [[ -z "${sd}" ]]
then
  echo "ERROR: No subject directory defined via --sd. This is required!"
  exit 1;
fi
if [[ ! -d "${sd}" ]]
then
  echo "INFO: The subject directory did not exist, creating it now."
  if ! mkdir -p "$sd" ; then echo "ERROR: directory creation failed" ; exit 1; fi
fi
if [[ "$(stat -c "%u:%g" "$sd")" == "0:0" ]] && [[ "$(id -u)" != "0" ]] && [[ "$(stat -c "%a" "$sd" | tail -c 2)" -lt 6 ]]
then
  echo "ERROR: The subject directory ($sd) is owned by root and is not writable. FastSurfer cannot write results! "
  echo "This can happen if the directory is created by docker. Make sure to create the directory before invoking docker!"
  exit 1;
fi



################################### Prepare Base ###################################

cmd="$reconsurfdir/long_prepare_template.sh \
        --tid $tid --t1s $t1s --tpids $tpids \
        ${POSITIONAL_FASTSURFER[@]}"
RunIt "$cmd" $LF

################################### Run Base Seg ###################################

# t1 for base/template processing:
t1=$sd/$tid/mri/orig.mgz
cmd="$FASTSURFER_HOME/run_fastsurfer.sh \
        --sid $tid --sd $sd --t1 $t1 \
        --seg_only --no_cereb --no_hypothal \
        ${POSITIONAL_FASTSURFER[@]}"
RunIt "$cmd" $LF


################################### Run Base Surf ###################################

cmd="$FASTSURFER_HOME/run_fastsurfer.sh \
        --sid $tid --sd $sd \
        --surf_only --base \
        ${POSITIONAL_FASTSURFER[@]}"
RunIt "$cmd" $LF


################################### Run Long Seg ###################################


# This can run in parallel with base segd and surf steps above

# file name for longitudinal inputs (paths need to be pre-pended later)
extension=".nii.gz"
t1lfn="long_conform${extension}"
for ((i=0;i<${#tpids[@]};++i)); do
  #printf "%s with T1 %s\n" "${tpids[i]}" "${t1s[i]}"
  echo "Seg: ${tpids[i]} with T1 ${t1s[i]}\n"
  mdir="$sd/$tid/long-inputs/${tpids[i]}"
  # segment orig in base space
  cmd="$FASTSURFER_HOME/run_fastsurfer.sh \
        --sid ${tpids[i]} --t1 $mdir/$t1lfn --sd $sd \
        --seg_only --long $tid \
        ${POSITIONAL_FASTSURFER[@]}"
  RunIt "$cmd" $LF
done


################################### Run Long Surf ###################################

for ((i=0;i<${#tpids[@]};++i)); do
  #printf "%s with T1 %s\n" "${tpids[i]}" "${t1s[i]}"
  echo "Surf: ${tpids[i]} with T1 ${t1s[i]}\n"
   # segment orig in base space
  cmd="$FASTSURFER_HOME/run_fastsurfer.sh \
        --sid ${tpids[i]} --sd $sd \
        --surf_only --long $tid \
        ${POSITIONAL_FASTSURFER[@]}"
  RunIt "$cmd" $LF
done


