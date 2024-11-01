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
#  1. Prepare base inputs (time point co-registration etc)
#  2. Run Base Segmentation (aparcDKT only)
#  3. Run Base Surface creation (skip some steps there using -base flag)
#  4. Run Long Segmentation (can be in parallel with 2 and 3 above)
#  5. Run Long Surface creation (depends on all previous steps)
#
#  Note, that of course 2 and 3, as well as 4 and 5 can be run in a single run_fastsurfer call.
#  Also note, that 4 (long seg) can be run in parallel to the base runs (2 and 3).
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
parallel=0
log=""
python="python3.10 -s" # avoid user-directory package inclusion


function usage()
{
cat << EOF

Usage: long_fastsurfer.sh --tid <tid> --t1s <T1_1> <T1_2> .. --tpids <tID1> <tID2> .. [OPTIONS]

long_fastsurfer.sh takes a list of T1 full head image and sequentially creates:
     (i)   a template subject directory 
     (ii)  directories for each processed time point in template space,
           here you find the final longitudinal results

FLAGS:

  --tid <templateID>        ID for subject template/base directory inside
                              \$SUBJECTS_DIR to be created"
  --t1s <T1_1> <T1_2> ..    T1 full head inputs for each time point (do not need
                              to be bias corrected). Requires ABSOLUTE paths!
  --tpids <tID1> >tID2> ..  IDs for future time points directories inside
                              \$SUBJECTS_DIR to be created later (during --long)
  --sd  <subjects_dir>      Output directory \$SUBJECTS_DIR (or pass via env var)
  --parallel_long           (Highly Experimental) Parallelize the long script
  --py <python_cmd>         Command for python, used in both pipelines.
                              Default: "$python"
                              (-s: do no search for packages in home directory)
  -h --help                 Print Help

With the exception of --t1, --t2, --sid, --seg_only and --surf_only, all
run_fastsurfer.sh options are supported, see 'run_fastsurfer.sh --help'.


REFERENCES:

If you use this for research publications, please cite:

For FastSurfer (both):
Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M, FastSurfer - A
 fast and accurate deep learning based neuroimaging pipeline, NeuroImage 219
 (2020), 117012. https://doi.org/10.1016/j.neuroimage.2020.117012

Henschel L*, Kuegler D*, Reuter M. (*co-first). FastSurferVINN: Building
 Resolution-Independence into Deep Learning Segmentation Methods - A Solution
 for HighRes Brain MRI. NeuroImage 251 (2022), 118933. 
 http://dx.doi.org/10.1016/j.neuroimage.2022.118933

And for longitudinal processing:
Reuter M, Schmansky NJ, Rosas HD, Fischl B. Within-subject template estimation
 for unbiased longitudinal image analysis, NeuroImage 61:4 (2012).
 https://doi.org/10.1016/j.neuroimage.2012.02.084

For cerebellum sub-segmentation:
Faber J*, Kuegler D*, Bahrami E*, et al. (*co-first). CerebNet: A fast and
 reliable deep-learning pipeline for detailed cerebellum sub-segmentation.
 NeuroImage 264 (2022), 119703.
 https://doi.org/10.1016/j.neuroimage.2022.119703

For hypothalamus sub-segemntation:
Estrada S, Kuegler D, Bahrami E, Xu P, Mousa D, Breteler MMB, Aziz NA, Reuter M.
 FastSurfer-HypVINN: Automated sub-segmentation of the hypothalamus and adjacent
 structures on high-resolutional brain MRI. Imaging Neuroscience 2023; 1 1–32.
 https://doi.org/10.1162/imag_a_00034

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
POSITIONAL_FASTSURFER=()
i=0
while [[ $# -gt 0 ]]
do
# make key lowercase
key=$(echo "$1" | tr '[:upper:]' '[:lower:]')

shift # past argument
case $key in
  --tid) tid="$1" ; shift ;;
  --log) LF="$1" ; shift ;;
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
  --parallel_long) parallel=1 ;;
  --py) python="$1" ; shift ;;
  -h|--help) usage ; exit ;;
  --sid|--t1|--t2)
    echo "ERROR: --sid, --t1 and --t2 are not valid for long_fastsurfer.sh, these values are"
    echo "  populated via --tpids, --tid and --t1s, respectively."
    exit 1
    ;;
  --seg_only|--surf_only)
    echo "ERROR: --seg_only and --surf_only are not supported by long_fastsurfer.sh, only a full"
    echo "  pipeline run is a valid longitudinal run!"
    exit 1
    ;;
  *)    # unknown option
    POSITIONAL_FASTSURFER[i]=$key
    i=$((i + 1))
    ;;
esac
done


####################################### CHECKS ####################################


source "${reconsurfdir}/functions.sh"

# Warning if run as root user
check_allow_root

if [ "${#t1s[@]}" -lt 1 ]
 then
  echo "ERROR: Must supply T1 inputs (full head) via --t1s <t1w file 1> [<t1w file 2> ...]!"
  exit 1
fi

if [ "${#tpids[@]}" -lt 1 ]
 then
  echo "ERROR: Must supply time points ids via --tpids <timepoint id 1> [<timepoint id 2> ...]!"
  exit 1
fi

if [ -z "$tid" ]
 then
  echo "ERROR: Must supply subject template name via --tid <template id>!"
  exit 1
fi

# check that t1s list is same length as tpids
if [ "${#tpids[@]}" -ne "${#t1s[@]}" ]
 then
  echo "ERROR: Length of tpids must equal t1s!"
  exit 1
fi

# check that SUBJECTS_DIR exists
if [[ -z "${sd}" ]]
then
  echo "ERROR: No subject directory defined via --sd. This is required!"
  exit 1
elif [[ ! -d "${sd}" ]]
then
  echo "INFO: The subject directory did not exist, creating it now."
  if ! mkdir -p "$sd" ; then echo "ERROR: directory creation failed" ; exit 1; fi
elif [[ "$(stat -c "%u:%g" "$sd")" == "0:0" ]] && [[ "$(id -u)" != "0" ]] && [[ "$(stat -c "%a" "$sd" | tail -c 2)" -lt 6 ]]
then
  echo "ERROR: The subject directory ($sd) is owned by root and is not writable. FastSurfer cannot write results! "
  echo "  This can happen if the directory is created by docker. Make sure to create the directory before invoking docker!"
  exit 1
fi

if [[ -z "$LF" ]]
then
  LF="$sd/$tid/scripts/long_fastsurfer.log"
fi


################################### Prepare Base ##################################

echo "Base Setup $tid"
cmda=("$reconsurfdir/long_prepare_template.sh"
     --tid "$tid" --t1s "${t1s[@]}" --tpids "${tpids[@]}"
     --py "$python"
     "${POSITIONAL_FASTSURFER[@]}")
run_it "$LF" "${cmda[@]}"

################################### Run Base Seg ##################################

echo "Base Seg $tid"
cmda=("$FASTSURFER_HOME/run_fastsurfer.sh"
        --sid "$tid" --sd "$sd" --base
        --seg_only --py "$python"
        "${POSITIONAL_FASTSURFER[@]}")
run_it "$LF" "${cmda[@]}"

################################### Run Base Surf #################################

echo "Base Surf $tid"
cmda=("$FASTSURFER_HOME/run_fastsurfer.sh"
        --sid "$tid" --sd "$sd"
        --surf_only --base --py "$python"
        "${POSITIONAL_FASTSURFER[@]}")
if [[ "$parallel" == "1" ]] ; then
  base_surf_cmdf="$SUBJECTS_DIR/$tid/scripts/base_surf.cmdf"
  base_surf_cmdf_log="$SUBJECTS_DIR/$tid/scripts/base_surf.cmdf.log"
  {
    echo "Log file of base surface pipeline"
    date
  } > "$base_surf_cmdf_log"
  echo "#/bin/bash" > "$base_surf_cmdf"
  run_it_cmdf "$LF" "$base_surf_cmdf" "${cmda[@]}"
  bash "$base_surf_cmdf" 2>&1 >> "$base_surf_cmdf_log" &
  base_surf_pid=$!
  trap "if [[ -n \"\$(ps --no-headers $base_surf_pid)\" ]] ; then kill $base_surf_pid ; fi" EXIT
else
  run_it "$LF" "${cmda[@]}"
fi


################################### Run Long Seg ##################################

# This can run in parallel with base seg and surf steps above
time_points=()
for ((i=0;i<${#tpids[@]};++i)); do
  time_points+=("${tpids[$i]}=from-base")
done
cmda=("$FASTSURFER_HOME/brun_fastsurfer.sh" --subjects "${time_points[@]}" --sd "$sd" --seg_only --long "$tid"
      "${POSITIONAL_FASTSURFER[@]}")

if [[ "$parallel" == "1" ]] ; then
  long_seg_cmdf="$SUBJECTS_DIR/$tid/scripts/long_seg.cmdf"
  long_seg_cmdf_log="$SUBJECTS_DIR/$tid/scripts/long_seg.cmdf.log"
  {
    echo "Log file of longitudinal segmentation pipeline"
    date
  } > "$long_seg_cmdf_log"
  echo "#/bin/bash" > "$long_seg_cmdf"
  run_it_cmdf "$LF" "$long_seg_cmdf" "${cmda[@]}"
  # at the end of the job below, the gpu can be released (for tight management of resources, run
  # Surfaces in different jobs. Alternative, add a command to "$long_seg_cmdf" that releases the gpu or
  # triggers the next "subject"
  #TQDM_DISABLE=1
  bash "$long_seg_cmdf" 2>&1 >> "$long_seg_cmdf_log" &
  long_seg_pid=$!
  trap "if [[ -n \"\$(ps --no-headers $long_seg_pid)\" ]] ; then kill $long_seg_pid ; fi" EXIT
else
  run_it "$LF" "${cmda[@]}"
fi

################################### Run Long Surf #################################

cmda=("$FASTSURFER_HOME/brun_fastsurfer.sh" --subjects "${time_points[@]}" --sd "$sd" --surf_only --long "$tid"
      "${POSITIONAL_FASTSURFER[@]}")
if [[ "$parallel" == "1" ]] ; then
  cmda+=("--parallel_subjects")

  what_failed=()
  wait $base_surf_pid
  success1=$?
  {
    echo "Base Surface pipeline Log:"
    echo "======================================="
    cat "$base_surf_cmdf_log"
    if [ "$success1" -ne 0 ] ; then
      echo "Base Surface pipeline terminated with error: $success1"
      what_failed+=("Base Surface Pipeline")
    else
      echo "Base Surface pipeline finished successful!"
      rm "$base_surf_cmdf_log" # the content of this file is transferred to LF
    fi
    echo "======================================="
  } | tee -a "$LF"
  wait $long_seg_pid
  success2=$?
  {
    echo "Longitudinal Segmentation pipeline Log:"
    echo "======================================="
    cat "$long_seg_cmdf_log"
    if [ "$success2" -ne 0 ] ; then
      echo "Longitudinal Segmentation pipeline terminated with error: $success2"
      what_failed+=("Longitudinal Segmentation Pipeline")
    else
      echo "Longitudinal Segmentation pipeline finished successful!"
      rm "$long_seg_cmdf_log" # the content of this file is transferred to LF
    fi
    echo "======================================="
  if [[ "$success1" -ne 0 ]] || [[ "$success2" -ne 0 ]] ; then
    echo "Terminating because ${what_failed[*]} failed!"
    exit 1
  fi
  } | tee -a "$LF"
fi

run_it "$LF" "${cmda[@]}"
