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
#  1. Prepare longitudinal Person-specific Template (time point co-registration etc)
#  2. Run Person-Template Segmentation (aparcDKT only)
#  3. Run Person-Template Surface creation (skip some steps there using -base flag)
#  4. Run Long Segmentation (can be in parallel with 2 and 3 above)
#  5. Run Long Surface creation (depends on all previous steps)
#
#  Note, that of course 2 and 3, as well as 4 and 5 can be run in a single run_fastsurfer call.
#  Also note, that 4 (long seg) can be run in parallel to the template runs (2 and 3).
#
###################################################################################################


# Set default values for arguments
if [[ -z "${BASH_SOURCE[0]}" ]]; then THIS_SCRIPT="$0"
else THIS_SCRIPT="${BASH_SOURCE[0]}"
fi
if [[ -z "$FASTSURFER_HOME" ]]
then
  FASTSURFER_HOME=$(cd "$(dirname "$THIS_SCRIPT")" &> /dev/null && pwd)
  echo "Setting ENV variable FASTSURFER_HOME to script directory ${FASTSURFER_HOME}. "
  echo "Change via environment to location of your choice if this is undesired (export FASTSURFER_HOME=/dir/to/FastSurfer)"
  export FASTSURFER_HOME
fi


# Paths
reconsurfdir="$FASTSURFER_HOME/recon_surf"


# setup variables that are actually passed
tid=""
sd="$SUBJECTS_DIR"
tpids=()
t1s=()
parallel="false"
LF=""
brun_flags=()
python="python3 -s" # avoid user-directory package inclusion

# Stage control
run_stages=("prepare" "template_seg" "template_surf" "long_seg" "long_surf")
valid_stages=("prepare" "template_seg" "template_surf" "long_seg" "long_surf" "all")
stage_specified="false"  # Track if user has specified any --stage flags


function usage()
{
cat << EOF

Usage: long_fastsurfer.sh --tid <tid> --t1s <T1_1> <T1_2> .. --tpids <tID1> <tID2> .. [OPTIONS]

long_fastsurfer.sh takes a list of T1 full head image and sequentially creates:
     (i)   a template directory for the specific person
     (ii)  directories for each processed time point in template space,
           here you find the final longitudinal results

FLAGS:

  --tid <templateID>        ID for person-specific template directory inside
                              \$SUBJECTS_DIR to be created"
  --t1s <T1_1> <T1_2> ..    T1 full head inputs for each time point (do not need
                              to be bias corrected). Requires ABSOLUTE paths!
  --tpids <tID1> >tID2> ..  IDs for future time points directories inside
                              \$SUBJECTS_DIR to be created later (during --long)
  --sd  <subjects_dir>      Output directory \$SUBJECTS_DIR (or pass via env var)
  --py <python_cmd>         Command for python, used in both pipelines.
                              Default: "$python"
                              (-s: do no search for packages in home directory)
  -h --help                 Print Help

Stage control:
  --stage <stage>           Run specific stage(s). Can be specified multiple times.
                              Valid stages: prepare, template_seg, template_surf,
                                            long_seg, long_surf, all
                              Default: all (runs full pipeline)

                            Stage dependencies:
                              - prepare: none (requires --t1s and --tpids)
                              - template_seg: prepare
                              - template_surf: prepare, template_seg
                              - long_seg: prepare
                              - long_surf: prepare, template_seg, template_surf, long_seg

Parallelization options:
  All of the following options will activate parallel processing of the template and the longitudinal time-point images
  where possible. Additionally, the number of different processes for segmentation and surface reconstructionis set.
  --parallel <n>|max        See above, sets the size of the processing pool for segmentation and surface reconstruction
  --parallel_seg <n>|max    See above, only sets the size of the processing pool for segmentation (default: 1)
  --parallel_surf <n>|max   See above, only sets the size of the processing pool for surface reconstruction (default: 1)


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
  --stage)
    stage=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    if [[ ! " ${valid_stages[*]} " =~ " ${stage} " ]]; then
      echo "ERROR: Invalid stage '$1'. Valid stages: ${valid_stages[*]}"
      exit 1
    fi

    # Handle 'all' as a reset - clear any previous stages and set all stages
    if [[ "$stage" == "all" ]]; then
      run_stages=("prepare" "template_seg" "template_surf" "long_seg" "long_surf")
      stage_specified="true"
    else
      # Clear default stages only on first --stage flag (when stage_specified is "false")
      if [[ "$stage_specified" == "false" ]]; then
        run_stages=()
        stage_specified="true"
      fi
      run_stages+=("$stage")
    fi
    shift
    ;;
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
  --parallel|--parallel_seg|--parallel_surf) parallel="true" ; brun_flags+=("$key" "$1") ; shift ;;
  --py) python="$1" ; shift ;;
  -h|--help) usage ; exit ;;
  --remove_suffix) echo "ERROR: The --remove_suffix option is not supported by long_prepare_template.sh" ; exit 1 ;;
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
  --allow_root|--debug) brun_flags+=("$key") ;;  # --allow_root must be passed to brun
  *)    # unknown option
    POSITIONAL_FASTSURFER[i]=$key
    i=$((i + 1))
    ;;
esac
done


####################################### CHECKS ####################################


source "${reconsurfdir}/functions.sh"

# Function to check if a stage should run
function should_run_stage() {
  local stage=$1
  for s in "${run_stages[@]}"; do
    if [[ "$s" == "$stage" ]]; then return 0; fi
  done
  return 1
}

# Function to validate stage dependencies
function check_stage_dependencies() {
  # Check 'prepare' stage outputs once if any dependent stage needs it and prepare won't run
  if ! should_run_stage "prepare"; then
    # Basic checks: template directory and files from 'prepare'
    if [[ ! -d "$sd/$tid" ]] || [[ ! -f "$sd/$tid/mri/orig.mgz" ]]; then
      echo "ERROR: Stages require 'prepare' to have been run first."
      echo "  Template directory $sd/$tid or $sd/$tid/mri/orig.mgz not found."
      exit 1
    fi
    # Additional check: template must be fully prepared for --base/--long runs
    if [[ ! -f "$sd/$tid/base-tps.fastsurfer" ]]; then
      echo "ERROR: Stages require 'prepare' to have been fully completed."
      echo "  Template indicator file $sd/$tid/base-tps.fastsurfer not found."
      exit 1
    fi
    # Long stages additionally require long-inputs for all time points
    if should_run_stage "long_seg" || should_run_stage "long_surf"; then
      missing_long_inputs=()
      for tpid in "${tpids[@]}"; do
        if [[ ! -f "$sd/$tid/long-inputs/$tpid/long_conform.nii.gz" ]]; then
          missing_long_inputs+=("$tpid")
        fi
      done
      if [[ ${#missing_long_inputs[@]} -gt 0 ]]; then
        echo "ERROR: Long stages require 'prepare' long-inputs for ALL time points."
        echo "  Missing long-inputs for: ${missing_long_inputs[*]}"
        exit 1
      fi
    fi
  fi
  # Here template_seg and long_seg are fine to run
  # Check other stage dependencies
  for stage in "${run_stages[@]}"; do
    case $stage in
      template_surf)
        # Only check for 'template_seg' outputs if 'template_seg' is NOT going to run
        if ! should_run_stage "template_seg"; then
          if [[ ! -f "$sd/$tid/mri/aparc.DKTatlas+aseg.deep.mgz" ]]; then
            echo "ERROR: Stage 'template_surf' requires 'template_seg' to have been run first."
            exit 1
          fi
        fi
        ;;
      long_surf)
        # Only check for 'template_surf' outputs if 'template_surf' is NOT going to run
        if ! should_run_stage "template_surf"; then
          if [[ ! -f "$sd/$tid/surf/lh.white" ]]; then
            echo "ERROR: Stage 'long_surf' requires 'template_surf' to have been run first."
            exit 1
          fi
        fi
        # Only check for 'long_seg' outputs if 'long_seg' is NOT going to run
        if ! should_run_stage "long_seg"; then
          # Check that ALL time points have completed long_seg
          missing_long_seg=()
          for tpid in "${tpids[@]}"; do
            if [[ ! -f "$sd/$tpid/mri/aparc.DKTatlas+aseg.deep.mgz" ]]; then
              missing_long_seg+=("$tpid")
            fi
          done
          if [[ ${#missing_long_seg[@]} -gt 0 ]]; then
            echo "ERROR: Stage 'long_surf' requires 'long_seg' to have been run first for ALL time points."
            echo "  Missing segmentation for: ${missing_long_seg[*]}"
            exit 1
          fi
        fi
        ;;
    esac
  done
}

# Warning if run as root user
check_allow_root "${brun_flags[@]}" # --allow_root must be passed to brun

# Validate required inputs based on which stages will run
if [ -z "$tid" ]
 then
  echo "ERROR: Must supply person-specific template name via --tid <template id>!"
  exit 1
fi

# tpids required for prepare, long_seg, and long_surf stages
if should_run_stage "prepare" || should_run_stage "long_seg" || should_run_stage "long_surf"; then
  if [ "${#tpids[@]}" -lt 1 ]
   then
    echo "ERROR: Must supply time point IDs via --tpids <timepoint id 1> [<timepoint id 2> ...]!"
    exit 1
  fi
fi

# t1s only required for prepare stage
if should_run_stage "prepare"; then
  if [ "${#t1s[@]}" -lt 1 ]
   then
    echo "ERROR: Must supply T1 inputs (full head) via --t1s <t1w file 1> [<t1w file 2> ...] when running 'prepare' stage!"
    exit 1
  fi

  # For prepare stage, t1s and tpids must have same length
  if [ "${#tpids[@]}" -ne "${#t1s[@]}" ]
   then
    echo "ERROR: Length of tpids must equal t1s!"
    exit 1
  fi
fi

# check that SUBJECTS_DIR exists
check_create_subjects_dir_properties "$sd"

if [[ -z "$LF" ]] ; then LF="$sd/$tid/scripts/long_fastsurfer.log" ; fi
# make sure the directory for the logfile exists, create automatically if the directory is not in $sd
if [[ ! -d "$(dirname "$LF")" ]]
then
  if [[ "${LF:0:${#sd}}" == "$sd" ]] ; then mkdir -p "$sd/$tid/scripts"
  else
    echo "ERROR: The directory for the logfile is outside of the SUBJECTS_DIR and did not exist, please"
    echo "  create the directory $(dirname "$LF")!"
    exit 1
  fi
fi
function log () { echo "$1" | tee -a "$LF" ; }

## make sure +eo are unset
set +eo > /dev/null

# Validate stage dependencies
check_stage_dependencies

log "Logging outputs of $THIS_SCRIPT to $LF"
log "======================================="

################################### Prepare Person-Template ##################################

if should_run_stage "prepare"; then
  log "Person-Template Setup $tid"
  cmda=("$reconsurfdir/long_prepare_template.sh"
       --tid "$tid" --t1s "${t1s[@]}" --tpids "${tpids[@]}"
       --py "$python"
       "${POSITIONAL_FASTSURFER[@]}")
  # run_it will exit the bash script if the command fails (with exit code 1)
  run_it "$LF" "${cmda[@]}"
fi

################################### Run Person-Template Seg ##################################

if should_run_stage "template_seg"; then
  log "Person-Template Segmentation $tid"
  cmda=("$FASTSURFER_HOME/run_fastsurfer.sh"
          --sid "$tid" --sd "$sd" --base
          --seg_only --py "$python"
          "${POSITIONAL_FASTSURFER[@]}")
  # run_it will exit the bash script if the command fails (with exit code 1)
  run_it "$LF" "${cmda[@]}"
fi

################################### Run Person-Template Surf #################################

if should_run_stage "template_surf"; then
  log "Person-Template Surface Reconstruction $tid"
  cmda=("$FASTSURFER_HOME/run_fastsurfer.sh"
          --sid "$tid" --sd "$sd"
          --surf_only --base --py "$python"
          "${POSITIONAL_FASTSURFER[@]}")
  # Only background template_surf when long_surf will also run (so it can wait for completion)
  if [[ "$parallel" == "true" ]] && should_run_stage "long_surf"; then
    base_surf_cmdf="$SUBJECTS_DIR/$tid/scripts/base_surf.cmdf"
    base_surf_cmdf_log="$SUBJECTS_DIR/$tid/scripts/base_surf.cmdf.log"
    {
      echo "Log file of person-template surface pipeline"
      date
    } > "$base_surf_cmdf_log"
    echo "#/bin/bash" > "$base_surf_cmdf"
    run_it_cmdf "$LF" "$base_surf_cmdf" "${cmda[@]}"
    log "Starting person-template surface reconstruction, logs temporarily diverted to $base_surf_cmdf_log..."
    log "Output from this process will be delayed to when it has finished."
    log "======================================="
    bash "$base_surf_cmdf" >> "$base_surf_cmdf_log" 2>&1 &
    base_surf_pid=$!
    # shellcheck disable=SC2064
    trap "if [[ -n \"\$(ps --no-headers $base_surf_pid)\" ]] ; then kill $base_surf_pid ; fi" EXIT
  else
    run_it "$LF" "${cmda[@]}"
  fi
fi

################################### Run Long Seg ##################################

# Prepare time_points array for long_seg and long_surf stages
if should_run_stage "long_seg" || should_run_stage "long_surf"; then
  time_points=()
  for ((i=0;i<${#tpids[@]};++i)); do
    time_points+=("${tpids[$i]}=from-base")
  done
fi

if should_run_stage "long_seg"; then
  cmda=("$FASTSURFER_HOME/brun_fastsurfer.sh" --subjects "${time_points[@]}" --sd "$sd" --seg_only --long "$tid"
        "${brun_flags[@]}" "${POSITIONAL_FASTSURFER[@]}")

  # Only background long_seg when long_surf will also run (so it can wait for completion)
  if [[ "$parallel" == "true" ]] && should_run_stage "long_surf"; then
    long_seg_cmdf="$SUBJECTS_DIR/$tid/scripts/long_seg.cmdf"
    long_seg_cmdf_log="$SUBJECTS_DIR/$tid/scripts/long_seg.cmdf.log"
    {
      echo "Log file of longitudinal segmentation pipeline"
      date
    } > "$long_seg_cmdf_log"
    echo "#/bin/bash" > "$long_seg_cmdf"
    run_it_cmdf "$LF" "$long_seg_cmdf" "${cmda[@]}"
    log "Starting longitudinal segmentations, logs temporarily diverted to $long_seg_cmdf_log..."
    log "Output from this process will be delayed to when it has finished."
    log "======================================="
    #TQDM_DISABLE=1
    bash "$long_seg_cmdf" >> "$long_seg_cmdf_log" 2>&1 &
    long_seg_pid=$!
    # shellcheck disable=SC2064
    trap "if [[ -n \"\$(ps --no-headers $long_seg_pid)\" ]] ; then kill $long_seg_pid ; fi" EXIT
  else
    run_it "$LF" "${cmda[@]}"
  fi
fi

################################### Run Long Surf #################################

if should_run_stage "long_surf"; then
  cmda=("$FASTSURFER_HOME/brun_fastsurfer.sh" --subjects "${time_points[@]}" --sd "$sd" --surf_only --long "$tid"
        "${brun_flags[@]}" "${POSITIONAL_FASTSURFER[@]}")
  if [[ "$parallel" == "true" ]] ; then
    # Append the template surface and longitudinal segmentation logs, exit if either failed
    what_failed=()
    log "======================================="

    # Wait for base_surf if it was started
    if [[ -n "${base_surf_pid:-}" ]]; then
      log "Waiting for person-template surface reconstruction to finish... (this may take 30+ minutes)"
      wait "$base_surf_pid"
      success1=$?
      log "done."
      log "Person-Template Surface pipeline Log:"
      log "======================================="
      tee -a "$LF" < "$base_surf_cmdf_log"
      if [ "$success1" -ne 0 ] ; then
        log "Person-Template Surface pipeline terminated with error: $success1"
        what_failed+=("Person-Template Surface Pipeline")
      else
        log "Person-Template Surface pipeline finished successfully!"
        rm "$base_surf_cmdf_log" # the content of this file is transferred to LF
      fi
      log "======================================="
    fi

    # Wait for long_seg if it was started
    if [[ -n "${long_seg_pid:-}" ]]; then
      log "Waiting for longitudinal segmentations to finish..."
      wait "$long_seg_pid"
      success2=$?
      log "Longitudinal Segmentation pipeline Log:"
      log "======================================="
      tee -a "$LF" < "$long_seg_cmdf_log"
      if [ "$success2" -ne 0 ] ; then
        log "Longitudinal Segmentation pipeline terminated with error: $success2"
        what_failed+=("Longitudinal Segmentation Pipeline")
      else
        log "Longitudinal Segmentation pipeline finished successfully!"
        rm "$long_seg_cmdf_log" # the content of this file is transferred to LF
      fi
      log "======================================="
    fi

    if [[ ${#what_failed[@]} -gt 0 ]] ; then
      log "Terminating because ${what_failed[*]} failed!"
      exit 1
    fi
  fi
  run_it "$LF" "${cmda[@]}"
fi

log "======================================="
if should_run_stage "long_surf"; then what="Full"; else what="Stage(s) ${run_stages[*]}"; fi
log "$what longitudinal processing for $tid finished!"
