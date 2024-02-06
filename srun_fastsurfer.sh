#!/bin/bash

# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

# script for batch-running FastSurfer on a SLURM cluster
# optimized for separate GPU and CPU scheduling.

singularity_image=$HOME/singularity-images/fastsurfer.sif
hpc_work="default"
in_dir=$(pwd)
out_dir=$(pwd)/processed
num_cases_per_task=16
num_cpus_per_task=16
POSITIONAL_FASTSURFER=()
seg_only="false"
surf_only="false"
cpu_only="false"
do_cleanup="true"
debug="false"
submit_jobs="true"
partition=""
partition_surf=""
partition_seg=""
extra_singularity_options=""
extra_singularity_options_surf=""
extra_singularity_options_seg=""
email=""
pattern="*.{nii.gz,nii,mgz}"
subject_list=""
subject_list_awk_code_sid="\$1"
subject_list_awk_code_args="\$2"
subject_list_delim="="
jobarray=""
timelimit_seg=5
timelimit_surf=180

function usage()
{
cat << EOF
Script to orchestrate resource-optimized FastSurfer runs on SLURM clusters.

Usage:
srun_fastsurfer.sh [--data <directory to search images>]
    [--sd <output directory>] [--work <work directory>]
    (--pattern <search pattern for images>|--subject_list <path to subject_list file>
                                           [--subject_list_delim <delimiter>]
                                           [--subject_list_awk_code_sid <subject_id code>]
                                           [--subject_list_awk_code_t1 <image_path code>])
    [--singularity_image <path to fastsurfer singularity image>]
    [--extra_singularity_options [(seg|surf)=]<singularity option string>] [--num_cases_per_task <number>]
    [--num_cpus_per_task <number of cpus to allocate for seg>] [--cpu_only] [--time (surf|seg)=<timelimit>]
    [--partition [(surf|seg)=]<slurm partition>] [--slurm_jobarray <jobarray specification>] [--skip_cleanup]
    [--email <email address>] [--debug] [--dry] [--help]
    [<additional fastsurfer options>]

Author:   David Kügler, david.kuegler@dzne.de
Date:     Nov 3, 2023
Version:  1.0
License:  Apache License, Version 2.0

Documentation of Options:
General options:
--dry: performs all operations, but does not actually submit jobs to slurm
--debug: Additional debug output.
--help: print this help.

Data- and subject-related options:
--sd: output directory will have N+1 subdirectories (default: \$(pwd)/processed):
  - one directory per case plus
  - slurm with two subdirectories:
    - logs for slurm logs,
    - scripts for intermediate slurm scripts for debugging.
  Note: files will be copied here only after all jobs have finished, so most IO happens on
  a work directory, which can use IO-optimized cluster storage (see --work).
--work: directory with fast filesystem on cluster
  (default: \$HPCWORK/fastsurfer-processing/$(date +%Y%m%d-%H%M%S))
  NOTE: THIS SCRIPT considers this directory to be owned by this script and job!
  No modifications should be made to the directory after the job is started until it is
  finished (if the job fails, cleanup of this directory may be necessary) and it should be
  empty!
--data: (root) directory to search in for t1 files (default: current work directory).
--pattern: glob string to find image files in 'data directory' (default: *.{nii,nii.gz,mgz}),
   for example --data /data/ --pattern \*/\*/mri/t1.nii.gz
   will find all images of format /data/<somefolder>/<otherfolder>/mri/t1.nii.gz
--subject_list: alternative way to define cases to process, files are of format:
  subject_id1=/path/to/t1.mgz
  ...
  This option invalidates the --pattern option.
  May also add additional parameters like:
  subject_id1=/path/to/t1.mgz --vox_size 1.0
--subject_list_delim: alternative delimiter in the file (default: "="). For example, if you
  pass --subject_list_delim "," the subject_list file is parsed as a comma-delimited csv file.
--subject_list_awk_code_sid <subject_id code>: alternative way to construct the subject_id
  from the row in the subject_list (default: '\$1').
--subject_list_awk_code_args <t1_path code>: alternative way to construct the image_path and
  additional parameters from the row in the subject_list (default: '\$2'), other examples:
  '\$2/\$1/mri/orig.mgz', where the first field (of the subject_list file) is the subject_id
  and the second field is the containing folder, e.g. the study.
  Example for additional parameters:
  --subject_list_delim "," --subject_list_awk_code_args '\$2 " --vox_size " \$4'
  to implement from the subject_list line
  subject-101,raw/T1w-101A.nii.gz,study-1,0.9
  to (additional arguments must be comma-separated)
  --sid subject-101 --t1 <data-path>/raw/T1w-101A.nii.gz --vox_size 0.9

FastSurfer options:
--fs_license: path to the freesurfer license (either absolute path or relative to pwd)
--seg_only: only run the segmentation pipeline
--surf_only: only run the surface pipeline (--sd must contain previous --seg_only processing)
--***: also standard FastSurfer options can be passed, like --3T, --no_cereb, etc.

Singularity-related options:
--singularity_image: Path to the singularity image to use for segmentation and surface
  reconstruction (default: \$HOME/singularity-images/fastsurfer.sif).
--extra_singularity_options: Extra options for singularity, needs to be double quoted to allow quoted strings,
  e.g. --extra_singularity_options "-B /\$(echo \"/path-to-weights\"):/fastsurfer/checkpoints".
  Supports two formats similar to --partition: --extra_singularity_options <option string> and
  --extra_singularity_options seg=<option string> and --extra_singularity_options surf=<option string>.

SLURM-related options:
--cpu_only: Do not request gpus for segmentation (only affects segmentation, default: request gpus).
--num_cpus_per_task: number of cpus to request for segmentation pipeline of FastSurfer (--seg_only),
  (default: 16).
--num_cases_per_task: number of cases batched into one job (slurm jobarray), will process
  in cases in parallel, id num_cases_per_task is smaller than the total number of cases,
  (default: 16).
--skip_cleanup: Do not schedule step 3, cleanup (which moves the data from --work to --sd, etc.,
  default: do the cleanup).
--slurm_jobarray: a slurm-compatible list of jobs to run, this can be used to rerun segmentation cases
  that have failed, for example '--slurm_jobarray 4,7' would only run the cases associated with a
  (previous) run of srun_fastsurfer.sh, where log files '<sd>/logs/seg_*_{4,7}.log' indicate failure.
--partition: (comma-separated list of) partition(s), supports 2 formats (and their combination):
   --partition seg=<slurm partition>,<other slurm partition>: will schedule the segmentation job on
     listed slurm partitions. It is recommended to select nodes/partitions with GPUs here.
   --partition surf=<partitions>: will schedule surface reconstruction jobs on listed partitions
   --partition <slurm partition>: default partition to used, if specific partition is not given
     (one of the above).
  default: slurm default partition
--time: a per-subject time limit for individual steps, must be number in minutes:
   --time seg=<timelimit>: time limit for the segmentation pipeline (per subject), default: seg=5 (5min).
   --time surf=<timelimit>: time limit for the surface reconstruction (per subject), default: surf=180 (180min)
--email: email address to send slurm status updates.

Accepts additional FastSurfer options, such as --seg_only and --surf_only and only performs the
respective pipeline.
This script will start three slurm jobs:
1. a segmentation job (alternatively, if --surf_only, this copies previous segmentation data from
  the subject directory (--sd)
2. a surface reconstruction job (skipped, if --seg_only)
3. a cleanup job, that moves the data from the work directory (--work) to the subject directory
  (--sd).

Jobs will be grouped into slurm job_arrays with serial segmentation and parallel surface
reconstruction (via job arrays and job steps). This way, segmentation can be scheduled on machines
with GPUs and surface reconstruction on machines without, while efficiently assigning cpus and gpus,
see --partition flag.
Note, that a surface reconstruction job will request up to a total of '<num_cases_per_task> * 2'
cpus and '<num_cases_per_task> * 10G' memory per job. However, these can be distributed across
'<num_cases_per_task>' nodes in parallel job steps.

This tool requires functions in stools.sh and the brun_fastsurfer.sh scripts (expected in same
folder as this script) in addition to the fastsurfer singularity image.
EOF
}

# the memory required for the surface and the segmentation pipeline depends on the
# voxel size of the image, here we use values proven to work for 0.7mm (and also 0.8 and 1m)
mem_seg_cpu=10 # in GB, seg on cpu, actually required: 9G
mem_seg_gpu=7 # in GB, seg on gpu, actually required: 6G
mem_surf_parallel=6 # in GB, hemi in parallel
mem_surf_noparallel=4 # in GB, hemi in series
num_cpus_surf=1 # base number of cpus to use for surfaces (doubled if --parallel)

do_parallel="false"

# PRINT USAGE if called without params
if [[ $# -eq 0 ]]
then
  usage
  exit
fi

if [ -z "${BASH_SOURCE[0]}" ]; then
    THIS_SCRIPT="$0"
else
    THIS_SCRIPT="${BASH_SOURCE[0]}"
fi

set -e

source "$(dirname "$THIS_SCRIPT")/stools.sh"

# PARSE Command line
inputargs=("$@")
while [[ $# -gt 0 ]]
do
# make key lowercase
key=$(echo "$1" | tr '[:upper:]' '[:lower:]')

case $key in
  --fs_license)
    fs_license="$2"
    shift # past argument
    shift # past value
    ;;
  --data)
    in_dir="$2"
    shift
    shift
    ;;
  --sd)
    out_dir="$2"
    shift # past argument
    shift # past value
    ;;
  --pattern)
    pattern="$2"
    shift # past argument
    shift # past value
    ;;
  --subject_list|--subjects_list)
    subject_list="$2"
    shift
    shift
    ;;
  --subject_list_delim|--subjects_list_delim)
    subject_list_delim="$2"
    shift
    shift
    ;;
  --subject_list_awk_code)
    echo "--subject_list_awk_code is outdated, use subject_list_awk_code_sid and subject_list_awk_code_args!"
    exit 1
    ;;
  --subject_list_awk_code_sid|--subjects_list_awk_code_sid)
    subject_list_awk_code_sid="$2"
    shift
    shift
    ;;
  --subject_list_awk_code_args|--subjects_list_awk_code_args)
    subject_list_awk_code_args="$2"
    shift
    shift
    ;;
  --num_cases_per_task)
    num_cases_per_task="$2"
    shift # past argument
    shift # past value
    ;;
  --num_cpus_per_task)
    num_cpus_per_task="$2"
    shift # past argument
    shift # past value
    ;;
  --cpu_only)
    cpu_only="true"
    shift # past argument
    ;;
  --skip_cleanup)
    do_cleanup="false"
    shift # past argument
    ;;
  --work)
    hpc_work="$2"
    shift
    shift
    ;;
  --surf_only)
    surf_only="true"
    shift
    ;;
  --seg_only)
    seg_only="true"
    shift
    ;;
  --parallel)
    do_parallel="true"
    shift
    ;;
  --singularity_image)
    singularity_image="$2"
    shift
    shift
    ;;
  --partition)
    partition_temp="$2"
    # make key lowercase
    lower_value=$(echo "$2" | tr '[:upper:]' '[:lower:]')
    if [[ "$lower_value" =~ seg=* ]]
    then
      partition_seg=${partition_temp:4}
    elif [[ "$lower_value" =~ surf=* ]]
    then
      partition_surf=${partition_temp:5}
    else
      partition=$2
    fi
    shift
    shift
    ;;
  --extra_singularity_options)
    singularity_opts_temp="$2"
    # make key lowercase
    lower_value=$(echo "$2" | tr '[:upper:]' '[:lower:]')
    if [[ "$lower_value" =~ seg=* ]]
    then
      extra_singularity_options_seg=${singularity_opts_temp:4}
    elif [[ "$lower_value" =~ surf=* ]]
    then
      extra_singularity_options_surf=${singularity_opts_temp:5}
    else
      extra_singularity_options=$2
    fi
    shift
    shift
    ;;
  --time)
    time_temp="$2"
    # make key lowercase
    lower_value=$(echo "$2" | tr '[:upper:]' '[:lower:]')
    if [[ "$lower_value" =~ ^seg=[0-9]+ ]]
    then
      timelimit_seg=${time_temp:4}
    elif [[ "$lower_value" =~ surf=([0-9]+|[0-9]{0,1}(:[0-9]{2}){0,1}) ]]
    then
      timelimit_surf=${time_temp:5}
    else
      echo "Invalid parameter to --time: $2, must be seg|surf=<integer (minutes)>"
      exit 1
    fi
    shift
    shift
    ;;
  --email)
    email="$2"
    shift
    shift
    ;;
  --dry)
    submit_jobs="false"
    shift
    ;;
  --debug)
    debug="true"
    shift
    ;;
  --slurm_jobarray)
    jobarray=$2
    shift
    shift
    ;;
  --help)
    usage
    exit
    ;;
  *)    # unknown option
    POSITIONAL_FASTSURFER[$i]=$1
    i=$(($i + 1))
    shift
    ;;
esac
done

# create a temporary logfile, which we copy over to the final log file location once it
# is available
tmpLF=$(mktemp)
LF=$tmpLF

function log() { echo "$@" | tee -a "$LF" ; }
function logf() { printf "$@" | tee -a "$LF" ; }

log "Log of FastSurfer SLURM script"
log $(date -R)
log "$THIS_SCRIPT ${inputargs[*]}"
log ""

make_hpc_work="false"

if [[ -d "$hpc_work" ]]
then
  # delete_hpc_work is true, if the hpc_work directory was created by this script.
  delete_hpc_work="false"
else
  delete_hpc_work="true"
fi
if [[ "$hpc_work" == "default" ]]
then
  if [[ -z "$HPCWORK" ]]
  then
    echo "Neither --work nor \$HPCWORK are defined, make sure to pass --work!"
    exit 1
  else
    # check_hpc_work only has log messages, if it also exists
    check_hpc_work "$HPCWORK/fastsurfer-processing" "false"
    hpc_work_already_exists="true"
    # create a new and unused directory
    while [[ "$hpc_work_already_exists" == "true" ]]; do
      hpc_work=$HPCWORK/fastsurfer-processing/$(date +%Y%m%d-%H%M%S)
      if [[ -a "$hpc_work" ]]; then sleep 1; else hpc_work_already_exists="false"; fi
    done
    make_hpc_work="true"
  fi
else
  # also checks, if hpc_work is also empty (required)
  # check_hpc_work only has log messages, if it also exists
  check_hpc_work "$hpc_work" "true"
fi

if [[ "$debug" == "true" ]]
then
  function debug () { log "$@" ; }
  function debugf () { logf "$@" ; }
else
  # all debug messages go into logfile no matter what, but here, not to the console
  function debug () { echo "$@" >> "$LF" ;  }
  function debugf () { printf "$@" >> "$LF" ;  }
  if [[ "$submit_jobs" == "false" ]]
  then
    log "dry run, no jobs or operations are performed"
    log ""
  fi
fi

debug "Debug parameters to script srun_fastsurfer:"
debug ""
debug "SLURM options:"
debug "submit jobs and perform operations: $submit_jobs"
debug "perform the cleanup step: $do_cleanup"
debug "seg/surf running on slurm partition:" \
  "$(first_non_empty_arg "$partition_seg" "$partition")" "/" \
  "$(first_non_empty_arg "$partition_surf" "$partition")"
debug "num_cpus_per_task/max. num_cases_per_task: $num_cpus_per_task/$num_cases_per_task"
debug "segmentation on cpu only: $cpu_only"
debug "Data options:"
debug "source dir: $in_dir"
if [[ -n "$subject_list" ]]
then
  debug "Reading subjects from subject_list file $subject_list"
  debug "subject_list read options:"
  debug "  delimiter: '${subject_list_delim}'"
  debug "  sid awk code: '${subject_list_awk_code_sid}'"
  debug "  args awk code: '${subject_list_awk_code_args}'"
else
  debug "pattern to search for images: $pattern"
fi
debug "output (subject) dir: $out_dir"
debug "work dir: $hpc_work"
debug ""
debug "FastSurfer parameters:"
debug "singularity image: $singularity_image"
debug "FreeSurfer license: $fs_license"
if [[ "$seg_only" == "true" ]]; then debug "--seg_only"; fi
if [[ "$surf_only" == "true" ]]; then debug "--surf_only"; fi
if [[ "$do_parallel" == "true" ]]; then debug "--parallel"; fi
for p in "${POSITIONAL_FASTSURFER[@]}"
do
  if [[ "$p" == --* ]]; then debugf "\n%s" "$p";
  else debugf " %s" "$p";
  fi
done
debug "Running in$(ls -l /proc/$$/exe | cut -d">" -f2)"
debug ""

if [[ "${pattern/#\/}" != "$pattern" ]]
then
  echo "ERROR: Absolute paths in --pattern are not allowed, set a base path with --data (this may even be /, i.e. root)."
  exit 1
fi

check_singularity_image "$singularity_image"
check_fs_license "$fs_license"
check_seg_surf_only "$seg_only" "$surf_only"
check_out_dir "$out_dir"

if [[ "$cpu_only" == "true" ]] && [[ "$timelimit_seg" -lt 6 ]]
then
  log "WARNING!!!"
  log "------------------------------------------------------------------------"
  log "You specified the segmentation shall be performed on the cpu, but the"
  log "time limit per segmentation is less than 6 minutes (default is optimized "
  log "for GPU acceleration @ 5 minutes). This is very likely insufficient!"
  log "------------------------------------------------------------------------"
fi

# step zero: make directories
if [[ "$submit_jobs" == "true" ]]
then
  if [[ "$make_hpc_work" == "true" ]]; then mkdir "$hpc_work" ; fi
  make_hpc_work_dirs "$hpc_work"
  log "Setting up the work directory..."
fi

wait # for directories to be made

# step one: copy singularity image to hpc
all_cases_file="/$hpc_work/scripts/subject_list"

log "cp \"$singularity_image\" \"$hpc_work/images/fastsurfer.sif\""
script_dir="$(dirname "$THIS_SCRIPT")"
log "cp \"$script_dir/brun_fastsurfer.sh\" \"$script_dir/stools.sh\" \"$hpc_work/scripts\""
log "cp \"$fs_license\" \"$hpc_work/scripts/.fs_license\""
log "Create Status/Success file at $hpc_work/scripts/subject_success"

tofile="cat"
if [[ "$submit_jobs" == "true" ]]
then
  cp "$singularity_image" "$hpc_work/images/fastsurfer.sif" &
  cp "$script_dir/brun_fastsurfer.sh" "$script_dir/stools.sh" "$hpc_work/scripts" &
  cp "$fs_license" "$hpc_work/scripts/.fs_license" &
  log "#Status/Success file of srun_fastsurfer-run $(date)" > "$hpc_work/scripts/subject_success" &

  tofile="tee $all_cases_file"
fi

# step two: copy input data to hpc
if [[ -n "$subject_list" ]]
then
  # the test for files (check_subject_images) requires paths to be wrt
  cases=$(translate_cases "$in_dir" "$subject_list" "$in_dir" "${subject_list_delim}" "${subject_list_awk_code_sid}" "${subject_list_awk_code_args}")
  check_subject_images "$cases"
  if [[ "$debug" == "true" ]]
  then
    log "Debug output of the parsed subject_list:"
    log "$cases"
    log ""
  fi

  cases=$(translate_cases "$in_dir" "$subject_list" "/source" "${subject_list_delim}" "${subject_list_awk_code_sid}" "${subject_list_awk_code_args}" | $tofile)
else
  cases=$(read_cases "$in_dir" "$pattern" "/source" | $tofile)
fi
num_cases=$(echo "$cases" | wc -l)

if [[ "$num_cases" -lt 1 ]] || [[ -z "$cases" ]]
then
  wait
  log "WARNING: No cases found using the parameters provided. Aborting job submission!"
  if [[ "$submit_jobs" == "true" ]] && [[ "$do_cleanup" == "true" ]]
  then
    log "Cleaning temporary work directory!"
    rm -R "$hpc_work/images"
    rm -R "$hpc_work/scripts"
    if [[ "$delete_hpc_work" == "true" ]]
    then
      # delete_hpc_work is true, if the hpc_work directory was created by this script.
      rm -R "$hpc_work"
    fi
  fi
  exit 0
fi

cleanup_mode="mv"
if [[ "$do_cleanup" == "true" ]]
then
  if [[ -n "$jobarray" ]]; then jobarray_defined="true"
  else jobarray_defined="false"
  fi
  check_cases_in_out_dir "$out_dir" "$cases" "$jobarray_defined"
  if [[ "$cleanup_mode" == "cp" ]]
  then
    log "Overwriting existing cases in $out_dir by data generated with FastSurfer."
  fi
fi

if [[ "$submit_jobs" != "true" ]]
then
  log "Copying singularity image and scripts..."
fi
wait
# for copy and other stuff

brun_fastsurfer="scripts/brun_fastsurfer.sh"
fastsurfer_options=()
log_name="slurm-submit"
if [[ "$debug" == "true" ]]
then
  fastsurfer_options=("${fastsurfer_options[@]}" --debug)
fi
cleanup_depend=""
surf_depend=""

slurm_email=()
if [[ -n "$email" ]]
then
  slurm_email=(--mail-user "$email")
  if [[ "$debug" == "true" ]]
  then
    log "Sending emails on ALL conditions"
    slurm_email=("${slurm_email[@]}" --mail-type "ALL,ARRAY_TASKS")
  else
    log "Sending emails on END,FAIL conditions"
    slurm_email=("${slurm_email[@]}" --mail-type "END,FAIL,ARRAY_TASKS")
  fi
fi
jobarray_size="$(($(($num_cases - 1)) / $num_cases_per_task + 1))"
real_num_cases_per_task="$(($(($num_cases - 1)) / $jobarray_size + 1))"
if [[ "$jobarray_size" -gt 1 ]]
then
  if [[ -n "$jobarray" ]]
  then
    jobarray_option=("--array=$jobarray")
  else
    jobarray_option=("--array=1-$jobarray_size")
  fi
  fastsurfer_options=("${fastsurfer_options[@]}" --batch "slurm_task_id/$jobarray_size")
  jobarray_depend="aftercorr"
else
  jobarray_option=()
  jobarray_depend="afterok"
fi

if [[ "$surf_only" != "true" ]]
then
  fastsurfer_seg_options=(# brun_fastsurfer options (inside singularity)
                          --subject_list /data/scripts/subject_list
                          --statusfile /data/scripts/subject_success
                          # run_fastsurfer options (inside singularity)
                          --sd "/data/cases" --threads "$num_cpus_per_task"
                          --seg_only "${POSITIONAL_FASTSURFER[@]}")

  seg_cmd_filename=$hpc_work/scripts/slurm_cmd_seg.sh
  if [[ "$submit_jobs" == "true" ]]
  then
    seg_cmd_file=$seg_cmd_filename
  else
    seg_cmd_file=$(mktemp)
  fi  # END OF NEW

  slurm_partition=$(first_non_empty_partition "$partition_seg" "$partition")
  {
    echo "#!/bin/bash"
    echo "module load singularity"
    echo "singularity exec --nv -B \"$hpc_work:/data,$in_dir:/source:ro\" --no-home \\"
    if [[ -n "$extra_singularity_options" ]] || [[ -n "$extra_singularity_options_seg" ]]; then
      echo "  $extra_singularity_options $extra_singularity_options_seg\\"
    fi
    echo "  $hpc_work/images/fastsurfer.sif \\"
    echo "  /data/$brun_fastsurfer ${fastsurfer_options[*]} ${fastsurfer_seg_options[*]}"
    echo "# discard the exit code of run_fastsurfer (exit with success), so following"
    echo "# jobarray items will be started by slurm under the aftercorr dependency"
    echo "# see https://github.com/Deep-MI/FastSurfer/pull/434#issuecomment-1910805112"
    echo "exit 0"
  } > $seg_cmd_file
  if [[ "$cpu_only" == "true" ]]; then mem_seg="$mem_seg_cpu"
  else mem_seg="$mem_seg_gpu"
  fi
  # note that there can be a decent startup cost for each run, running multiple cases
  # per task significantly reduces this
  seg_slurm_sched=("--mem=${mem_seg}G" "--cpus-per-task=$num_cpus_per_task"
                   --time=$(($timelimit_seg * $real_num_cases_per_task + 5))
                   $slurm_partition "${slurm_email[@]}"
                   "${jobarray_option[@]}" -J "FastSurfer-Seg-$USER"
                   -o "$hpc_work/logs/seg_%A_%a.log" "$seg_cmd_filename")
  if [[ "$cpu_only" == "true" ]]
  then
    debug "Schedule SLURM job without gpu"
  else
    seg_slurm_sched=(--gpus=1 "${seg_slurm_sched[@]}")
  fi
  log "chmod +x $seg_cmd_filename"
  chmod +x $seg_cmd_file
  log "sbatch --parsable ${seg_slurm_sched[*]}"
  echo "--- sbatch script $seg_cmd_filename ---"
  cat $seg_cmd_file
  echo "--- end of script ---"
  if [[ "$submit_jobs" == "true" ]]
  then
    seg_jobid=$(sbatch --parsable ${seg_slurm_sched[*]})
    log "Submitted Segmentation Jobs $seg_jobid"
  else
    log "Not submitting the Segmentation Jobs to slurm (--dry)."
    seg_jobid=SEG_JOB_ID
  fi

  log_name="${log_name}_${seg_jobid}"
  cleanup_depend="afterany:$seg_jobid"
  surf_depend="--depend=$jobarray_depend:$seg_jobid"
elif [[ "$surf_only" == "true" ]]
then
  # do not run segmentation, but copy over all cases from data to work
  copy_jobid=
  make_copy_job "$hpc_work" "$out_dir" "$hpc_work/scripts/subject_list" "$LF" "$submit_jobs"
  if [[ -n "$copy_jobid" ]]
  then
    surf_depend="--depend=afterok:$copy_jobid"
  else
    echo "ERROR: \$copy_jobid not defined!"
    exit 1
  fi
  log_name="${log_name}_${copy_jobid}"
fi

if [[ "$seg_only" != "true" ]]
then
  if [[ "$do_parallel" == "true" ]]
  then
    fastsurfer_surf_options=(--parallel)
  else
    fastsurfer_surf_options=()
  fi
  fastsurfer_surf_options=(# brun_fastsurfer options (outside of singularity)
                           --subject_list "$hpc_work/scripts/subject_list"
                           --parallel_subjects
                           --statusfile "$hpc_work/scripts/subject_success"
                           # run_fastsurfer options (inside singularity)
                           --sd /data/cases --surf_only
                           --fs_license /data/scripts/.fs_license
                           "${fastsurfer_surf_options[@]}"
                           "${POSITIONAL_FASTSURFER[@]}")
  surf_cmd_filename=$hpc_work/scripts/slurm_cmd_surf.sh
  if [[ "$submit_jobs" == "true" ]]; then surf_cmd_file=$surf_cmd_filename
  else surf_cmd_file=$(mktemp)
  fi
  if [[ "$do_parallel" == "true" ]]; then
    cores_per_task=$((num_cpus_surf * 2))
    mem_surf=$mem_surf_parallel
  else
    cores_per_task=$num_cpus_surf
    mem_surf=$mem_surf_noparallel
  fi
  mem_per_core=$((mem_surf / cores_per_task))
  if [[ "$mem_surf" -gt "$((mem_per_core * cores_per_task))" ]]; then
    mem_per_core=$((mem_per_core+1))
  fi
  slurm_partition=$(first_non_empty_partition "$partition_surf" "$partition")
  {
    echo "#!/bin/bash"
    echo "module load singularity"
    echo "run_fastsurfer=(srun -J singularity-surf -o $hpc_work/logs/surf_%A_%a_%s.log"
    echo "                --ntasks=1 --time=$timelimit_surf --nodes=1"
    echo "                --cpus-per-task=$cores_per_task --mem=${mem_surf}G"
    echo "                --hint=nomultithread"
    echo "                singularity exec --no-home -B '$hpc_work:/data'"
    if [[ -n "$extra_singularity_options" ]] || [[ -n "$extra_singularity_options_surf" ]]; then
      echo "                $extra_singularity_options $extra_singularity_options_surf"
    fi
    echo "                '$hpc_work/images/fastsurfer.sif'"
    echo "                /fastsurfer/run_fastsurfer.sh)"
    echo "$hpc_work/$brun_fastsurfer --run_fastsurfer \"\${run_fastsurfer[*]}\" \\"
    echo "  ${fastsurfer_options[*]} ${fastsurfer_surf_options[*]}"
  } > $surf_cmd_file
  surf_slurm_sched=("--mem-per-cpu=${mem_per_core}G" "--cpus-per-task=$cores_per_task"
                    "--ntasks=$real_num_cases_per_task"
                    "--nodes=1-$real_num_cases_per_task" "--hint=nomultithread"
                    "${jobarray_option[@]}" "$surf_depend"
                    -J "FastSurfer-Surf-$USER" -o "$hpc_work/logs/surf_%A_%a.log"
                    $slurm_partition "${slurm_email[@]}" "$surf_cmd_filename")
  chmod +x $surf_cmd_file
  log "sbatch --parsable ${surf_slurm_sched[*]}"
  echo "--- sbatch script $surf_cmd_filename ---"
  cat $surf_cmd_file
  echo "--- end of script ---"
  if [[ "$submit_jobs" == "true" ]]
  then
    surf_jobid=$(sbatch --parsable ${surf_slurm_sched[*]})
    log "Submitted Surface Jobs $surf_jobid"
  else
    log "Not submitting the Surface Jobs to slurm (--dry)."
    surf_jobid=SURF_JOB_ID
  fi

  log_name="${log_name}_${surf_jobid}"
  cleanup_depend="afterany:$surf_jobid"
fi

# step four: copy results back and clean the output directory
if [[ "$do_cleanup" == "true" ]]
then
  # delete_hpc_work is true, if the hpc_work directory was created by this script.
  make_cleanup_job "$hpc_work" "$out_dir" "$cleanup_depend" "$cleanup_mode" "$LF" "$delete_hpc_work" "$submit_jobs"
else
  log "Skipping the cleanup (no cleanup job scheduled, find your results in $hpc_work."
fi


if [[ "$submit_jobs" == "true" ]]
then
  log_dir="$out_dir/slurm/logs"
  mkdir -p "$log_dir"
  cp $tmpLF "$log_dir/$log_name.log"
fi
rm $tmpLF
