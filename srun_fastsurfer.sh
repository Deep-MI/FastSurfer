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
email=""
pattern="*.{nii.gz,nii,mgz}"
subject_list=""
subject_list_awk_code="\$1:\$2"
subject_list_delim=","
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
                                           [--subject_list_awk_code <subject_id code>:<subject_path code>])
    [--singularity_image <path to fastsurfer singularity image>] [--num_cases_per_task <number>]
    [--num_cpus_per_task <number of cpus to allocate for seg>] [--cpu_only] [--time (surf|seg)=<timelimit>]
    [--partition [(surf|seg)=]<slurm partition>] [--slurm_jobarray <jobarray specification>] [--skip_cleanup]
    [--email <email address>] [--debug] [--dry] [--help]
    [<additional fastsurfer options>]

Author:   David Kügler, david.kuegler@dzne.de
Date:     Nov 3, 2023
Version:  1.0
License:  Apache License, Version 2.0

Documentation of Options:
--sd: output directory will have N+1 subdirectories (default: \$(pwd)/processed):
  - one directory per case plus
  - slurm with two subdirectories:
    - logs for slurm logs,
    - scripts for intermediate slurm scripts for debugging.
  Note: files will be copied here only after all jobs have finished, so most IO happens on
  a work directory, which can use IO-optimized cluster storage (see --work).
--dry: performs all operations, but does not actually submit jobs to slurm
--data: (root) directory to search in for t1 files (default: current work directory).
--work: directory with fast filesystem on cluster
  (default: \$HPCWORK/fastsurfer-processing/$(date +%Y%m%d-%H%M%S))
--fs_license: path to the freesurfer license
--pattern: glob string to find image files in 'data directory' (default: *.{nii,nii.gz,mgz}),
   for example --data /data/ --pattern \*/\*/mri/t1.nii.gz
   will find all images of format /data/<somefolder>/<otherfolder>/mri/t1.nii.gz
--subject_list: alternative way to define cases to process, a csv file: comma-delimited.
  column 1: subject id, column 2: path to input image.
  This option invalidates the --pattern option.
--subject_list_delim: alternative delimiter in the file (default: ",").
--subject_list_awk_code <subject_id code>:<subject_path code>: alternative way to construct
  subject_id and subject_path from the row in the subject_list (default: '\$1:\$2'), other
  examples: '\$1:\$2/\$1/mri/orig.mgz',

--singularity_image: Path to the singularity image to use for segmentation and surface
  reconstruction (default: \$HOME/singularity-images/fastsurfer.sif).

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
--debug: Additional debug output.
--help: print this help.

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
mem_surf_parallel=20 # in GB, hemi in parallel
mem_surf_noparallel=18 # in GB, hemi in series
num_cpus_surf=2 # base number of cpus to use for surfaces (doubled if --parallel)

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
  --subject_list)
    subject_list="$2"
    shift
    shift
    ;;
  --subject_list_delim)
    subject_list_delim="$2"
    shift
    shift
    ;;
  --subject_list_awk_code)
    subject_list_awk_code="$2"
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

echo "Log of FastSurfer SLURM script"
date -R
echo "$THIS_SCRIPT ${inputargs[*]}"
echo ""

make_hpc_work="false"

if [[ -d "$hpc_work" ]]
then
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
  check_hpc_work "$hpc_work" "true"
fi

if [[ "$debug" == "true" ]]
then
  echo "Debug parameters to script srun_fastsurfer:"
  echo ""
  echo "SLURM options:"
  echo "submit jobs and perform operations: $submit_jobs"
  echo "perform the cleanup step: $do_cleanup"
  echo "seg/surf running on slurm partition:" \
    "$(first_non_empty_partition2 "$partition_seg" "$partition")" "/" \
    "$(first_non_empty_partition2 "$partition_surf" "$partition")"
  echo "num_cpus_per_task/max. num_cases_per_task: $num_cpus_per_task/$num_cases_per_task"
  echo "segmentation on cpu only: $cpu_only"
  echo "Data options:"
  echo "source dir: $in_dir"
  if [[ -n "$subject_list" ]]
  then
    echo "Reading subjects from subject_list file $subject_list"
    echo "subject_list read options: delimiter: '${subject_list_delim}', awk code: '${subject_list_awk_code}'"
  else
    echo "pattern to search for images: $pattern"
  fi
  echo "output (subject) dir: $out_dir"
  echo "work dir: $hpc_work"
  echo ""
  echo "FastSurfer parameters:"
  echo "singularity image: $singularity_image"
  echo "FreeSurfer license: $fs_license"
  if [[ "$seg_only" == "true" ]]; then echo "--seg_only"; fi
  if [[ "$surf_only" == "true" ]]; then echo "--surf_only"; fi
  if [[ "$do_parallel" == "true" ]]; then echo "--parallel"; fi
  for p in "${POSITIONAL_FASTSURFER[@]}"
  do
    if [[ "$p" = --* ]]; then printf "\n%s" "$p";
    else printf " %s" '$p';
    fi
  done
  echo ""
  echo "Running in$(ls -l /proc/$$/exe | cut -d">" -f2)"
  echo ""
else
  if [[ "$submit_jobs" == "false" ]]; then echo "dry run, no jobs or operations are performed"; echo ""; fi
fi

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
  echo "WARNING!!!"
  echo "------------------------------------------------------------------------"
  echo "You specified the segmentation shall be performed on the cpu, but the"
  echo "time limit per segmentation is less than 6 minutes (default is optimized "
  echo "for GPU acceleration @ 5 minutes). This is very likely insufficient!"
  echo "------------------------------------------------------------------------"
fi

# step zero: make directories
if [[ "$submit_jobs" == "true" ]]
then
  if [[ "$make_hpc_work" == "true" ]]; then mkdir "$hpc_work" ; fi
  make_hpc_work_dirs "$hpc_work"
  echo "Setting up the work directory..."
fi

wait # for directories to be made

# step one: copy singularity image to hpc
all_cases_file="/$hpc_work/scripts/subject_list"

echo "cp \"$singularity_image\" \"$hpc_work/images/fastsurfer.sif\""
echo "cp \"$(dirname $THIS_SCRIPT)/brun_fastsurfer.sh\" \"$hpc_work/scripts\""
echo "cp \"$fs_license\" \"$hpc_work/scripts/.fs_license\""
echo "Create Status/Success file at $hpc_work/scripts/subject_success"

tofile="cat"
if [[ "$submit_jobs" == "true" ]]
then
  cp "$singularity_image" "$hpc_work/images/fastsurfer.sif" &
  cp "$(dirname $THIS_SCRIPT)/brun_fastsurfer.sh" "$hpc_work/scripts" &
  cp "$fs_license" "$hpc_work/scripts/.fs_license" &
  echo "#Status/Success file of srun_fastsurfer-run $(date)" > "$hpc_work/scripts/subject_success" &

  tofile="tee $all_cases_file"
fi

# step two: copy input data to hpc
if [[ -n "$subject_list" ]]
then
  # the test for files (check_subject_images) requires paths to be wrt
  cases=$(translate_cases "$in_dir" "$subject_list" "$in_dir" "${subject_list_delim}" "${subject_list_awk_code}")
  check_subject_images "$cases"

  cases=$(translate_cases "$in_dir" "$subject_list" "/source" "${subject_list_delim}" "${subject_list_awk_code}" | $tofile)
else
  cases=$(read_cases "$in_dir" "$pattern" "/source" | $tofile)
fi
num_cases=$(echo "$cases" | wc -l)

if [[ "$submit_jobs" != "true" ]]
then
  echo "Copying singularity image and scripts..."
fi
wait
# for copy and other stuff

brun_fastsurfer="scripts/brun_fastsurfer.sh"
fastsurfer_options=()
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
    echo "Sending emails on ALL conditions"
    slurm_email=("${slurm_email[@]}" --mail-type "ALL,ARRAY_TASKS")
  else
    echo "Sending emails on END,FAIL conditions"
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
    echo "srun --ntasks=1 --nodes=1 --cpus-per-task=$num_cpus_per_task \\"
    echo "  singularity exec --nv -B \"$hpc_work:/data,$in_dir:/source:ro\" --no-home \\"
    echo "  $hpc_work/images/fastsurfer.sif \\"
    echo "  /data/$brun_fastsurfer ${fastsurfer_options[*]} ${fastsurfer_seg_options[*]}"
  } > $seg_cmd_file
  if [[ "$cpu_only" == "true" ]]; then mem_seg="$mem_seg_cpu"
  else mem_seg="$mem_seg_gpu"
  fi
  # note that there can be a decent startup cost for each run, running multiple cases per task significantly reduces this
  seg_slurm_sched=("--mem=${mem_seg}G" "--cpus-per-task=$num_cpus_per_task"
                   --time=$(($timelimit_seg * $real_num_cases_per_task + 5))
                   $slurm_partition "${slurm_email[@]}"
                   "${jobarray_option[@]}" -J "FastSurfer-Seg-$USER"
                   -o "$hpc_work/logs/seg_%A_%a.log" "$seg_cmd_filename")
  if [[ "$cpu_only" == "true" ]]
  then
    if [[ "$debug" == "true" ]]
    then
      echo "Schedule SLURM job without gpu"
    fi
  else
    seg_slurm_sched=(--gpus=1 "${seg_slurm_sched[@]}")
  fi
  echo "chmod +x $seg_cmd_filename"
  chmod +x $seg_cmd_file
  echo "sbatch --parsable ${seg_slurm_sched[*]}"
  echo "--- sbatch script $seg_cmd_filename ---"
  cat $seg_cmd_file
  echo "--- end of script ---"
  if [[ "$submit_jobs" == "true" ]]
  then
    seg_jobid=$(sbatch --parsable ${seg_slurm_sched[*]})
    echo "Submitted Segmentation Jobs $seg_jobid"
  else
    echo "Not submitting the Segmentation Jobs to slurm (--dry)."
    seg_jobid=SEG_JOB_ID
  fi

  cleanup_depend="afterany:$seg_jobid"
  surf_depend="--depend=$jobarray_depend:$seg_jobid"
elif [[ "$surf_only" == "true" ]]
then
  # do not run segmentation, but copy over all cases from data to work
  copy_jobid=
  make_copy_job "$hpc_work" "$out_dir" "$hpc_work/scripts/subject_list" "$submit_jobs"
  if [[ -n "$copy_jobid" ]]
  then
    surf_depend="--depend=afterok:$copy_jobid"
  else
    echo "ERROR: \$copy_jobid not defined!"
    exit 1
  fi
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
  echo "sbatch --parsable ${surf_slurm_sched[*]}"
  echo "--- sbatch script $surf_cmd_filename ---"
  cat $surf_cmd_file
  echo "--- end of script ---"
  if [[ "$submit_jobs" == "true" ]]
  then
    surf_jobid=$(sbatch --parsable ${surf_slurm_sched[*]})
    echo "Submitted Surface Jobs $surf_jobid"
  else
    echo "Not submitting the Surface Jobs to slurm (--dry)."
    surf_jobid=SURF_JOB_ID
  fi

  cleanup_depend="afterany:$surf_jobid"
fi

# step four: copy results back and clean the output directory
if [[ "$do_cleanup" == "true" ]]
then
  make_cleanup_job "$hpc_work" "$out_dir" "$cleanup_depend" "$delete_hpc_work" "$submit_jobs"
else
  echo "Skipping the cleanup (no cleanup job scheduled, find your results in $hpc_work."
fi