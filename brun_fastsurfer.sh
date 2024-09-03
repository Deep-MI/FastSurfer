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

# script for batch-running FastSurfer

subjects=()
subjects_stdin="true"
POSITIONAL_FASTSURFER=()
task_count=""
task_id=""
surf_only="false"
seg_only="false"
debug="false"
run_fastsurfer=()
parallel_subjects="1"
parallel_surf="false"
statusfile=""

function usage()
{
  cat << EOF
Script to run FastSurfer on multiple subjects in parallel/series.

Usage:
brun_fastsurfer.sh --subject_list <file> [other options]
OR
brun_fastsurfer.sh --subjects <subject_id>=<file> [<subject_id>=<file> [...]] [other options]
OR
brun_fastsurfer.sh [other options]

Other options:
brun_fastsurfer.sh [...] [--batch "<i>/<n>"] [--parallel_subjects [surf=][<N>]]
    [--run_fastsurfer <script to run fastsurfer>] [--statusfile <filename>] [--debug] [--help]
    [<additional run_fastsurfer.sh options>]

Author:   David Kügler, david.kuegler@dzne.de
Date:     Nov 6, 2023
Version:  1.0
License:  Apache License, Version 2.0

Documentation of Options:
Generally, brun_fastsurfer works similar to run_fastsurfer, but loops over multiple subjects from
i. a list passed through stdin of the format (one subject per line)
---
<subject_id>=<path to t1 image>[ <subject-specific parameters>[ ...]]
...
---
ii. a subject_list file using the same format (use Ctrl-D to end the input), or
iii. a list of subjects directly passed (this does not support subject-specific parameters)

--batch "<i>/<n>": run the i-th of n batches (starting at 1) of the full list of subjects
  (default: 1/1, == run all). "slurm_task_id" is a valid option for "<i>".
  Note, brun_fastsurfer.sh will also automatically detect being run in a SLURM JOBARRAY and split
  according to \$SLURM_ARRAY_TASK_ID and \$SLURM_ARRAY_TASK_COUNT (unless values are specifically
  assigned with the --batch argument).
--parallel_subjects [surf=][<n>]: parallel execution of <n> or all (if <n> is not provided) subjects,
  specifically interesting for the surface pipeline (--surf_only) (default: serial execution, or
  '--parallel_subjects 1'). (Note, that currently only n=1 and n=-1 (no limit) are implemented.)
  Note, it is not recommended to parallelize the segmentation using --parallel_subjects on gpus,
  as that will cause out-of-memory errors, use --parallel_subjects surf=<n> to process segmentation
  in series and surfaces of <n> subjects in parallel.
  Note, that --parallel_subjects surf=<n> is not compatible with either --seg_only or --surf_only.
  The script will print the output of individual subjects interleaved, but prepend the subject_id.
--run_fastsurfer <path/command>: This option enables the startup of fastsurfer in a more controlled
  manner, for example to delegate the fastsurfer run to container:
  --run_fastsurfer "singularity exec --nv --no-home -B <dir>:/data /fastsurfer/run_fastsurfer.sh"
  Note, paths to files and --sd have to be defined in the container file system in this case.
--statusfile <filename>: a file to document which subject ran successfully. Also used to skip
  surface recon, if the previous segmentation failed.
--debug: Additional debug output.
--help: print this help.

With the exception of --t1 and --sid, all run_fastsurfer.sh options are supported, see
'run_fastsurfer.sh --help'.

This tool requires functions in stools.sh (expected in same folder as this script).
EOF
}

if [ -z "${BASH_SOURCE[0]}" ]; then THIS_SCRIPT="$0"
else THIS_SCRIPT="${BASH_SOURCE[0]}"
fi

# PRINT USAGE if called without params
if [[ $# -eq 0 ]]
then
  usage
  exit
fi

# PARSE Command line
inputargs=("$@")
POSITIONAL=()
SED_CLEANUP_SUBJECTS='s/\r$//;s/\s*\r\s*/\n/g;s/\s*$//;/^\s*$/d'
i=0
while [[ $# -gt 0 ]]
do
# make key lowercase
key=$(echo "$1" | tr '[:upper:]' '[:lower:]')
shift # past argument

case $key in
  # parse/get the subjects to iterate over
  #===================================================
  --subject_list|--subjects_list)
    if [[ ! -f "$1" ]]
    then
      echo "ERROR: Could not find the subject list $1!"
      exit 1
    fi
    # append the subjects in the listfile (cleanup first) to the subjects array
    mapfile -t -O ${#subjects} subjects < <(sed "$SED_CLEANUP_SUBJECTS" "$1")
    subjects_stdin="false"
    shift # past value
    ;;
  --subjects)
    subjects_stdin="false"
    while [[ ! "$1" =~ ^-- ]] ; do subjects+=("$1") ; shift ; done
    ;;
  # brun_fastsurfer-specific/custom options
  #===================================================
  --batch) task_count=$(echo "$1" | cut -f2 -d/) ;  task_id=$(echo "$1" | cut -f1 -d/) ; shift ;;
  --run_fastsurfer) run_fastsurfer=($1) ; shift ;;
  --parallel_subjects)
    if [[ "$#" -lt 1 ]] || [[ "$1" =~ ^-- ]]
    then # no additional parameter to --parallel_subjects, the next cmd args is unrelated
      parallel_subjects="max"
    else # has parameter
      lower_value="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
      if [[ "$lower_value" =~ ^surf(=-[0-9]*|=max)?$ ]]
      then # parameter is surf=max or surf=<negative number> or surf
        parallel_surf="true"
        parallel_subjects="max"
      elif [[ "$lower_value" =~ ^surf=[0-9]*$ ]]
      then # parameter is surf=<positive number>
        parallel_surf="true"
        parallel_subjects="${lower_value:5}"
      elif [[ "$lower_value" =~ ^(-[0-9]+|max)$ ]] ; then parallel_subjects="max"
      elif [[ "$lower_value" =~ ^[0-9]+$ ]] ; then parallel_subjects="$lower_value"
      else
        echo "Invalid option for --parallel_subjects: $1"
        exit 1
      fi
      shift
    fi
    ;;
  --statusfile) statusfile="$1" ; shift ;;
  --debug) debug="true" ;;
  --help) usage ; exit ;;
  # run_fastsurfer.sh options, with extra effect in brun_fastsurfer
  #===================================================
  --surf_only) surf_only="true" ;;
  --seg_only) seg_only="true" ;;
  --sid|--t1)
    echo "ERROR: --sid and --t1 are not valid for brun_fastsurfer.sh, these values are populated"
    echo "via --subjects or --subject_list."
    exit 1
    ;;
  *)    # unknown option/run_fastsurfer.sh option
    POSITIONAL_FASTSURFER[$i]="$key"
    i=$(($i + 1))
    ;;
esac
done
if [[ "${#POSITIONAL[@]}" -gt 0 ]]
then
  set -- "${POSITIONAL[@]}" # restore positional parameters
fi

echo "$THIS_SCRIPT ${inputargs[*]}"
date -R
echo ""

set -eo pipefail

source "$(dirname "$THIS_SCRIPT")/stools.sh"

if [[ -n "$SLURM_ARRAY_TASK_ID" ]]
then
  if [[ -z "$task_count" ]]
  then
    task_count=$SLURM_ARRAY_TASK_COUNT
  fi
  if [[ -z "$task_id" ]] || [[ "$task_id" == "slurm_task_id" ]]
  then
    task_id=$SLURM_ARRAY_TASK_ID
  fi
  echo "SLURM TASK ARRAY detected"
fi

if [[ "$subjects_stdin" == "true" ]]
then
  echo "Reading subjects from stdin, press Ctrl-D to end input (one subject per line)"
  mapfile -t -O ${#subjects[@]} subjects < <(sed "$SED_CLEANUP_SUBJECTS")
fi

if [[ "$debug" == "true" ]]
then
  echo "---START DEBUG---"
  echo "Debug parameters to script brun_fastsurfer:"
  echo ""
  echo "subjects: "
  printf "%s\n" "${subjects[@]}"
  echo "---"
  echo "task_id/task_count: ${task_id:-not specified}/${task_count:-not specified}"
  if [[ "$parallel_subjects" != "1" ]]
  then
    if [[ "$parallel_surf" == "true" ]] ; then echo "--parallel_subjects surf=$parallel_subjects"
    else echo "--parallel_subjects $parallel_subjects"
    fi
  fi
  if [[ "${run_fastsurfer[*]}" == "" ]] ;  then echo "running default run_fastsurfer"
  else echo "running ${run_fastsurfer[*]}"
  fi
  if [[ -n "$statusfile" ]] ;  then echo "statusfile: $statusfile" ; fi
  echo ""
  printf "FastSurfer parameters:"
  if [[ "$seg_only" == "true" ]]; then printf "\n--seg_only"; fi
  if [[ "$surf_only" == "true" ]]; then printf "\n--surf_only"; fi
  for p in "${POSITIONAL_FASTSURFER[@]}"
  do
    if [[ "$p" = --* ]]; then printf "\n%s" "$p";
    else printf " %s" "$p";
    fi
  done
  echo ""
  echo ""
  shell=$(ls -l "/proc/$$/exe" | cut -d">" -f2)
  echo "Running in shell $shell: $($shell --version 2>/dev/null | head -n 1)"
  echo ""
  echo "---END DEBUG  ---"
fi

if [[ "${#subjects[@]}" == 0 ]]
then
  echo "ERROR: No subjects specified!"
  exit 1
fi

if [[ "${run_fastsurfer[*]}" == "" ]]
then
  if [[ -n "$FASTSURFER_HOME" ]]
  then
    run_fastsurfer=("$FASTSURFER_HOME/run_fastsurfer.sh")
    echo "INFO: run_fastsurfer not explicitly specified, using \$FASTSURFER_HOME/run_fastsurfer.sh."
  elif [[ -f "$(dirname "$THIS_SCRIPT")/run_fastsurfer.sh" ]]
  then
    run_fastsurfer=("$(dirname "$THIS_SCRIPT")/run_fastsurfer.sh")
    echo "INFO: run_fastsurfer not explicitly specified, using ${run_fastsurfer[0]}."
  elif [[ -f "/fastsurfer/run_fastsurfer.sh" ]]
  then
    run_fastsurfer=("/fastsurfer/run_fastsurfer.sh")
    echo "INFO: run_fastsurfer not explicitly specified, using /fastsurfer/run_fastsurfer.sh."
  else
    echo "ERROR: Could not find FastSurfer, please set the \$FASTSURFER_HOME environment variable."
  fi
fi

num_subjects=${#subjects[@]}
if [[ -z "$task_id" ]] && [[ -z "$task_count" ]]
then
  subject_start=0
  subject_end=$num_subjects
elif [[ -z "$task_id" ]] || [[ -z "$task_count" ]]
then
  echo "Both task_id and task_count have to be defined, invalid --batch argument?"
  exit 1
else
  subject_start=$(((task_id - 1) * num_subjects / task_count))
  subject_end=$((task_id * num_subjects / task_count))
  subject_end=$((subject_end < num_subjects ? subject_end : num_subjects))
  echo "Processing subjects $subject_start to $subject_end"
fi
subject_len=$((subject_end - subject_start))

if [[ "$parallel_subjects" != "1" ]] && [[ "$subject_len" == 1 ]]
then
  if [[ "$debug" == "true" ]] ; then echo "DEBUG: --parallel_subjects deactivated, since only one subject" ; fi
  parallel_subjects="1"
fi

seg_surf_only=""
if [[ "$surf_only" == "true" ]] ; then seg_surf_only=--surf_only
elif [[ "$seg_only" == "true" ]] ; then seg_surf_only=--seg_only
fi

if [[ "$parallel_surf" == "true" ]]
then
  if [[ -n "$seg_surf_only" ]]
  then
    echo "ERROR: Cannot combine --parallel_subjects surf=<n> and --seg_only or --surf_only."
  fi
  seg_surf_only="--surf_only"
fi

### IF THE SCRIPT GETS TERMINATED, ADD A MESSAGE
trap 'echo "brun_fastsurfer.sh terminated via signal at $(date -R)!"' SIGINT SIGTERM

if [[ "$parallel_subjects" != "1" ]]
then
  echo "Running up to $parallel_subjects in parallel"
fi

function read_args_from_string ()
{
  tab=$(printf '\t')
  str_len=${#1}
  position=0
  while [[ "$position" -le "$str_len" ]]
  do
    if [[ -z "${1:$position}" ]]; then position=$((str_len + 1)); continue ; fi
    arg=$(expr "${1:$position} " : "\(\(\\\\.\|[^'\"[:space:]\\\\]\+\|'\([^']*\|''\)*'\|\"\([^\"\\\\]\+\|\\\\.\)*\"\)\+\).*")
    if [[ -z "$arg" ]]
    then
      # could not parse
      echo "Could not parse the line ${1:$position}, maybe incorrect quoting or escaping?"
      exit 1
    else
      # arg parsed
      if [[ "$position" == "0" ]]; then image_path=$arg
      else args=("${args[@]}" "$arg")
      fi
      position=$((position + ${#arg}))
    fi
    while [[ "${1:$position:1}" == " " ]] || [[ "${1:$position:1}" == "$tab" ]]; do position=$((position + 1)) ; done
  done
  export image_path
  export args
}

pids=()
subjectids=()
ROOT_IFS=$IFS
IFS=$'\n'

echo "${subjects[@]}"
# i is a 1-to-n index of the subject
i=$subject_start
for subject in "${subjects[@]:$subject_start:$subject_len}"
do
  IFS=$ROOT_IFS
  i=$((i + 1))
  if [[ "$debug" == "true" ]]
  then
    echo "DEBUG: subject $i: $subject"
  fi
  subject_id=$(echo "$subject" | cut -d= -f1)

  if [[ -n "$statusfile" ]] && [[ "$surf_only" == "true" ]]
  then
    status=$(awk -F ": " "/^$subject_id/ { print \$2 }" "$statusfile")
    ## if status in statusfile is "Failed", skip this
    if [[ "$status" =~ /^Failed.*--seg_only/ ]]
    then
      echo "Skipping $subject_id's surface recon because the segmentation failed."
      echo "$subject_id: Skipping surface recon (failed segmentation)" >> "$statusfile"
      continue
    fi
  fi

  image_parameters=$(echo "$subject" | cut -d= -f2-1000 --output-delimiter="=")
  args=(--sid "$subject_id")
  IFS=$' \t'
  read_args_from_string "$image_parameters"
  IFS=$ROOT_IFS
  if [[ "$parallel_surf" == "true" ]]
  then
    # parallel_surf implies $seg_surf_only == "" (see line 353), i.e. both seg and surf
    cmd=("${run_fastsurfer[@]}" "--seg_only" --t1 "$image_path" "${args[@]}" "${POSITIONAL_FASTSURFER[@]}")
    if [[ "$debug" == "true" ]] ; then echo "DEBUG:" "${cmd[@]}" ; fi
    if [[ "$parallel_subjects" != "1" ]] ; then "${cmd[@]}" | prepend "$subject_id: "
    else "${cmd[@]}"
    fi
    if [[ -n "$statusfile" ]]
    then
      print_status "$subject_id" "--seg_only" "$?" | tee -a "$statusfile"
    fi
  fi

  if [[ "$surf_only" == "false" ]] && [[ "$parallel_surf" == "false" ]]
  then
    args=("${args[@]}" --t1 "$image_path")
  fi
  if [[ "$debug" == "true" ]]
  then
    echo "DEBUG: ${run_fastsurfer[*]} $seg_surf_only" "${args[@]}" "${POSITIONAL_FASTSURFER[@]}" "[&]"
  fi
  if [[ "$parallel_subjects" != "1" ]]
  then
    "${run_fastsurfer[@]}" $seg_surf_only "${args[@]}" "${POSITIONAL_FASTSURFER[@]}" | prepend "$subject_id: " &
    pids=("${pids[@]}" "$!")
    subjectids=("${subjectids[@]}" "$subject_id")
  else # serial execution
    "${run_fastsurfer[@]}" $seg_surf_only "${args[@]}" "${POSITIONAL_FASTSURFER[@]}"
    if [[ -n "$statusfile" ]]
    then
      print_status "$subject_id" "$seg_surf_only" "$?" | tee -a "$statusfile"
    fi
  fi
  IFS=$'\n'
done
IFS=$ROOT_IFS

if [[ "$parallel_subjects" != "1" ]]
then
  # indexing in arrays is a 0-base operation, so array[0] is the first element
  i=-1
  for pid in "${pids[@]}"
  do
    i=$((i + 1))
    wait "$pid"
    if [[ -n "$statusfile" ]]
    then
      print_status "${subjectids[$i]}" "$seg_surf_only" "$?" | tee -a "$statusfile"
    fi
  done
fi

# always exit successful
exit 0
