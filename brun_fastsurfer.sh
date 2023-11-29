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

subjects=""
subjects_stdin="true"
POSITIONAL_FASTSURFER=()
task_count=""
task_id=""
surf_only="false"
seg_only="false"
debug="false"
run_fastsurfer="default"
parallel_subjects="false"
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
brun_fastsurfer.sh [...] [--batch "<i>/<n>"] [--parallel_subjects] [--run_fastsurfer <script to run fastsurfer>]
    [--statusfile <filename>] [--debug] [--help]
    [<additional run_fastsurfer.sh options>]

Author:   David Kügler, david.kuegler@dzne.de
Date:     Nov 6, 2023
Version:  1.0
License:  Apache License, Version 2.0

Documentation of Options:
Generally, brun_fastsurfer works similar to run_fastsurfer, but loops over multiple subjects from
i. a list passed through stdin of the format (one subject per line)
---
<subject_id>=<path to t1 image>
...
---
ii. a subject_list file using the same format (use Ctrl-D to end the input), or
iii. a list of subjects directly passed

--batch "<i>/<n>": run the i-th of n batches (starting at 1) of the full list of subjects
  (default: 1/1, == run all).
  Note, brun_fastsurfer.sh will also automatically detect being run in a SLURM JOBARRAY and split
  according to \$SLURM_ARRAY_TASK_ID and \$SLURM_ARRAY_TASK_COUNT (unless values are specifically
  assigned with the --batch argument).
--parallel_subjects: parallel execution of all subjects, specifically interesting for the surface
  pipeline (--surf_only). (default: serial execution).
  Note, it is not recommended to parallelize the segmentation using --parallel_subjects on gpus,
  as that will cause out-of-memory errors.
--run_fastsurfer <path/command>: This option enables the startup of fastsurfer in a more controlled
  manner, for example to delegate the fastsurfer run to container:
  --run_fastsurfer "singularity exec --nv --no-home -B <dir>:/data /fastsurfer/run_fastsurfer.sh"
  Note, paths to files and --sd have to be defined in the container file system in this case.
--statusfile <filename>: a file to document which subject ran successfully. Also used to skip
  surface recon, if the previous segmentation failed.
--debug: Additional debug output.
--help: print this help.

Almost all run_fastsurfer.sh options are supported, see run_fastsurfer.sh --help

This tool requires functions in stools.sh and the brun_fastsurfer.sh scripts (expected in same
folder as this script) in addition to the fastsurfer singularity image.
EOF
}

if [ -z "${BASH_SOURCE[0]}" ]; then
    THIS_SCRIPT="$0"
else
    THIS_SCRIPT="${BASH_SOURCE[0]}"
fi

# PRINT USAGE if called without params
if [[ $# -eq 0 ]]
then
  usage
  exit
fi

# PARSE Command line
newline="
"
inputargs=("$@")
POSITIONAL=()
i=0
while [[ $# -gt 0 ]]
do
# make key lowercase
key=$(echo "$1" | tr '[:upper:]' '[:lower:]')

case $key in
    --subject_list)
      if [[ ! -f "$2" ]]
      then
        echo "ERROR: Could not find the subject list $2!"
        exit 1
      fi
      subjects="$subjects $(cat $2)"
      subjects_stdin="false"
    shift # past argument
    shift # past value
    ;;
    --subjects)
      subjects_stdin="false"
      shift # argument
      while [[ "$(expr match \"$1\" '--.')" == 0 ]]
      do
        if [[ -n "$subjects" ]]; then subjects="$subjects$newline"; fi
        subjects="$subjects$1"
        shift # next value
      done
    ;;
    --batch)
      task_count=$(echo "$2" | cut -f2 -d/)
      task_id=$(echo "$2" | cut -f1 -d/)
      shift
      shift
    ;;
    --parallel_subjects)
      parallel_subjects="true"
      shift
    ;;
    --statusfile)
      statusfile="$statusfile"
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
    --sid|--t1)
      echo "ERROR: --sid and --t1 are not valid for brun_fastsurfer.sh, these values are populated"
      echo "via --subjects or --subject_list."
      exit 1
    ;;
    --debug)
      debug="true"
      shift
      ;;
    --help)
      usage
      exit
      ;;
    --run_fastsurfer)
      run_fastsurfer="$2"
      shift
      shift
      ;;
    *)    # unknown option
      POSITIONAL_FASTSURFER[$i]="$1"
      i=$(($i + 1))
      shift
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo "$THIS_SCRIPT ${inputargs[*]}"
date -R
echo ""

if [[ -n "$SLURM_ARRAY_TASK_ID" ]]
then
  if [[ -z "$task_count" ]]
  then
    task_count=$SLURM_ARRAY_TASK_COUNT
  fi
  if [[ -z "$task_id" ]]
  then
    task_id=$SLURM_ARRAY_TASK_ID
  fi
  echo "SLURM TASK ARRAY detected"
fi

if [[ "$debug" == "true" ]]
then
  echo "---START DEBUG---"
  echo "Debug parameters to script brun_fastsurfer:"
  echo ""
  echo "subjects: "
  echo $subjects
  echo "---"
  echo "task_id/task_count: $task_id/$task_count"
  if [[ "$run_fastsurfer" != "/fastsurfer/run_fastsurfer.sh" ]]
  then
    echo "running $run_fastsurfer"
  fi
  if [[ -n "$statusfile" ]]
  then
    echo "statusfile: $statusfile"
  fi
  echo ""
  echo "FastSurfer parameters:"
  if [[ "$seg_only" == "true" ]]; then echo "--seg_only"; fi
  if [[ "$surf_only" == "true" ]]; then echo "--surf_only"; fi
  for p in "${POSITIONAL_FASTSURFER[@]}"
  do
    if [[ "$p" = --* ]]; then printf "\n%s" "$p";
    else printf " %s" "$p";
    fi
  done
  echo ""
  echo "Running in$(ls -l /proc/$$/exe | cut -d">" -f2)"
  echo ""
  echo "---END DEBUG  ---"
fi

if [[ "$subjects_stdin" == "true" ]]
then
  echo "Reading subjects from stdin, press Ctrl-D to end input (one subject per line)"
  subjects="$(cat)"
fi

if [[ -z "$subjects" ]]
then
  echo "ERROR: No subjects specified!"
  exit 1
fi

i=1
num_subjects=$(echo "$subjects" | wc -l)

if [[ "$run_fastsurfer" == "default" ]]
then
  if [[ -n "$FASTSURFER_HOME" ]]
  then
    run_fastsurfer=$FASTSURFER_HOME/run_fastsurfer.sh
    echo "INFO: run_fastsurfer not explicitly specified, using \$FASTSURFER_HOME/run_fastsurfer.sh."
  elif [[ -f "$(dirname $THIS_SCRIPT)/run_fastsurfer.sh" ]]
  then
    run_fastsurfer="$(dirname $THIS_SCRIPT)/run_fastsurfer.sh"
    echo "INFO: run_fastsurfer not explicitly specified, using $run_fastsurfer."
  elif [[ -f "/fastsurfer/run_fastsurfer.sh" ]]
  then
    run_fastsurfer="/fastsurfer/run_fastsurfer.sh"
    echo "INFO: run_fastsurfer not explicitly specified, using /fastsurfer/run_fastsurfer.sh."
  else
    echo "ERROR: Could not find FastSurfer, please set the \$FASTSURFER_HOME environment variable."
  fi
fi

if [[ -z "$task_id" ]] && [[ -z "$task_count" ]]
then
  subject_start=1
  subject_end=$num_subjects
elif [[ -z "$task_id" ]] || [[ -z "$task_count" ]]
then
  echo "Both task_id and task_count have to be defined, invalid --batch argument?"
else
  subject_start=$(($(($task_id - 1)) * "$num_subjects" / "$task_count" + 1))
  subject_end=$(("$task_id" * "$num_subjects" / "$task_count"))
  echo "Processing subjects $subject_start to $subject_end"
fi

seg_surf_only=""
if [[ "$surf_only" == "true" ]]
then
  seg_surf_only=--surf_only
elif [[ "$seg_only" == "true" ]]
then
  seg_surf_only=--seg_only
fi

### IF THE SCRIPT GETS TERMINATED, ADD A MESSAGE
trap "{ echo \"brun_fastsurfer.sh terminated via signal at \$(date -R)!\" }" SIGINT SIGTERM

pids=()
subjectids=()
for subject in $subjects
do
  if [[ "$debug" == "true" ]]
  then
    echo "DEBUG: subject $i: $subject"
  fi
  # if the subject is in the selected batch
  if [[ "$i" -ge "$subject_start" ]] && [[ "$i" -le "$subject_end" ]]
  then
    subject_id=$(echo "$subject" | cut -d= -f1)

    if [[ -n "$statusfile" ]] && [[ "$surf_only" == "true" ]]
    then
      status=$(awk -F ": " "/^$subject_id/ { print \$2 }" "$statusfile")
      ## if status in statusfile is "Failed", skip this
      if [[ "$status" =~ /^Failed.--seg_only/ ]]
      then
        echo "Skipping $subject_id's surface recon because the segmentation failed."
        echo "$subject_id: Skipping surface recon (failed segmentation)" >> "$statusfile"
        continue
      fi
    fi

    image_path=$(echo "$subject" | cut -d= -f2)
    args=(--sid "$subject_id")
    if [[ "$surf_only" == "false" ]]
    then
      args=("${args[@]}" --t1 "$image_path")
    fi
    if [[ "$debug" == "true" ]]
    then
      echo "DEBUG: $run_fastsurfer $seg_surf_only" "${args[@]}" "${POSITIONAL_FASTSURFER[@]}" "[&]"
    fi
    if [[ "$parallel_subjects" == "true" ]]
    then
      $run_fastsurfer "$seg_surf_only" "${args[@]}" "${POSITIONAL_FASTSURFER[@]}" &
      pids=("${pids[@]}" "$!")
      subjectids=("${subjectids[@]}" "$subject_id")
    else
      $run_fastsurfer "$seg_surf_only" "${args[@]}" "${POSITIONAL_FASTSURFER[@]}"
      if [[ -n "$statusfile" ]]
      then
        print_status "$subject_id" "$seg_surf_only" "$?" | tee -a "$statusfile"
      fi
    fi
  fi
  i=$(($i + 1))
done

if [[ "$parallel_subjects" == "true" ]]
then
  i=0
  for pid in "${pids[@]}"
  do
    wait "$pid"
    if [[ -n "$statusfile" ]]
    then
      print_status "${subjectids[$i]}" "$seg_surf_only" "$?" | tee -a "$statusfile"
    fi
    i=$(($i + 1))
  done
fi

# always exit successful
exit 0