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
parallel_pipelines="1"
num_parallel_surf="1"
num_parallel_seg="1"
statusfile=""
python="python3.10 -s"

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
brun_fastsurfer.sh [...] [--batch "<i>/<n>"] [--parallel <N>|max] [--parallel_seg <N>|max] [--parallel_surf <N>|max]
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
--parallel <n>|max: parallel execution of run_fastsurfer for <n> images. Creates <n> processes with
  each process performing segmentation and surface reconstruction. The default is this serial execution
  mode with n=1: '--parallel 1'.
--parallel_seg <n>|max and
--parallel_surf <m>|max: activate independent segmentation and surface reconstruction pipelines.
  Segmentation and Surface reconstruction have independent processing queues. After successful
  segmentation (<n> parallel processes), cases are transferred into the surface queue (<m> parallel
  processes). Together max. m+n processes will run. Logfiles unchanged, console output for individual
  subjects is interleaved with subject_id prepended.
--run_fastsurfer <path/command>: This option enables the startup of fastsurfer in a more controlled
  manner, for example to delegate the fastsurfer run to container:
  --run_fastsurfer "singularity exec --nv --no-mount home,cwd -e -B <dir>:/data /fastsurfer/run_fastsurfer.sh"
  Note, paths to files and --sd have to be defined in the container file system in this case.
--statusfile <filename>: a file to document which subject ran successfully. Also used to skip
  surface recon, if the previous segmentation failed.
--threads <n>,
--threads_seg <n>, and
--threads_surf <n>: specify number of threads for each parallel "process", i.e.
  total_threads=num_seg_processes * num_seg_threads + num_surf_processes * num_surf_threads.
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

function verify_parallel()
{
  # 1: flag, 2: value
  value="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
  if [[ "$value" =~ ^(max|-[0-9]+|0)$ ]] ; then verify_value="max"
  elif [[ "$value" =~ ^[0-9]+$ ]] ; then verify_value="$value"
  else echo "ERROR: Invalid value for $1: '$2', must be integer or 'max'." ; exit 1
  fi
  export verify_value
}

function warn_old()
{
  echo "WARNING: The syntax to ${key-'--parallel_subjects'} seg=/surf= is outdated and will be removed in FastSurfer 3,"
  echo "  use --parallel <n>, --parallel_seg <n>, or --parallel_surf <n>!"
}

function fail_bash_version_lt4()
{
  if [[ ! "$(bash --version | head -n 1)" =~ [vV]ersion[[:space:]][4-9] ]]
  then
    echo "ERROR: The brun_fastsurfer script requires at minimum bash version 4 for the options --subject_list and"
    echo "  subjects via stdin. Specifying a specific number of concurrent processes (--parallel <num>,"
    echo "  --parallel_seg <num>, --parallel_surf <num>; num is a positive integer) also requires bash 4+."
    exit 1
  fi
}

# PARSE Command line
inputargs=("$@")
POSITIONAL=()
res_device="auto"
res_viewagg_device="auto"
SED_CLEANUP_SUBJECTS='s/\r$//;s/\s*\r\s*/\n/g;s/\s*$//;/^\s*$/d'
prev_ifs="$IFS"
i=0
while [[ $# -gt 0 ]]
do
# make key lowercase
arg="$1"
key=$(echo "$arg" | tr '[:upper:]' '[:lower:]')
shift # past argument

case $key in
  # parse/get the subjects to iterate over
  #===================================================
  --subject_list|--subjects_list)
    fail_bash_version_lt4
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
  --subjects) subjects_stdin="false" ; while [[ ! "$1" =~ ^-- ]] ; do subjects+=("$1") ; shift ; done ;;
  # brun_fastsurfer-specific/custom options
  #===================================================
  --batch) task_count=$(echo "$1" | cut -f2 -d/) ;  task_id=$(echo "$1" | cut -f1 -d/) ; shift ;;
  --run_fastsurfer) IFS=" " ; run_fastsurfer=($1) ; shift ;;
  --parallel|--parallel_subjects)
    # future syntax: --parallel n|max|n/m
    # currently the former syntax --parallel_subjects seg=n|max, etc. is still supported but deprecated
    if [[ "$key" == "--parallel_subjects" ]] ; then
      echo "WARNING: The --parallel_subjects option is obsolete and replaced with --parallel <option>."
    fi
    if [[ "$#" -lt 1 ]] || [[ "$1" =~ ^-- ]]
    then
      if [[ "$key" == "--parallel_subjects" ]] ; then
        parallel_pipelines="1" ; num_parallel_surf="max" ; num_parallel_seg="max" ; warn_old
      else POSITIONAL_FASTSURFER+=("--parallel")
        echo "WARNING: --parallel without any argument for hemisphere parallelization is superseded by --threads 2+"
        echo "  and will be removed in FastSurfer 3."
      # this is just the parallel option
      fi
    else # has parameter
      value="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
      if [[ "$value" =~ ^surf(=-[0-9]*|=max)?$ ]] ; then num_parallel_surf="max" ; parallel_pipelines="2" ; warn_old
      elif [[ "$value" =~ ^surf=[0-9]*$ ]] ; then  num_parallel_surf="${value:5}" ; parallel_pipelines="2" ; warn_old
      elif [[ "$value" =~ ^seg(=-[0-9]*|=max)?$ ]] ; then num_parallel_seg="max" ; parallel_pipelines="2" ; warn_old
      elif [[ "$value" =~ ^seg=[0-9]*$ ]] ; then num_parallel_seg="${value:4}" ; parallel_pipelines="2" ; warn_old
      elif [[ "$value" =~ ^(-[0-9]+|max)$ ]] ; then parallel_pipelines=1 ; num_parallel_seg="max" ; num_parallel_surf="max"
      elif [[ "$value" =~ ^[0-9]+$ ]] ; then parallel_pipelines=1 ; num_parallel_seg="$value" ; num_parallel_surf="$value"
      elif [[ "$value" =~ ^[0-9]+/[-9]+$ ]] ; then parallel_pipelines=2
        num_parallel_seg="$(echo "$value" | cut -d/ -f1)" ; num_parallel_surf="$(echo "$value" | cut -d/ -f2)"
      else echo "Invalid option for $key: $1" ; exit 1
      fi
      shift
    fi
    ;;
  --parallel_seg)
    parallel_pipelines=2 ; verify_parallel "$key" "$1" ; num_parallel_seg="$verify_value"; shift ;;
  --parallel_surf)
    parallel_pipelines=2 ; verify_parallel "$key" "$1" ; num_parallel_surf="$verify_value" ; shift ;;
  --statusfile) statusfile="$1" ; shift ;;
  --py) python="$1" ; POSITIONAL_FASTSURFER+=(--py "$1") ; shift ;; # this may be needed to get the device count
  --debug) debug="true" ;;
  --help) usage ; exit ;;
  # run_fastsurfer.sh options, with extra effect in brun_fastsurfer
  #===================================================
  --device) res_device="$1" ; shift ;;
  --viewagg_device) res_viewagg_device="$1" ; shift ;;
  --surf_only) surf_only="true" ;;
  --seg_only) seg_only="true" ;;
  --sid|--t1)
    echo "ERROR: --sid and --t1 are not valid for brun_fastsurfer.sh, these values are populated"
    echo "  via --subjects or --subject_list."
    exit 1
    ;;
  *)    # unknown option/run_fastsurfer.sh option, make sure this is arg (to keep the case)
    POSITIONAL_FASTSURFER["$i"]="$arg"
    i=$((i + 1))
    ;;
esac
done
 # restore positional parameters
if [[ "${#POSITIONAL[@]}" -gt 0 ]] ; then set -- "${POSITIONAL[@]}" ; fi
IFS="$prev_ifs"

source "$(dirname "$THIS_SCRIPT")/stools.sh"

if [[ -n "$SLURM_ARRAY_TASK_ID" ]]
then
  if [[ -z "$task_count" ]] ; then task_count=$SLURM_ARRAY_TASK_COUNT ; fi
  if [[ -z "$task_id" ]] || [[ "$task_id" == "slurm_task_id" ]] ; then task_id=$SLURM_ARRAY_TASK_ID ; fi
  echo "SLURM TASK ARRAY detected"
fi

function get_device_list()
{
  # outputs device list numbers separated with "," to stdout
  # error messages are sent to stderr
  local list="" host="" out="" i prev_ifs="$IFS"
  if [[ "$1" =~ ^(cpu|mps|auto|cuda)$ ]] || [[ "$1" =~ ^cuda:[0-9]+(,[0-9]+)*$ ]] ; then list=",$1"
  elif [[ "$1" =~ ^cuda:[0-9]+([-,][0-9]+(-[0-9]+)?)* ]] ; then
    IFS="," ; host=${1:0:5}
    for i in ${1:5} ; do IFS="-" ; v=($i) ; list+=",$(seq -s"," "${v[0]}" "$(("${v[1]}" - 1))")" ; done
  else
    echo "ERROR: Invalid format for device|viewagg_device: $1 must be auto|cpu|mps|cuda[:X[,Y][-Z]...]"
    exit 1
  fi
  IFS=","
  for i in ${list:1} ; do
    if [[ "${out/$i/}" != "$out" ]] ; then echo "WARNING: Duplicate device $i in $1!" ; fi
    if [[ -n "$i" ]] ; then out+=",$i" ; fi
  done
  IFS="$prev_ifs"
  device_value="$host${out:1}"  # remove leading ","
  export device_value
}

# we do not know here, whether a gpu is available, so even if we have none or one, give a warning message.
if [[ "$num_parallel_seg" != 1 ]] && [[ "$surf_only" != "true" ]] && [[ "$res_device" =~ ^auto|cuda$ ]]
then
  # device is auto or cuda, auto-detect the device count and make it match with num_parallel_seg
  detected_devices=$($python -c "import torch.cuda.device_count as d ; print(*range(d()), sep=',')")
  _devices=($detected_devices)
  num_devices="${#_devices[@]}"
  echo "INFO: Auto-detecting CUDA-capable devices to parallelize segmentation, found $num_devices device(s)."
  if [[ "${#detected_devices[@]}" -le 1 ]] ; then echo "  => No changes!" # keep auto/cuda
  elif [[ "$num_parallel_seg" == "max" ]] || [[ "$num_parallel_seg" -gt "$num_devices" ]] ; then
    res_device="cuda:$detected_devices"
    num_parallel_seg=$num_devices
    echo "  => Setting number of parallel segmentations to number of devices (one segmentation per device)."
  else
    res_device="cuda:$(seq -s"," 0 $((num_parallel_seg - 1)))"
    echo "  => Limited by $num_parallel_seg parallel segmentations."
  fi
else
  get_device_list "$res_device" ; res_device="$device_value"
  get_device_list "$res_viewagg_device" ; res_viewagg_device="$device_value"
fi
if [[ "$subjects_stdin" == "true" ]]
then
  fail_bash_version_lt4
  if [[ -t 0 ]] || [[ "$debug" == "true" ]]; then
    echo "Reading subjects from stdin, press Ctrl-D to end input (one subject per line)"
  fi
  mapfile -t -O ${#subjects[@]} subjects < <(sed "$SED_CLEANUP_SUBJECTS")
fi

echo "$THIS_SCRIPT ${inputargs[*]}"
date -R
echo ""

if [[ "$debug" == "true" ]]
then
  echo "---START DEBUG---"
  echo "Debug parameters to script brun_fastsurfer:"
  echo ""
  echo "subjects: "
  printf "%s\n" "${subjects[@]}"
  echo "---"
  echo "task_id/task_count: ${task_id:-not specified}/${task_count:-not specified}"
  if [[ "$parallel_pipelines" == "1" ]]
  then
    if [[ "$num_parallel_seg" == 1 ]] ; then echo "parallelization: no subject parallelization"
    else echo "parallelization: $num_parallel_seg parallel subjects, segmentation/surface in series"
    fi
  else
    echo "parallelization: Independent segmentation and surface parallelization:"
    echo "  segmentation: $num_parallel_seg parallel subjects, surface: $num_parallel_surf parallel subjects"
  fi
  IFS=" "
  if [[ "${run_fastsurfer[*]}" == "" ]] ;  then echo "running default run_fastsurfer"
  else echo "running ${run_fastsurfer[*]}"
  fi
  if [[ -n "$statusfile" ]] ;  then echo "statusfile: $statusfile" ; fi
  printf "\nFastSurfer parameters:"
  if [[ "$seg_only" == "true" ]]; then printf "\n--seg_only" ; fi
  if [[ "$surf_only" == "true" ]]; then printf "\n--surf_only" ; fi
  if [[ -n "$res_device" ]] ; then printf "\n--device %s" "$res_device" ; fi
  if [[ -n "$res_viewagg_device" ]] ; then printf "\n--viewagg_device %s" "$res_viewagg_device" ; fi
  for p in "${POSITIONAL_FASTSURFER[@]}"
  do
    if [[ "$p" = --* ]]; then printf "\n%s" "$p" ;
    else printf " %s" "$p" ;
    fi
  done
  printf "\n\n"
  shell=$(ls -l "/proc/$$/exe" | cut -d">" -f2)
  printf "Running in shell %s: %s\n" "$shell" "$($shell --version 2>/dev/null | head -n 1)"
  echo "---END DEBUG  ---"
fi

if [[ "${#subjects[@]}" == 0 ]] ; then echo "ERROR: No subjects specified!" ; exit 1 ; fi

if [[ -z "${run_fastsurfer[*]}" ]]
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
  echo "ERROR: Both task_id and task_count have to be defined, invalid --batch argument?"
  exit 1
else
  subject_start=$(((task_id - 1) * num_subjects / task_count))
  subject_end=$((task_id * num_subjects / task_count))
  subject_end=$((subject_end < num_subjects ? subject_end : num_subjects))
  echo "INFO: Processing subjects $((subject_start + 1)) to $subject_end"
fi

if [[ "$parallel_pipelines" == "2" ]] ; then
  if [[ "$seg_only" == "true" ]] ; then parallel_pipelines=1
  elif [[ "$surf_only" == "true" ]] ; then parallel_pipelines=1; num_parallel_seg=$num_parallel_surf
  fi
fi

### IF THE SCRIPT GETS TERMINATED, ADD A MESSAGE
for signal in SIGINT SIGTERM ; do
  trap "$(printf 'echo "brun_fastsurfer.sh terminated via signal %s at $(date -R)!"' "$signal")" $signal
done

function iterate_subjects_with_token()
{
  # 1: subject_start
  # 2: subject_end
  # *: subjects
  subject_start="$1"
  subject_end="$2"
  shift
  shift
  subjects=("$@")
  # i is a 1-to-n index of the subject
  local i="$subject_start"
  for subject in "${subjects[@]:$subject_start:$((subject_end - subject_start))}"
  do
    echo "#@#!NEXT-SUBJECT:$subject"
  done
}

function run_single()
{
  # Usage:
  # run_single subject-specific-string "debug" "statusfile" "parallel_pipelines" "num_parallel_seg" "num_parallel_surf" "run_fastsurfer" "POSITIONAL_FASTSURFER..."
  # run_single "<subject_id>=<path-to-t1> <more options>" ...

  # 1: subject-specific-string
  # 2: debug
  # 3: statusfile
  # 4: parallel_pipelines
  # 5: num_parallel_seg
  # 6: num_parallel_surf
  # *: run_fastsurfer
  # |
  # *: POSITIONAL_FASTSURFER

  local subject_id="<undefined>" mode="" args=() do_seg=1 do_surf=1 skip=0 parallel cmd status=unknown statustext
  local debug="$2" statusfile="$3" parallel_pipelines="$4" num_parallel_seg="$5" num_parallel_surf="$6"
  local run_fastsurfer=() POSITIONAL_FASTSURFER=()
  local position=0 arg image_path="<undefined>" returncode=0
  local regex="\(\(\\\\.\|[^'\"[:space:]\\\\]\+\|'\([^']*\|''\)*'\|\"\([^\"\\\\]\+\|\\\\.\)*\"\)\+\).*"
  subject_id=$(echo "$1" | cut -d= -f1)
  image_parameters=$(echo "$1" | cut -d= -f2-1000 --output-delimiter="=")
  for run in {1..6}; do shift ; done
  for i in "$@" ; do shift; if [[ "$i" == "|" ]] ; then break ; fi ; run_fastsurfer+=("$i") ; done
  POSITIONAL_FASTSURFER=("$@")

  # read image_parameters into args and image_path
  while [[ "$position" -le "${#image_parameters}" ]]
  do
    if [[ -z "${image_parameters:$position}" ]]; then position=$((${#image_parameters} + 1)); continue ; fi
    arg="$(expr "${image_parameters:$position} " : "$regex")"
    if [[ -z "$arg" ]]
    then
      # could not parse
      echo "ERROR: Could not parse the line ${image_parameters:$position}, maybe incorrect quoting or escaping?"
      exit 1
    else
      # arg parsed
      if [[ "$position" == "0" ]]; then image_path=$arg ; else args+=("$arg") ; fi
      position=$((position + ${#arg}))
    fi
    while [[ "${image_parameters:$position:1}" == " " ]] ; do position=$((position + 1)) ; done
  done

  args+=(--sid "$subject_id")
  for i in "${args[@]}"; do
    if [[ "$i" =~ ^--(viewagg_)?device$ ]] ; then
      echo "ERROR: --device or --viewagg_device may not be specified in the subject-specific arguments!"
      exit 1
    fi
  done
  for i in "${POSITIONAL_FASTSURFER[@]}" "${args[@]}"; do
    if [[ "$i" == "--seg_only" ]] ; then mode="$i" ; do_surf=0
    elif [[ "$i" == "--surf_only" ]] ; then mode="$i" ; do_seg=0
    fi
  done
  if [[ "$do_seg" == 0 ]] && [[ "$do_surf" == 0 ]]
  then
    echo "INFO: Skipping subject_id $subject_id (deselected)"
    skip=1
  elif [[ -n "$statusfile" ]] && [[ "$do_surf" == 1 ]] && [[ "$do_seg" == 0 ]]
  then
    ## if status in statusfile is "Failed" last, skip this
    prev_ifs="$IFS" ; IFS=''
    while read -r line ; do
      IFS=" "
      subject="$(echo "$line" | cut -d" " -f1)"
      if [[ "$subject" == "$subject_id:" ]] ; then
        statustext="${line:$((${#subject} + 1))}"
        if [[ "${statustext}" =~ ^Failed[[:space:]](--seg_only|seg|both) ]] ; then status=failed
        elif [[ "${statustext}" =~ ^Finished[[:space:]](--seg_only|seg|both)[[:space:]]success ]] ; then
          status=success
        fi
      fi
      IFS=""
    done < "$statusfile"
    IFS="$prev_ifs"
    if [[ "$status" == "failed" ]]
    then
      echo "INFO: Skipping $subject_id's surface recon because the segmentation failed."
      echo "$subject_id: Skipping surface recon (failed segmentation)" >> "$statusfile"
      skip=1
    fi
  fi
  # if we are not skipping this, process
  if [[ "$skip" == 0 ]]
  then
    cmd=("${run_fastsurfer[@]}" --t1 "$image_path" "${POSITIONAL_FASTSURFER[@]}" "${args[@]}")
    if [[ "$debug" == "true" ]] ; then echo "DEBUG:" "${cmd[@]}" ; fi
    # multiple subjects in parallel is possible, then parallel = 1, else parallel = 0
    if [[ "$num_parallel_seg" == "max" ]] || [[ "$parallel_pipelines" == 2 ]] && [[ "$num_parallel_surf" == "max" ]]
    then parallel=1 # one of "running pipeline" is max
    else parallel=$([[ $((num_parallel_seg + (parallel_pipelines - 1) * num_parallel_surf)) == 2 ]] && echo 0 || echo 1)
    fi
    if [[ "$parallel" == 1 ]] ; then "${cmd[@]}" | prepend "$subject_id: " ; else "${cmd[@]}" ; fi
    returncode="${PIPESTATUS[0]}"
    if [[ -n "$statusfile" ]] ; then print_status "$subject_id" "$mode" "$returncode" >> "$statusfile"; fi
    if [[ "$returncode" != 0 ]]; then echo "WARNING: $subject_id finished with exit code $returncode!" ; fi
  fi
  echo "#@#!NEXT-SUBJECT:$subject_id=$image_parameters"
}

function is_numbered_device() { if [[ "$1" =~ ^auto|cpu|mps|cuda$ ]] ; then return 1 ; else return 0 ; fi }

# this is a handler to convert the device name to a number
function device2number() { echo "${1:5}" ; }

timeout_read_token=5
function process_by_token()
{
  # 1: "surf" / "seg" / "both"
  # 2: timeout_read_token
  # 3: debug
  # 4: statusfile
  # 5: parallel_pipelines
  # 6: num_parallel_seg
  # 7: num_parallel_surf
  # 8: device
  # 9: viewagg_evice
  # *: run_fastsurfer
  # |
  # *: POSITIONAL_FASTSURFER

  local subject_id max_processes subject_buffer=() read_in=1 returncode spawn_task=1 mode="$1" line device_ready=0
  local timeout_read_token="$2" debug="$3" statusfile="$4" parallel_pipelines="$5" num_parallel_seg="$6" this_args=()
  local num_parallel_surf="$7" device=() vdevice=() regx="^cuda:[0-9]+," parallel_warn=0 res_args=() prev_ifs="$IFS"
  IFS=","
  if [[ "$8" =~ $regx ]] ; then for e in ${8:5} ; do device+=("${8:0:5}$e") ; done ; else device=("$8") ; fi
  if [[ "$9" =~ $regx ]] ; then for e in ${9:5} ; do vdevice+=("${9:0:5}$e") ; done ; else vdevice=("$9") ; fi
  IFS="$prev_ifs"

  for run in {1..9}; do shift ; done
  local run_single_args=("$debug" "$statusfile" "$parallel_pipelines" "$num_parallel_seg" "$num_parallel_surf" "$@")

  if [[ "$mode" == "surf" ]] ; then max_processes="$num_parallel_surf" ; else max_processes="$num_parallel_seg" ; fi
  while [[ "$read_in" == 1 ]] || [[ "${#subject_buffer[@]}" -gt 0 ]]
  do
    if [[ "$read_in" == 1 ]]
    then
      IFS=""
      read -r -t "$timeout_read_token" line
      returncode="$?"
      if [[ "$returncode" == 1 ]] ; then read_in=0 # EOF, terminate looking at for input
      elif [[ "$returncode" == 0 ]] ; then # successfully read a line
        if [[ "${line:0:17}" == "#@#!NEXT-SUBJECT:" ]] ; then
          if [[ "$debug" == "true" ]] ; then echo "DEBUG: $mode-subject ${line:17}" ; fi
          subject_buffer+=("${line:17}")
        else echo "$line"
        fi
      # else, e.g. 142 => timeout
      fi
    fi

    # check job count
    if [[ "$max_processes" == "max" ]] ; then spawn_task=1
    else
      fail_bash_version_lt4
      mapfile -t running_jobs < <(jobs -pr)
      if [[ "${#running_jobs[@]}" -lt "$max_processes" ]] ; then spawn_task=1
      elif [[ "$read_in" == 0 ]] ; then wait "${running_jobs[@]}" # wait for any task to finish (std is already closed)
      else spawn_task=0
      fi
    fi

    # if can spawn and has job in queue, start a job
    if [[ "$spawn_task" == 1 ]] && [[ "${#subject_buffer[@]}" -gt 0 ]]
    then
      if [[ "$mode" == "surf" ]] ; then device_ready=2 ; dev="cpu" ; vdev="cpu"
      else
        device_ready=0
        if [[ "${#device[@]}" -gt 1 ]] ; then
          # go through device assignments, if the processes finished, release the device assignment
          for name in "${device[@]}" ; do
            i="$(device2number "$name")"
            if [[ -z "${used_device[i]}" ]] || [[ -z "$(ps --no-headers "${used_device[i]}")" ]] ; then
              res_args=("--device" "$name") ; used_device[i]=""; device_ready=$((device_ready + 1)) ; dev="$name" ; break
            fi
          done
        else device_ready=$((device_ready + 1)) ; res_args=("--device" "${device[0]}")
        fi
        if [[ "${#vdevice[@]}" -gt 1 ]] ; then
          # go through viewagg device assignments, if the processes finished, release the device assignment
          for name in "${vdevice[@]}" ; do
            i="$(device2number "$name")"
            if [[ -z "${used_vdevice[i]}" ]] || [[ -z "$(ps --no-headers "${used_vdevice[i]}")" ]] ; then
              res_args=("--viewagg_device" "$name") ; used_vdevice[i]="" ; device_ready=$((device_ready + 1)) ; vdev="$name" ; break
            fi
          done
        else device_ready=$((device_ready + 1)) ; res_args=("--viewagg_device" "${vdevice[0]}")
        fi
      fi
      if [[ "$device_ready" -gt 1 ]]
      then
        subject_id=$(echo "${subject_buffer[0]}" | cut -d= -f1)
        if [[ "$debug" == "true" ]] ; then echo "DEBUG: Processing $subject_id with $mode" ; fi

        # start next process
        this_args=( "${run_single_args[@]}" "${res_args[@]}")
        if [[ "$mode" == "seg" ]] ; then run_single "${subject_buffer[0]}" "${this_args[@]}" --seg_only &
        elif [[ "$mode" == "surf" ]] ; then run_single "${subject_buffer[0]}" "${this_args[@]}" --surf_only &
        else run_single "${subject_buffer[0]}" "${this_args[@]}" &
        fi
        pid=$!
        # if dev or vdev is a numbered device, add it to the used devices list (multiple devices)
        if is_numbered_device "$dev" ; then i=$(device2number "$dev") ; used_device[i]="$pid" ; fi
        if is_numbered_device "$vdev" ; then i=$(device2number "$vdev") ; used_vdevice[i]="$pid" ; fi
        subject_buffer=("${subject_buffer[@]:1}")
      else
        if [[ $parallel_warn == 1 ]] ; then
          echo "WARNING: All devices are in use for parallel seg processing!"
        else
          echo "WARNING: All devices are in use, make sure you are trying to use more parallel seg processes than"
          echo "  you have devices passed in --device AND --viewagg_device., e.g. '--device cuda:0,1 --viewagg_device"
          echo "  cpu --parallel_seg 2' is fine, but '--device cuda:0,1 --parallel_seg 3' or"
          echo "  '--viewagg_device cuda:0,1 --parallel_seg 3' will cause issues."
          parallel_warn=1
        fi
        # wait for a device to become available
        sleep 5
      fi
    fi
  done
  if [[ "$(bash --version | head -n 1)" =~ [vV]ersion[[:space:]][4-9] ]] ; then mapfile -t running_jobs < <(jobs -pr)
  else running_jobs=()
  fi
  # wait for jobs to finish
  if [[ "$debug" == "true" ]]
  then
    echo "DEBUG: Finished scheduling $mode-jobs... waiting for ${#running_jobs[@]} jobs to finish:"
    IFS=" "
    echo "       ${running_jobs[*]}"
    wait "${running_jobs[@]}"
    echo "DEBUG: All $mode-jobs finished!"
  else
    wait "${running_jobs[@]}"
  fi
}

function filter_token()
{
  IFS=""
  while read -r line ; do if [[ "${line:0:17}" != "#@#!NEXT-SUBJECT:" ]] ; then echo "$line" ; fi ; done
}


process_by_token_args=(
  "$timeout_read_token" "$debug" "$statusfile" "$parallel_pipelines" "$num_parallel_seg" "$num_parallel_surf"
  "$res_device" "$res_viewagg_device" "${run_fastsurfer[@]}" "|" "${POSITIONAL_FASTSURFER[@]}"
)
iterate_subjects_with_token_args=("$subject_start" "$subject_end" "${subjects[@]}")
if [[ "$parallel_pipelines" == 1 ]] ; then
  mode=both
  if [[ "$seg_only" == "true" ]] ; then mode=seg
  elif [[ "$surf_only" == "true" ]] ; then mode=surf
  fi
  if [[ "$num_parallel_seg" != "max" ]] ; then fail_bash_version_lt4 ; fi
  iterate_subjects_with_token "${iterate_subjects_with_token_args[@]}" | \
    process_by_token "$mode" "${process_by_token_args[@]}" | \
    filter_token
else
  # multiple pipelines
  if [[ "$num_parallel_seg" != "max" ]] || [[ "$num_parallel_surf" != "max" ]] ; then fail_bash_version_lt4 ; fi
  iterate_subjects_with_token "${iterate_subjects_with_token_args[@]}" | \
    process_by_token "seg" "${process_by_token_args[@]}" | \
    process_by_token "surf" "${process_by_token_args[@]}" | \
    filter_token
fi

# always exit successful
exit 0
