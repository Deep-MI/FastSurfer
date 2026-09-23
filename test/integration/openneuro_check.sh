#!/bin/bash
#
# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
#
# Run run_fastsurfer_bids.py end to end against real data from OpenNeuro.
#
# Not a CI test and not wired into any workflow: it downloads over the network and a full run
# takes hours per session. It is the check to do by hand before trusting the entry point on a
# real dataset, which is the one thing the unit tests cannot do, since they run on empty files.
#
# A case is a BIDS layout the entry point has to get right, not a dataset anyone needs to know
# beforehand: run the script and pick one. The accession is printed with the run, so a failure can
# be repeated on the same images.
#
# Usage:
#   test/integration/openneuro_check.sh <work_dir>                     # ask which case, how far
#   test/integration/openneuro_check.sh <work_dir> --case sessions --depth seg
#   test/integration/openneuro_check.sh <work_dir> --all --depth dry
#   test/integration/openneuro_check.sh <work_dir> --all --depth full --fs_license <file>
#
# Depths: dry (discovery and routing only), seg (segmentation, minutes per session, no license),
# full (surfaces too, hours per session, needs --fs_license).
#
# On a cluster, --slurm submits the cases through srun_fastsurfer.sh, with its options after the
# -- (--partition, --work, --singularity_image). The jobs outlive this script, so it stops after
# submitting; --check_only checks the output directory once they are done:
#
#   test/integration/openneuro_check.sh <work_dir> --all --depth seg --slurm \
#       -- --partition <gpu> --work <scratch> --singularity_image <fastsurfer.sif>
#   test/integration/openneuro_check.sh <work_dir> --all --depth seg --check_only
#
# Anything after -- is passed to run_fastsurfer_bids.py, and so on to run_fastsurfer.sh.
#
# Env: SUBJECT and SESSIONS pin the drawn subject of the 'sessions' case, OPENNEURO_ACCESSION
# picks another dataset for it, WANT_SESSIONS how many of its sessions to take.

set -euo pipefail

FASTSURFER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPECTED_FILES="$FASTSURFER_HOME/test/pipelinetest/data/expected-files.yaml"

CASES="sessions mixed_t2 no_sessions"

# What each case is here to check. The menu shows this rather than the accession: the layout is the
# point, and naming the dataset only invites picking by familiarity or download size.
case_description() { # $1: case
  case "$1" in
    sessions)    echo "several sessions of one subject, a T1w in each, no T2w" ;;
    mixed_t2)    echo "a T2w in some sessions only, and one session with a T2w but no T1w" ;;
    no_sessions) echo "no session level: sub-<label>/anat directly, which BIDS allows" ;;
  esac
}

case_accession() { # $1: case
  case "$1" in
    sessions)    echo "${OPENNEURO_ACCESSION:-ds004937}" ;;
    mixed_t2)    echo "ds000031" ;;
    no_sessions) echo "ds002785" ;;
  esac
}

# The options the layout is about, as opposed to the ones every case takes.
case_options() { # $1: case
  case "$1" in
    mixed_t2) echo "--use_t2" ;;
    *) echo "" ;;
  esac
}

# The images to fetch, one key per line, relative to the dataset root. Fixed for the small cases so
# that everyone runs the same images; the 'sessions' case draws them instead, see draw_keys.
case_keys() { # $1: case
  case "$1" in
    mixed_t2)
      # ses-003 has a T1w only, ses-012 has both, and ses-016 has a T2w but no T1w, so it has to be
      # skipped rather than processed or refused
      echo "sub-01/ses-003/anat/sub-01_ses-003_T1w.nii.gz"
      echo "sub-01/ses-012/anat/sub-01_ses-012_T1w.nii.gz"
      echo "sub-01/ses-012/anat/sub-01_ses-012_T2w.nii.gz"
      echo "sub-01/ses-016/anat/sub-01_ses-016_T2w.nii.gz"
      ;;
    no_sessions)
      echo "sub-0001/anat/sub-0001_T1w.nii.gz"
      ;;
  esac
}

work_dir="" ; fs_license="" ; depth="" ; selected="" ; slurm="false" ; check_only="false"
passthrough=()
while [[ $# -gt 0 ]] ; do
  case "$1" in
    --fs_license) fs_license="$2" ; shift 2 ;;
    --slurm) slurm="true" ; shift ;;
    --check_only) check_only="true" ; shift ;;
    --case) selected="$2" ; shift 2 ;;
    --all) selected="$CASES" ; shift ;;
    --depth) depth="$2" ; shift 2 ;;
    --dry|--dry_run) depth="dry" ; shift ;;
    --help|-h) sed -n '/^# Run run_fastsurfer_bids.py end to end/,/^$/p' "${BASH_SOURCE[0]}" ; exit 0 ;;
    --) shift ; passthrough=("$@") ; break ;;
    -*) echo "ERROR: unknown option $1" >&2 ; exit 1 ;;
    *) work_dir="$1" ; shift ;;
  esac
done
if [[ -z "$work_dir" ]] ; then
  echo "ERROR: give a work directory, e.g. $0 /data/tmp/bids_check --all --depth seg" >&2
  exit 1
fi
for case_name in $selected ; do
  if [[ " $CASES " != *" $case_name "* ]] ; then
    echo "ERROR: unknown case $case_name, pick from: $CASES" >&2 ; exit 1
  fi
done

# ----------------------------------------------------------------------------------------------
# 0. what to run
#
# The menu is only offered where someone is there to answer it: in a job or a pipe a missing --case
# is an error rather than a prompt nobody sees.

ask() { # $1: prompt, $2: options, $3: default; prints the chosen option
  local reply choice option i=1
  echo "" >&2
  for option in $2 ; do
    if [[ "$option" == "$3" ]] ; then echo "  $i) $option (default)" >&2
    else echo "  $i) $option" >&2 ; fi
    i=$((i + 1))
  done
  read -r -p "$1 [$3]: " reply
  if [[ -z "$reply" ]] ; then echo "$3" ; return ; fi
  i=1
  for option in $2 ; do
    if [[ "$reply" == "$i" ]] || [[ "$reply" == "$option" ]] ; then echo "$option" ; return ; fi
    i=$((i + 1))
  done
  echo "ERROR: $reply is not one of: $2" >&2 ; exit 1
}

if [[ -z "$selected" ]] ; then
  if [[ ! -t 0 ]] ; then
    echo "ERROR: no terminal to ask which case to run, pass --case <name> or --all:" >&2
    for case_name in $CASES ; do
      printf '  %-12s %s\n' "$case_name" "$(case_description "$case_name")" >&2
    done
    exit 1
  fi
  echo "Which BIDS layout should be checked?"
  for case_name in $CASES ; do
    printf '  %-12s %s\n' "$case_name" "$(case_description "$case_name")"
  done
  choice="$(ask "  case:" "$CASES all" "sessions")"
  if [[ "$choice" == "all" ]] ; then selected="$CASES" ; else selected="$choice" ; fi
fi

if [[ -z "$depth" ]] ; then
  if [[ ! -t 0 ]] ; then
    echo "ERROR: no terminal to ask how far to run, pass --depth dry|seg|full." >&2 ; exit 1
  fi
  echo ""
  echo "How far should it run?"
  echo "  dry   discovery and routing only, nothing is processed"
  echo "  seg   segmentation only, minutes per session, no license needed"
  echo "  full  surfaces too, hours per session, needs --fs_license"
  depth="$(ask "  depth:" "dry seg full" "seg")"
fi

case "$depth" in
  dry) ;;
  seg) passthrough+=("--seg_only") ;;
  full)
    if [[ -z "$fs_license" ]] ; then
      echo "ERROR: --depth full needs a FreeSurfer license, pass --fs_license <file>." >&2 ; exit 1
    fi ;;
  *) echo "ERROR: unknown depth $depth, pick dry, seg or full" >&2 ; exit 1 ;;
esac

# not with --slurm: there the processing happens in the container srun binds on the compute node,
# so the torch of the submitting shell says nothing
if [[ "$depth" != "dry" ]] && [[ "$slurm" == "false" ]] && ! python3 -c "import torch" 2>/dev/null ; then
  echo "ERROR: no torch in the python3 on PATH. Activate the FastSurfer environment first, so" >&2
  echo "       this fails now rather than several minutes into the run." >&2
  exit 1
fi

# ----------------------------------------------------------------------------------------------
# 1. fetch

fetch() { # $1: bids_dir, $2: dataset base url, $3: key relative to the dataset root
  local dest="$1/$3"
  if [[ -s "$dest" ]] ; then echo "  have  $3" ; return ; fi
  mkdir -p "$(dirname "$dest")"
  echo "  get   $3"
  curl -sSfL "$2/$3" -o "$dest"
}

# The bucket serves files, not listings, so what a subject has is found by asking for it. A HEAD
# costs nothing next to the download and keeps a subject with fewer sessions from aborting the run.
sessions_of() { # $1: base url, $2: subject; prints the candidates it has, up to WANT_SESSIONS
  local found=() ses
  for ses in ${SESSION_CANDIDATES:-ses-1 ses-2 ses-3 ses-4} ; do
    if curl -sSfLI --max-time 30 "$1/$2/$ses/anat/$2_${ses}_acq-mprage_T1w.nii.gz" -o /dev/null ; then
      found+=("$ses")
      if [[ "${#found[@]}" -ge "${WANT_SESSIONS:-2}" ]] ; then break ; fi
    fi
  done
  echo "${found[@]:-}"
}

# The 'sessions' case draws its subject rather than fixing one, so that repeated runs cover the
# dataset instead of the same images over and over. The draw is printed and SUBJECT= pins it, which
# is what to do when a run fails and has to be repeated on the same data. A subject already fetched
# wins over a new draw, so dry, then seg, then full on one work directory speak about one subject.
draw_keys() { # $1: bids_dir, $2: base url; prints a '# pin' line, then the keys
  local subject="${SUBJECT:-}" sessions="${SESSIONS:-}" candidates=() downloaded id ses attempt
  if [[ -z "$subject" ]] ; then
    downloaded=("$1"/sub-*/)
    if [[ -d "${downloaded[0]}" ]] ; then subject="$(basename "${downloaded[0]}")" ; fi
  fi
  if [[ -z "$subject" ]] ; then
    # participant_id is the first column; the header and a trailing \r (the file may be CRLF) go
    while IFS=$'\t' read -r id _ ; do
      id="${id%$'\r'}"
      if [[ "$id" == sub-* ]] ; then candidates+=("$id") ; fi
    done < "$1/participants.tsv"
    [[ "${#candidates[@]}" -gt 0 ]] || { echo "ERROR: no sub-* in participants.tsv" >&2 ; exit 1 ; }
    for attempt in 1 2 3 4 5 ; do
      subject="${candidates[RANDOM % ${#candidates[@]}]}"
      sessions="$(sessions_of "$2" "$subject")"
      if [[ -n "$sessions" ]] ; then break ; fi
      subject=""
    done
    [[ -n "$subject" ]] || { echo "ERROR: no usable subject drawn in 5 attempts" >&2 ; exit 1 ; }
  fi
  [[ -n "$sessions" ]] || sessions="$(sessions_of "$2" "$subject")"
  [[ -n "$sessions" ]] || { echo "ERROR: $subject has no T1w image" >&2 ; exit 1 ; }
  echo "# SUBJECT=$subject SESSIONS=\"$sessions\""
  for ses in $sessions ; do
    echo "$subject/$ses/anat/${subject}_${ses}_acq-mprage_T1w.nii.gz"
    echo "$subject/$ses/anat/${subject}_${ses}_acq-mprage_T1w.json"
  done
}

fetch_case() { # $1: case, $2: bids_dir
  local accession base key
  accession="$(case_accession "$1")"
  base="https://s3.amazonaws.com/openneuro.org/$accession"
  echo "== $1: $(case_description "$1")"
  echo "   $accession into $2"
  for key in dataset_description.json participants.tsv participants.json README CHANGES ; do
    fetch "$2" "$base" "$key" || echo "  skip  $key (not in this dataset)"
  done
  if [[ "$1" == "sessions" ]] ; then
    while IFS= read -r key ; do
      case "$key" in
        "#"*) echo "   repeat this exact run with: ${key#\# }" ;;
        *) fetch "$2" "$base" "$key" ;;
      esac
    done < <(draw_keys "$2" "$base")
  else
    while IFS= read -r key ; do
      [[ -n "$key" ]] && fetch "$2" "$base" "$key"
    done < <(case_keys "$1")
  fi
}

# ----------------------------------------------------------------------------------------------
# 2. run

run_case() { # $1: case, $2: bids_dir, $3: out_dir
  # an array rather than ${var:+...}: that form is unquoted, so any argument holding a space is
  # split again on the way into the command
  local args=("$2" "$3" participant --skip_bids_validator) option
  for option in $(case_options "$1") ; do args+=("$option") ; done
  if [[ "$slurm" == "true" ]] ; then args+=(--slurm) ; fi
  if [[ "$depth" == "dry" ]] ; then args+=(--dry) ; fi
  if [[ -n "$fs_license" ]] ; then args+=(--fs_license "$fs_license") ; fi
  if [[ "${#passthrough[@]}" -gt 0 ]] ; then args+=(-- "${passthrough[@]}") ; fi
  "$FASTSURFER_HOME/run_fastsurfer_bids.py" "${args[@]}"
}

# ----------------------------------------------------------------------------------------------
# 3. check the outputs
#
# What to expect is read back from the subject list the entry point wrote, so this checks whatever
# the case discovered rather than one hardcoded subject: every case it decided to process has to
# have its own directory, holding its own image. The expected file list is test/pipelinetest/data/
# expected-files.yaml rather than one written here, so this says what the pipeline test says a
# finished subject holds. Missing files are reported, not fatal: a seg run legitimately has no
# surfaces, and not every case has a T2.

status=0

# The command is run by check rather than before it: under `set -e` a bare failing test would kill
# the script, so every FAIL would exit silently instead of being reported.
check() { # $1: description, rest: the command whose success is the check
  local description="$1" ; shift
  if "$@" > /dev/null 2>&1 ; then echo "  ok    $description"
  else echo "  FAIL  $description" ; status=1
  fi
}

check_case() { # $1: out_dir
  local subject_list="$1/bids_subjects.txt" line output_id source_t1 subject_dir
  local archived missing present pattern

  echo ""
  echo "== checking $1"
  check "dataset_description.json" test -f "$1/dataset_description.json"
  check "  and it is valid json" \
    python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$1/dataset_description.json"
  check "generated subject list kept" test -f "$subject_list"
  [[ -f "$subject_list" ]] || { status=1 ; return ; }

  while IFS= read -r line ; do
    [[ -n "$line" ]] || continue
    output_id="${line%%=*}"
    source_t1="${line#*=}"
    source_t1="${source_t1%% --t2 *}"                          # the T2 of the line is not this check
    source_t1="${source_t1#\'}" ; source_t1="${source_t1%\'}"  # the list is shell-quoted for brun
    subject_dir="$1/$output_id"
    check "output directory $output_id" test -d "$subject_dir"
    [[ -d "$subject_dir" ]] || continue

    # the archival copy of the input has to be this session's image, not another session's
    archived=("$subject_dir"/mri/orig/001.*)
    if [[ -f "${archived[0]}" ]] && [[ "${archived[0]}" == *.nii.gz ]] ; then
      # cmp rather than md5sum, which macOS does not ship
      check "  mri/orig/001.nii.gz is this session's input" cmp -s "$source_t1" "${archived[0]}"
    else
      echo "  info  mri/orig/001 is ${archived[0]##*/}, not a byte copy, checksum not compared"
    fi

    missing=0 present=0
    while IFS= read -r pattern ; do
      [[ -n "$pattern" ]] || continue
      # compgen so a pattern that matches nothing is a miss rather than the literal string
      if compgen -G "$subject_dir/$pattern" > /dev/null ; then present=$((present + 1))
      else missing=$((missing + 1))
      fi
    done < <(sed -nE 's/^[[:space:]]*-[[:space:]]*"(.*)"[[:space:]]*$/\1/p' "$EXPECTED_FILES")
    echo "  info  $output_id: $present of $((present + missing)) expected outputs present"
  done < "$subject_list"
}

# ----------------------------------------------------------------------------------------------
# 4. each selected case in turn

for case_name in $selected ; do
  bids_dir="$work_dir/$case_name/bids"
  out_dir="$work_dir/$case_name/out"

  if [[ "$check_only" == "true" ]] ; then
    check_case "$out_dir"
    continue
  fi

  echo ""
  fetch_case "$case_name" "$bids_dir"

  echo ""
  echo "== running $case_name at depth $depth"
  start=$SECONDS
  run_case "$case_name" "$bids_dir" "$out_dir"
  if [[ "$depth" == "dry" ]] ; then continue ; fi
  if [[ "$slurm" == "true" ]] ; then
    # the jobs outlive this script, so there is nothing to check yet
    echo "== $case_name submitted. Once the jobs are done, check the output with:"
    echo "   $0 $work_dir --case $case_name --depth $depth --check_only"
    continue
  fi
  echo "== $case_name finished in $(( (SECONDS - start) / 60 )) minutes"

  check_case "$out_dir"
done

echo ""
if [[ "$status" == "0" ]] ; then echo "== the BIDS-specific checks passed"
else echo "== something the entry point is responsible for is wrong, see FAIL above" ; fi
exit "$status"
