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
# The two sessions are the point. Each is processed as its own cross-sectional case, so a
# multi-session subject is what shows the flat sub-<label>_ses-<label> output naming actually
# happening, and shows the two sessions do not land on top of each other.
#
# Usage:
#   test/integration/openneuro_check.sh <work_dir> --fs_license <license>   # full run, hours
#   test/integration/openneuro_check.sh <work_dir> -- --seg_only            # ~minutes, no license
#   test/integration/openneuro_check.sh <work_dir> --dry                    # discovery only
#
# Anything after -- is passed to run_fastsurfer_bids.py, and so on to run_fastsurfer.sh.
#
# Env: SUBJECT pins the subject instead of drawing one, SESSIONS pins its sessions,
# OPENNEURO_ACCESSION picks another dataset, WANT_SESSIONS how many sessions to take.

set -euo pipefail

FASTSURFER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPECTED_FILES="$FASTSURFER_HOME/test/pipelinetest/data/expected-files.yaml"

# ds004937 (https://doi.org/10.18112/openneuro.ds004937.v1.0.1), public, no login, ~12MB a session.
# Its subjects have four anat sessions, all T1w acq-mprage, and no T2w.
#
# The subject is drawn at random from participants.tsv rather than fixed, so that repeated runs
# cover the dataset instead of the same images over and over. It is printed, and SUBJECT= pins it,
# which is what to do when a run fails and has to be repeated on the same data.
ACCESSION="${OPENNEURO_ACCESSION:-ds004937}"
SUBJECT="${SUBJECT:-}"
SESSIONS="${SESSIONS:-}"
SESSION_CANDIDATES="${SESSION_CANDIDATES:-ses-1 ses-2 ses-3 ses-4}"
WANT_SESSIONS="${WANT_SESSIONS:-2}"
BASE_URL="https://s3.amazonaws.com/openneuro.org/${ACCESSION}"

work_dir="" ; fs_license="" ; dry="false" ; passthrough=()
while [[ $# -gt 0 ]] ; do
  case "$1" in
    --fs_license) fs_license="$2" ; shift 2 ;;
    --dry|--dry_run) dry="true" ; shift ;;
    --help|-h) sed -n '/^# Run run_fastsurfer_bids.py end to end/,/^$/p' "${BASH_SOURCE[0]}" ; exit 0 ;;
    --) shift ; passthrough=("$@") ; break ;;
    -*) echo "ERROR: unknown option $1" >&2 ; exit 1 ;;
    *) work_dir="$1" ; shift ;;
  esac
done
if [[ -z "$work_dir" ]] ; then
  echo "ERROR: give a work directory, e.g. $0 /data/tmp/bids_check --fs_license ~/license.txt" >&2
  exit 1
fi

bids_dir="$work_dir/bids"
out_dir="$work_dir/out"

# ----------------------------------------------------------------------------------------------
# 1. fetch

fetch() { # $1: key relative to the dataset root
  local dest="$bids_dir/$1"
  if [[ -s "$dest" ]] ; then echo "  have  $1" ; return ; fi
  mkdir -p "$(dirname "$dest")"
  echo "  get   $1"
  curl -sSfL "$BASE_URL/$1" -o "$dest"
}

t1_key() { echo "$1/$2/anat/$1_$2_acq-mprage_T1w.nii.gz" ; } # $1 subject, $2 session

# The bucket serves files, not listings, so what a subject has is found by asking for it. A HEAD
# costs nothing next to the download and keeps a subject with fewer sessions from aborting the run.
sessions_of() { # $1 subject, prints the candidate sessions it actually has, up to WANT_SESSIONS
  local found=() ses
  for ses in $SESSION_CANDIDATES ; do
    if curl -sSfLI --max-time 30 "$BASE_URL/$(t1_key "$1" "$ses")" -o /dev/null ; then
      found+=("$ses")
      if [[ "${#found[@]}" -ge "$WANT_SESSIONS" ]] ; then break ; fi
    fi
  done
  echo "${found[@]:-}"
}

echo "== fetching $ACCESSION into $bids_dir"
for key in dataset_description.json participants.tsv participants.json README CHANGES ; do
  fetch "$key" || echo "  skip  $key (not in this dataset)"
done

if [[ -z "$SUBJECT" ]] ; then
  # an already fetched subject wins over a new draw, so that --dry, then --seg_only, then the full
  # run on one work directory all speak about the same images
  downloaded=("$bids_dir"/sub-*/)
  if [[ -d "${downloaded[0]}" ]] ; then
    SUBJECT="$(basename "${downloaded[0]}")"
    echo "== reusing $SUBJECT, which is already in $bids_dir, so that repeated runs on one work"
    echo "   directory stay on the same images. For a new random draw: use a work directory that"
    echo "   does not exist yet, or pass SUBJECT=<sub-label> to choose."
  fi
fi

if [[ -z "$SUBJECT" ]] ; then
  # participant_id is the first column; the header and a trailing \r (the file may be CRLF) go
  candidates=()
  while IFS=$'\t' read -r id _ ; do
    id="${id%$'\r'}"
    if [[ "$id" == sub-* ]] ; then candidates+=("$id") ; fi
  done < "$bids_dir/participants.tsv"
  [[ "${#candidates[@]}" -gt 0 ]] || { echo "ERROR: no sub-* in participants.tsv" >&2 ; exit 1 ; }

  for attempt in 1 2 3 4 5 ; do
    SUBJECT="${candidates[RANDOM % ${#candidates[@]}]}"
    SESSIONS="$(sessions_of "$SUBJECT")"
    if [[ -n "$SESSIONS" ]] ; then break ; fi
    echo "  $SUBJECT has none of: $SESSION_CANDIDATES, drawing again"
    SUBJECT=""
  done
  [[ -n "$SUBJECT" ]] || { echo "ERROR: no usable subject drawn in 5 attempts" >&2 ; exit 1 ; }
  echo "== drew $SUBJECT at random from ${#candidates[@]} participants"
  echo "   repeat this exact run with: SUBJECT=$SUBJECT SESSIONS=\"$SESSIONS\" $0 $work_dir ..."
fi

[[ -n "$SESSIONS" ]] || SESSIONS="$(sessions_of "$SUBJECT")"
[[ -n "$SESSIONS" ]] || { echo "ERROR: $SUBJECT has no T1w in $SESSION_CANDIDATES" >&2 ; exit 1 ; }

echo "== fetching $SUBJECT ($SESSIONS)"
for ses in $SESSIONS ; do
  fetch "$(t1_key "$SUBJECT" "$ses")"
  fetch "$SUBJECT/$ses/anat/${SUBJECT}_${ses}_acq-mprage_T1w.json"
done

# ----------------------------------------------------------------------------------------------
# 2. discovery and routing, before committing to a run that takes hours

echo ""
echo "== what the entry point discovers and would run"
"$FASTSURFER_HOME/run_fastsurfer_bids.py" "$bids_dir" "$out_dir" participant \
  --skip_bids_validator --dry ${fs_license:+--fs_license "$fs_license"} \
  ${passthrough:+-- "${passthrough[@]}"}

if [[ "$dry" == "true" ]] ; then
  echo ""
  echo "== --dry given, stopping before the run"
  exit 0
fi

# ----------------------------------------------------------------------------------------------
# 3. the real run

if [[ -z "$fs_license" ]] && [[ " ${passthrough[*]:-} " != *" --seg_only "* ]] ; then
  echo "ERROR: surfaces need a FreeSurfer license. Pass --fs_license <file>, or -- --seg_only" >&2
  exit 1
fi
if ! python3 -c "import torch" 2>/dev/null ; then
  echo "ERROR: no torch in the python3 on PATH. Activate the FastSurfer environment first, so" >&2
  echo "       this fails now rather than several minutes into the run." >&2
  exit 1
fi

echo ""
echo "== running (this is the slow part)"
start=$SECONDS
"$FASTSURFER_HOME/run_fastsurfer_bids.py" "$bids_dir" "$out_dir" participant \
  --skip_bids_validator ${fs_license:+--fs_license "$fs_license"} \
  ${passthrough:+-- "${passthrough[@]}"}
echo "== run finished in $(( (SECONDS - start) / 60 )) minutes"

# ----------------------------------------------------------------------------------------------
# 4. check the outputs
#
# The expected file list is test/pipelinetest/data/expected-files.yaml rather than one written
# here, so this says what the pipeline test says a finished subject holds. Missing files are
# reported, not fatal: a --seg_only run legitimately has no surfaces, and this dataset has no T2w.

echo ""
echo "== checking $out_dir"
status=0

# The command is run by check rather than before it: under `set -e` a bare failing test would
# kill the script, so every FAIL would exit silently instead of being reported.
check() { # $1: description, rest: the command whose success is the check
  local description="$1" ; shift
  if "$@" > /dev/null 2>&1 ; then echo "  ok    $description"
  else echo "  FAIL  $description" ; status=1
  fi
}

check "BIDS derivatives dataset_description.json" test -f "$out_dir/dataset_description.json"
check "  and it is valid json" \
  python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$out_dir/dataset_description.json"
check "generated subject list kept" test -f "$out_dir/scripts/bids_subjects.txt"

for ses in $SESSIONS ; do
  subject_dir="$out_dir/${SUBJECT}_${ses}"
  check "flat output directory ${SUBJECT}_${ses}" test -d "$subject_dir"
  [[ -d "$subject_dir" ]] || continue

  # the archival copy of the input has to be this session's image, not the other one's
  source_t1="$bids_dir/$SUBJECT/$ses/anat/${SUBJECT}_${ses}_acq-mprage_T1w.nii.gz"
  archived=("$subject_dir"/mri/orig/001.*)
  if [[ -f "${archived[0]}" ]] && [[ "${archived[0]}" == *.nii.gz ]] ; then
    check "  mri/orig/001.nii.gz is this session's input" \
      test "$(md5sum < "$source_t1")" = "$(md5sum < "${archived[0]}")"
  else
    echo "  info  mri/orig/001 is ${archived[0]##*/}, not a byte copy, checksum not compared"
  fi

  missing=0 present=0
  while IFS= read -r pattern ; do
    [[ -n "$pattern" ]] || continue
    # compgen so a pattern that matches nothing is a miss rather than the literal string
    if compgen -G "$subject_dir/$pattern" > /dev/null ; then present=$((present + 1))
    else missing=$((missing + 1)) ; echo "    missing: $pattern"
    fi
  done < <(sed -nE 's/^[[:space:]]*-[[:space:]]*"(.*)"[[:space:]]*$/\1/p' "$EXPECTED_FILES")
  echo "  info  ${SUBJECT}_${ses}: $present of $((present + missing)) expected outputs present"
done

echo ""
if [[ "$status" == "0" ]] ; then echo "== the BIDS-specific checks passed"
else echo "== something the entry point is responsible for is wrong, see FAIL above" ; fi
exit "$status"
