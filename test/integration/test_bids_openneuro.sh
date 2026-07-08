#!/bin/bash
#
# Copyright 2026 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
# End-to-end smoke test for run_fastsurfer_bids.py against a small, real subset
# of the OpenNeuro dataset ds004937 (https://doi.org/10.18112/openneuro.ds004937.v1.0.1).
#
# Downloads one single-session subject (-> exercises the cross-sectional /
# brun_fastsurfer.sh path) and one four-session subject (-> exercises the
# longitudinal / long_fastsurfer.sh path) into a local BIDS dataset, then runs
# run_fastsurfer_bids.py on both subjects in a single call, relying on its
# auto-detection to route each subject to the correct pipeline.
#
# By default this only prints the commands that would run (--dry_run). Pass
# --run --fs_license <file> to actually execute FastSurfer (requires the full
# fastsurfer[all] Python environment, a FreeSurfer license, and a GPU/CPU with
# enough time -- the longitudinal subject alone processes 4 timepoints).
#
# Usage:
#   test_bids_openneuro.sh <work_dir> [--run --fs_license <file>] [-- <extra run_fastsurfer_bids.py flags>]

set -euo pipefail

if [[ -z "${FASTSURFER_HOME:-}" ]]; then
  FASTSURFER_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." &> /dev/null && pwd)
fi

work_dir="${1:?Usage: test_bids_openneuro.sh <work_dir> [--run --fs_license <file>] [-- <extra flags>]}"
shift || true

dry_run="--dry_run"
fs_license=()
extra_flags=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) dry_run="" ; shift ;;
    --fs_license) fs_license=(--fs_license "$2") ; shift 2 ;;
    --) shift ; extra_flags=("$@") ; break ;;
    *) echo "ERROR: Unknown option $1" ; exit 1 ;;
  esac
done

if [[ -z "$dry_run" && ${#fs_license[@]} -eq 0 ]]; then
  echo "ERROR: --run requires --fs_license <file>."
  exit 1
fi

bids_dir="$work_dir/bids"
output_dir="$work_dir/output"

echo "=== Step 1/2: Fetching OpenNeuro BIDS subset into $bids_dir ==="
"$(dirname "${BASH_SOURCE[0]}")/fetch_openneuro_bids_subset.sh" "$bids_dir"

echo
echo "=== Step 2/2: Running run_fastsurfer_bids.py (participant level) ==="
cross_sub="${CROSS_SUB:-sub-119BPAF161002}"
long_sub="${LONG_SUB:-sub-119BPAF161001}"

cmd=("$FASTSURFER_HOME/run_fastsurfer_bids.py"
     "$bids_dir" "$output_dir" participant
     --participant_label "$cross_sub" "$long_sub"
     --skip_bids_validator
     $dry_run
     "${fs_license[@]}")
if [[ ${#extra_flags[@]} -gt 0 ]]; then
  cmd+=(-- "${extra_flags[@]}")
fi

echo "+ ${cmd[*]}"
"${cmd[@]}"

echo
echo "Expected: $cross_sub (ses-1 only) routed through brun_fastsurfer.sh,"
echo "          $long_sub (ses-1..ses-4) routed through long_fastsurfer.sh."
