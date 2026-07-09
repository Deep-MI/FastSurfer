#!/bin/bash

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

# Developer-oriented smoke test for the BIDS add-on.
#
# This wrapper bundles the lightweight checks needed to validate the BIDS support:
#   1. targeted pytest coverage for the BIDS discovery/routing code
#   2. OpenNeuro-based dry run against one cross-sectional and one longitudinal subject
#
# By default it only performs dry-run checks and does not execute FastSurfer.

set -euo pipefail

if [[ -z "${FASTSURFER_HOME:-}" ]]; then
  FASTSURFER_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." &> /dev/null && pwd)
fi

usage() {
  cat <<EOF
Usage:
  check_bids_addon.sh [--python <python>] [--work_dir <dir>] [--venv <dir>] [--skip_pytest] [--skip_dry_run]

Runs the developer smoke checks for the BIDS add-on:
  - pytest: test/quicktest/test_bids.py and test/quicktest/test_run_fastsurfer_bids.py
  - dry run: test/integration/test_bids_openneuro.sh with --seg_only

Options:
  --python <python>   Python executable to use. Default: python
  --work_dir <dir>    Output directory for the integration dry-run.
                      Default: /tmp/fastsurfer_bids_smoketest
  --venv <dir>        Activate this virtual environment before running checks.
  --skip_pytest       Skip the targeted pytest step.
  --skip_dry_run      Skip the OpenNeuro dry-run step.
  -h, --help          Show this help.

Examples:
  test/integration/check_bids_addon.sh --venv .venv
  test/integration/check_bids_addon.sh --venv .venv --work_dir /data/local/tmp_big
EOF
}

python_cmd="python"
work_dir="/tmp/fastsurfer_bids_smoketest"
venv_dir=""
skip_pytest="false"
skip_dry_run="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) python_cmd="$2" ; shift 2 ;;
    --work_dir) work_dir="$2" ; shift 2 ;;
    --venv) venv_dir="$2" ; shift 2 ;;
    --skip_pytest) skip_pytest="true" ; shift ;;
    --skip_dry_run) skip_dry_run="true" ; shift ;;
    -h|--help) usage ; exit 0 ;;
    *) echo "ERROR: Unknown option $1" ; usage ; exit 1 ;;
  esac
done

if [[ -n "$venv_dir" ]]; then
  # shellcheck disable=SC1090
  source "$venv_dir/bin/activate"
fi

cd "$FASTSURFER_HOME"

if [[ "$skip_pytest" != "true" ]]; then
  ref_dir="$work_dir/pytest_ref"
  subjects_dir="$work_dir/pytest_subjects"
  mkdir -p "$ref_dir/sub-dummy" "$subjects_dir"
  echo "=== Step 1/2: Targeted BIDS pytest checks ==="
  REF_DIR="$ref_dir" SUBJECTS_DIR="$subjects_dir" \
    "$python_cmd" -m pytest \
    test/quicktest/test_bids.py \
    test/quicktest/test_run_fastsurfer_bids.py -q
fi

if [[ "$skip_dry_run" != "true" ]]; then
  echo
  echo "=== Step 2/2: OpenNeuro BIDS dry-run smoke test ==="
  bash test/integration/test_bids_openneuro.sh \
    "$work_dir/output" \
    --bids_dir "$work_dir/bids" \
    -- --seg_only
fi

echo
echo "BIDS add-on smoke checks completed successfully."