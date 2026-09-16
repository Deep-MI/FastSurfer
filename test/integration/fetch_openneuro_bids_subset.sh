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
# Fetches a small subset of subjects/sessions from the public OpenNeuro dataset
# ds004937 (https://doi.org/10.18112/openneuro.ds004937.v1.0.1) to assemble a
# minimal, valid BIDS dataset for testing run_fastsurfer_bids.py.
#
# The dataset has no T2w images and every subject has 4 anat sessions
# (ses-1..ses-4, all T1w, acq-mprage). To exercise both the cross-sectional and
# the longitudinal code path of run_fastsurfer_bids.py against real data with a
# single download, this script fetches:
#   - CROSS_SUB (default: sub-119BPAF161002): only ses-1  -> single session
#   - LONG_SUB  (default: sub-119BPAF161001): ses-1..ses-4 -> multi session
#
# Usage: fetch_openneuro_bids_subset.sh <target_bids_dir>
#
# Env overrides: OPENNEURO_ACCESSION, CROSS_SUB, LONG_SUB, LONG_SESSIONS

set -euo pipefail

target="${1:?Usage: fetch_openneuro_bids_subset.sh <target_bids_dir>}"

ACCESSION="${OPENNEURO_ACCESSION:-ds004937}"
CROSS_SUB="${CROSS_SUB:-sub-119BPAF161002}"
LONG_SUB="${LONG_SUB:-sub-119BPAF161001}"
LONG_SESSIONS="${LONG_SESSIONS:-ses-1 ses-2 ses-3 ses-4}"
BASE_URL="https://s3.amazonaws.com/openneuro.org/${ACCESSION}"

mkdir -p "$target"

download() {
  # 1: key relative to dataset root, e.g. sub-X/ses-1/anat/sub-X_ses-1_acq-mprage_T1w.nii.gz
  local key="$1"
  local dest="$target/$key"
  mkdir -p "$(dirname "$dest")"
  if [[ -s "$dest" ]]; then
    echo "  already present: $key"
    return
  fi
  echo "  fetching: $key"
  curl -sSfL "$BASE_URL/$key" -o "$dest"
}

echo "Fetching dataset-level BIDS files for $ACCESSION..."
download "dataset_description.json"
download "participants.tsv"
download "participants.json"
download "README"
download "CHANGES"

download_session_anat() {
  # 1: subject id (sub-X), 2: session id (ses-Y)
  local sub="$1" ses="$2"
  local t1_base="${sub}_${ses}_acq-mprage_T1w"
  download "$sub/$ses/anat/${t1_base}.nii.gz"
  download "$sub/$ses/anat/${t1_base}.json"
}

echo "Fetching cross-sectional subject $CROSS_SUB (single session, ses-1)..."
download_session_anat "$CROSS_SUB" "ses-1"

echo "Fetching longitudinal subject $LONG_SUB (sessions: $LONG_SESSIONS)..."
for ses in $LONG_SESSIONS; do
  download_session_anat "$LONG_SUB" "$ses"
done

echo "Done. BIDS subset written to $target"
echo "  Cross-sectional test subject: $CROSS_SUB (ses-1 only)"
echo "  Longitudinal test subject:    $LONG_SUB ($LONG_SESSIONS)"
