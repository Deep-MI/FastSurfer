#!/bin/bash

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

# Regenerate requirements.txt, the pinned dependency set that pip/uv installs, the macOS package
# and `build.py --pinned_requirements` install from.
#
# This is the resolution the docker build runs from pyproject.toml, done once for Linux and macOS
# together (uv --universal), so a version is only pinned if it exists on both. PyTorch is resolved
# against its CPU index and its local suffix (+cpu) is dropped, so the file stays backend-neutral:
# the backend is chosen at install time with --torch-backend.

set -e
set -o pipefail

if [[ "$#" -gt 0 ]]
then
  echo "Usage: tools/update_requirements.sh"
  echo "  Rewrites requirements.txt from pyproject.toml. Needs uv."
  exit 1
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_version=$(python3 "$repo/tools/read_toml.py" --file "$repo/pyproject.toml" --key tool.python.version)

compiled=$(cd "$repo" && uv pip compile --quiet --no-header --no-annotate --universal --extra container \
  pyproject.toml --python-version "$python_version" --torch-backend cpu)

tmp_target=$(mktemp "$repo/requirements.txt.tmp.XXXXXX")
trap 'rm -f "$tmp_target"' EXIT
{
  echo "#"
  echo "# Pinned requirements for FastSurfer, generated from pyproject.toml by"
  echo "#"
  echo "#    tools/update_requirements.sh"
  echo "#"
  echo "# Resolved for Linux and macOS together with python $python_version, so every pin exists on both."
  echo "# PyTorch backend variants are not pinned; select one at install time, for example:"
  echo "#"
  echo "#    uv pip compile --torch-backend=cu128 requirements.txt | uv pip sync --torch-backend=cu128 -"
  echo "#    uv pip compile --torch-backend=cpu requirements.txt | uv pip sync --torch-backend=cpu -"
  echo "#"
  # drop the local version suffix (torch==2.7.1+cpu); forks that then name the same version for
  # every platform collapse into one unconditional pin
  echo "$compiled" | python3 -c '
import re
import sys

forks = {}  # package name -> [(version, marker)], in the order uv wrote them
for line in sys.stdin.read().splitlines():
    match = re.match(r"^([A-Za-z0-9_.\-]+)==([^+ ;]+)(\+[^ ;]+)? *(;.*)?$", line)
    if match:
        forks.setdefault(match.group(1), []).append((match.group(2), match.group(4)))
for name, entries in forks.items():
    if len(entries) > 1 and len({version for version, _ in entries}) == 1:
        print(f"{name}=={entries[0][0]}")
    else:
        for version, marker in entries:
            print(f"{name}=={version}" + (f" {marker}" if marker else ""))
'
} > "$tmp_target"
mv "$tmp_target" "$repo/requirements.txt"
trap - EXIT
echo "Wrote $repo/requirements.txt ($(grep -c '==' "$repo/requirements.txt") pins)."
