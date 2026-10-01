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

"""
Keep test/pipelinetest/data/chain.yaml covering everything the pipelinetest suite compares.

The suite sorts its comparisons by the position a file has in chain.yaml, so that the first
reported failure is the first divergence and the ones after it are its consequences. A file the
chain does not list sorts last, which would put a genuine first divergence at the bottom of the
report and quietly undo that.

This reads the data files and never runs the suite, so it does not belong in test/pipelinetest,
which cannot even be imported without a processed subject to compare. It is here rather than in
test/lint because it needs a yaml parser, and test/lint runs under
``uv run --no-project --with pytest``, where only the standard library is available.
"""

from pathlib import Path

import pytest
import yaml

DATA = Path(__file__).parents[1] / "pipelinetest" / "data"

IMAGE_SUFFIXES = (".nii", ".nii.gz", ".mgz")


def read_yaml(name: str) -> dict:
    with open(DATA / name) as fp:
        return yaml.safe_load(fp)


@pytest.fixture(scope="module")
def chain_files() -> list[str]:
    stages = read_yaml("chain.yaml")["stages"]
    return [entry["file"] for entry in stages]


def compared_files() -> set[str]:
    """Every file the suite compares, however it is configured."""
    segmentations = {f.stem for f in DATA.glob("*.yaml") if f.stem.endswith(IMAGE_SUFFIXES)}
    intensities = set(read_yaml("image.intensity.yaml")["thresholds"])
    surfaces = set(read_yaml("surface.geometry.yaml")["tolerances"])
    return segmentations | intensities | surfaces


def test_chain_covers_every_comparison(chain_files: list[str]):
    """Every compared file has a position, so the first failure is the first divergence."""
    missing = sorted(compared_files() - set(chain_files))
    assert missing == [], (
        f"{len(missing)} compared file(s) are absent from chain.yaml and would sort last, "
        f"hiding a real first divergence at the bottom of the report: {missing}"
    )


def test_chain_has_no_duplicates(chain_files: list[str]):
    """A file listed twice has two positions, and which one sorts is then arbitrary."""
    duplicates = sorted({f for f in chain_files if chain_files.count(f) > 1})
    assert duplicates == [], f"chain.yaml lists these more than once: {duplicates}"


def test_chain_entries_are_complete(chain_files: list[str]):
    """Both keys are required: the file to match on, the stage to name in the summary."""
    stages = read_yaml("chain.yaml")["stages"]
    incomplete = [entry for entry in stages if not entry.get("file") or not entry.get("stage")]
    assert incomplete == [], f"chain.yaml entries need a file and a stage: {incomplete}"
