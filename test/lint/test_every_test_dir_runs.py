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

"""
Guard that every test directory is actually run by CI.

A directory that no workflow names is not a red test, it is a silent absence: CI stays green
while the tests it should have run never execute. That has happened twice. A matrix whose
``include`` entries carry no key from the matrix itself collapses into one job rather than
expanding into several, dropping every directory but the last, and a directory that is renamed
or added without touching a workflow is simply never picked up.

No yaml parser here on purpose: this runs under ``uv run --no-project --with pytest``, so only
the standard library is available, which is the same reason test_python_version.py parses with
a regex.
"""

import re
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent
TEST_ROOT = FASTSURFER_HOME / "test"
CI_FILES = sorted(
    list((FASTSURFER_HOME / ".github" / "workflows").glob("*.y*ml"))
    + list((FASTSURFER_HOME / ".github" / "actions").glob("*/action.y*ml"))
)

# `test/<name>`, optionally with the name supplied by a matrix, as in `test/${{ matrix.tests }}`
_PYTEST_PATH = re.compile(r"test/(\$\{\{\s*matrix\.(\w[\w-]*)\s*\}\}|[\w-]+)")
# a matrix dimension written inline, e.g. `tests: [image, shell]`
_MATRIX_LIST = re.compile(r"^\s*([\w-]+):\s*\[([^\]]*)\]\s*$", re.M)


def _block(text: str, header: str) -> str:
    """
    Return the body of a `header:` block, ending where the indentation returns to its level.

    Scoping matters: `branches: [dev]` under `on:` is the same shape as a matrix dimension, so
    searching the whole file would count it as one.
    """
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == header]
    if not starts:
        return ""
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = []
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break  # dedented back out of the block
        body.append(line)
    return "\n".join(body)


def _matrix_values(text: str, key: str) -> list[str]:
    """Collect every value a matrix dimension can take in one CI file."""
    values = []
    for name, items in _MATRIX_LIST.findall(_block(text, "matrix:")):
        if name == key:
            values += [item.strip().strip("\"'") for item in items.split(",") if item.strip()]
    return values


def _matrix_dimensions(text: str) -> set[str]:
    """The keys the matrix itself lists, which are what an include entry can filter on."""
    return {name for name, _ in _MATRIX_LIST.findall(_block(text, "matrix:"))}


def _include_keys(text: str) -> set[str]:
    """Collect the keys used inside a matrix's `include:` block, and nothing after it."""
    keys = set()
    for line in _block(text, "include:").splitlines():
        match = re.match(r"\s*-?\s*([\w-]+):", line)
        if match:
            keys.add(match.group(1))
    return keys


def _directories_ci_runs() -> set[str]:
    """Resolve every test/<name> any workflow or action passes to pytest."""
    covered: set[str] = set()
    for ci_file in CI_FILES:
        text = ci_file.read_text()
        for line in text.splitlines():
            if "pytest" not in line:
                continue
            for literal, matrix_key in _PYTEST_PATH.findall(line):
                if matrix_key:
                    covered.update(_matrix_values(text, matrix_key))
                else:
                    covered.add(literal)
    return covered


def _test_directories() -> set[str]:
    """Every test/<name> that holds tests, which is what CI has to reach."""
    return {
        directory.name
        for directory in TEST_ROOT.iterdir()
        if directory.is_dir() and any(directory.glob("test_*.py"))
    }


def test_ci_files_were_found() -> None:
    """A glob that silently matches nothing would make every other check here vacuous."""
    assert CI_FILES, f"no workflow or action files under {FASTSURFER_HOME / '.github'}"
    assert _directories_ci_runs(), "no pytest invocation found in any workflow or action"


@pytest.mark.parametrize("directory", sorted(_test_directories()))
def test_directory_is_run_by_ci(directory: str) -> None:
    """Every test/<name> has to be named by some workflow, directly or through a matrix."""
    covered = _directories_ci_runs()
    assert directory in covered, (
        f"test/{directory} holds tests but no workflow or action runs it, so its failures cannot "
        f"reach CI. The directories CI runs are: {', '.join(sorted(covered)) or 'none'}"
    )


def _workflows_with_an_include() -> list[Path]:
    """Every workflow whose matrix has an ``include`` block, which is what can collapse."""
    return [f for f in CI_FILES if _include_keys(f.read_text())]


@pytest.mark.parametrize("workflow", _workflows_with_an_include(), ids=lambda p: p.name)
def test_a_matrix_include_expands_to_one_job_per_entry(workflow: Path) -> None:
    """
    Check every matrix creates a job per include entry rather than merging them into one.

    GitHub adds an ``include`` object to every existing combination when none of its keys is a
    dimension of the matrix, so two such objects overwrite each other and only the last survives.
    Listing the values as a real dimension is what makes each include a filter instead.

    Parametrised over the workflows that actually have an include, rather than naming one file:
    unittest.yaml had this matrix and no longer does, and a test pinned to a file that stopped
    having the construct is a test that passes without checking anything.
    """
    text = workflow.read_text()
    include_keys = _include_keys(text)
    dimensions = _matrix_dimensions(text)
    assert include_keys & dimensions, (
        f"no key of the matrix include entries in {workflow.name} is a matrix dimension, so GitHub "
        f"merges them into a single job instead of one per entry. include keys: "
        f"{sorted(include_keys)}, matrix dimensions: {sorted(dimensions)}"
    )


_IGNORE_PATH = re.compile(r"--ignore=(\S+)")


def test_every_ignored_path_exists() -> None:
    """
    An ``--ignore`` of a path that is gone is silently a no-op.

    The linux job ignores test/shell/test_brun_bash32.py, whose tests need /bin/bash to be 3.2.
    Rename or move that file and the ignore stops matching, the file is collected again, and the
    skips it was added to remove come back with nothing to say so.
    """
    for ci_file in CI_FILES:
        for path in _IGNORE_PATH.findall(ci_file.read_text()):
            assert (FASTSURFER_HOME / path).exists(), (
                f"{ci_file.name} ignores {path}, which does not exist, so the ignore does nothing "
                "and whatever it was hiding is collected again"
            )


def test_some_workflow_still_has_a_matrix_include() -> None:
    """The parametrised check above silently covers nothing if no workflow has one left."""
    assert _workflows_with_an_include(), (
        "no workflow has a matrix include, so the collapse check covers nothing. Either that is "
        "correct and both checks can go, or an include was lost."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
