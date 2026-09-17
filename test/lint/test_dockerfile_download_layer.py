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
Keep the Dockerfile's early COPY covering everything the checkpoint download imports.

The weights layer sits above `COPY . /fastsurfer/` so that a source change does not re-fetch
about 700 MB. That only works while the narrow COPY above it carries the whole import closure of
download_checkpoints.py. Miss a file and the build fails at that layer, which is loud but wastes a
build; copy a whole package instead and every edit inside it re-invalidates the download, which is
quiet and undoes the point of the split.

Imports are read with ast rather than by importing anything, and the Dockerfile is parsed as text,
so this runs under ``uv run --no-project --with pytest`` like the rest of test/lint.
"""

import ast
import re
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent
DOCKERFILE = FASTSURFER_HOME / "tools" / "Docker" / "Dockerfile"
ENTRY_POINT = "FastSurferCNN/download_checkpoints.py"
# the packages whose files are part of this repository rather than the venv
REPO_PACKAGES = ("FastSurferCNN", "CerebNet", "CorpusCallosum", "HypVINN", "recon_surf")


def import_closure(entry: str) -> set[str]:
    """Every file in this repository that `entry` reaches through imports, including itself."""
    seen: set[str] = set()
    todo = [entry]
    while todo:
        relative = todo.pop()
        if relative in seen:
            continue
        seen.add(relative)
        source = FASTSURFER_HOME / relative
        if not source.exists():
            continue
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if module.split(".")[0] not in REPO_PACKAGES:
                    continue
                as_module = Path(module.replace(".", "/") + ".py")
                as_package = Path(module.replace(".", "/")) / "__init__.py"
                for candidate in (as_module, as_package):
                    if (FASTSURFER_HOME / candidate).exists():
                        todo.append(str(candidate))
                # a package's __init__ runs on import of anything inside it
                parent_init = Path(module.split(".")[0]) / "__init__.py"
                if (FASTSURFER_HOME / parent_init).exists():
                    todo.append(str(parent_init))
    return seen


def early_copy_sources() -> list[str]:
    """
    The build-context paths copied in the runtime stage before `COPY . /fastsurfer/`.

    Scoped to that stage: an earlier stage's COPY lands in a different image and would otherwise
    read as covering a file that the download layer cannot see.
    """
    lines = DOCKERFILE.read_text().splitlines()
    runtime = next(i for i, line in enumerate(lines) if re.match(r"^FROM\s+\S+\s+AS\s+runtime\s*$", line))
    full_copy = next(
        i for i, line in enumerate(lines) if re.match(r"^COPY\s+\.\s+/fastsurfer/?\s*$", line)
    )
    sources: list[str] = []
    for line in lines[runtime:full_copy]:
        match = re.match(r"^COPY\s+(?!--from)(.+)$", line)
        if match:
            # the last word is the destination, the rest are sources
            sources.extend(match.group(1).split()[:-1])
    return sources


def is_covered(relative: str, sources: list[str]) -> bool:
    """Whether an early COPY brings this file in, directly or as part of a copied directory."""
    path = Path(relative)
    return any(path == Path(s) or Path(s) in path.parents for s in sources)


@pytest.fixture(scope="module")
def sources() -> list[str]:
    return early_copy_sources()


def test_the_download_layer_precedes_the_source_copy():
    """Without this order the download keys on every file in the repository."""
    text = DOCKERFILE.read_text()
    full_copy = text.index("\nCOPY . /fastsurfer/")
    download = text.index("download_checkpoints.py --all")
    assert download < full_copy, (
        "the checkpoint download has moved below `COPY . /fastsurfer/`, so every source change "
        "now re-fetches the weights"
    )


def test_early_copy_covers_the_download_import_closure(sources: list[str]):
    """Anything reachable from the downloader has to be in the image before it runs."""
    missing = sorted(f for f in import_closure(ENTRY_POINT) if not is_covered(f, sources))
    assert missing == [], (
        f"the checkpoint download imports {len(missing)} file(s) that the early COPY does not "
        f"bring in, so that layer will fail to build: {missing}"
    )


def test_early_copy_stays_narrow(sources: list[str]):
    """
    Copying a whole package would re-invalidate the weights on unrelated edits.

    FastSurferCNN/utils is deliberate: the downloader reaches most of it. A second package
    appearing here is the case worth catching.
    """
    packages = sorted(s for s in sources if s in REPO_PACKAGES)
    assert packages == [], (
        f"the early COPY takes whole package(s) {packages}, so any edit inside them re-downloads "
        f"the weights; copy the files the downloader needs instead"
    )
