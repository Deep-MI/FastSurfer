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
Two structural properties of the Dockerfile that fail silently when they are broken.

Neither is about what the file says, both are about the shape of the instruction graph, so they are
computed from a parse rather than matched against fixed text. Both failure modes are invisible: one
costs build time, the other produces an image whose build argument was quietly empty.

The Dockerfile is parsed as text, so this runs under ``uv run --no-project --with pytest`` like the
rest of test/lint.
"""

import re
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent
DOCKERFILE = FASTSURFER_HOME / "tools" / "Docker" / "Dockerfile"

STAGE = re.compile(r"^FROM\s+\S+(?:\s+AS\s+(\S+))?", re.I)
DECLARES = re.compile(r"^(ARG|ENV)\s+([A-Za-z_][A-Za-z0-9_]*)")
BUILDS = re.compile(r"^(RUN|COPY|ADD)\b", re.I)
HEREDOC = re.compile(r"^RUN\b.*<<-?\s*\"?EOF\"?", re.I)
# $NAME and ${NAME...}, the latter possibly with a substitution such as ${DEVICE/cu11/11.}
REFERENCE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")


@pytest.fixture(scope="module")
def lines() -> list[str]:
    return DOCKERFILE.read_text().splitlines()


@pytest.fixture(scope="module")
def global_args(lines: list[str]) -> set[str]:
    """Args declared before the first FROM, which every stage has to re-declare to use."""
    first_from = next(i for i, line in enumerate(lines) if line.startswith("FROM"))
    return {m.group(2) for line in lines[:first_from] if (m := DECLARES.match(line))}


def stages(lines: list[str]):
    """Yield (name, first_line, instructions) per build stage, instructions as (lineno, text)."""
    name, start, body = None, 0, []
    for number, line in enumerate(lines, 1):
        if match := STAGE.match(line):
            if name:
                yield name, start, body
            name, start, body = match.group(1) or f"<anonymous at {number}>", number, []
        elif name:
            body.append((number, line))
    if name:
        yield name, start, body


def test_no_build_step_follows_a_label(lines: list[str]):
    """
    A label is metadata, but every later step in the stage keys on it.

    The runtime labels carry the version and the git hash, so they change with every commit. With a
    build step below them, each commit rebuilds the stage from the label down, which once meant
    re-downloading the network weights on every build. Nothing fails when this regresses, the build
    just quietly stops reusing its cache.
    """
    offenders = {}
    for name, _, body in stages(lines):
        label = next((number for number, line in body if line.startswith("LABEL")), None)
        if label is None:
            continue
        after = [number for number, line in body if BUILDS.match(line) and number > label]
        if after:
            offenders[name] = (label, after)
    assert offenders == {}, "\n".join(
        f"stage '{name}': LABEL on line {label} has {len(after)} RUN/COPY step(s) below it, "
        f"first on line {after[0]}. Move the label and the args it reads to the end of the stage."
        for name, (label, after) in offenders.items()
    )


def test_every_referenced_global_arg_is_declared_in_its_stage(lines: list[str], global_args: set[str]):
    """
    An arg declared before the first FROM is not in scope inside a stage until re-declared.

    It expands to the empty string instead, which no builder warns about. That silently disabled
    the `--insecure` escape hatch for the FreeSurfer download, and left the CUDA image with
    ``NVIDIA_REQUIRE_CUDA="cuda>="``, both for as long as nobody read the built image.
    """
    problems = []
    for name, _, body in stages(lines):
        declared, index = set(), 0
        while index < len(body):
            number, line = body[index]
            if match := DECLARES.match(line):
                declared.add(match.group(2))
            # a RUN heredoc reaches until its terminator, and the whole body can reference args
            chunk = [line]
            if HEREDOC.match(line):
                while index + 1 < len(body) and body[index + 1][1].strip() != "EOF":
                    index += 1
                    chunk.append(body[index][1])
            for text in chunk:
                # comments name args to explain them, in the Dockerfile and inside the heredocs
                if text.lstrip().startswith("#"):
                    continue
                for referenced in REFERENCE.findall(text):
                    if referenced in global_args and referenced not in declared:
                        problems.append(f"stage '{name}' line {number} uses ${referenced}")
            index += 1
    # one report per stage and arg, a loop body repeated per line is the same defect
    unique = sorted(set(problems))
    assert unique == [], "these build args expand to nothing, because the stage never re-declares them:\n" + "\n".join(
        f"  {problem}" for problem in unique
    )
