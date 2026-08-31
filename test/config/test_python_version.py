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
Guard the single-source-of-truth for the shipped python version.

``tool.python.version`` in pyproject.toml is the one place the interpreter FastSurfer is built
with is declared. Two consumers cannot read it and therefore hold hardcoded copies:

* the ``ARG PYTHON_VERSION`` default in tools/Docker/Dockerfile -- a Dockerfile cannot parse a
  toml file at build time,
* the ``python-version`` in .github/workflows/quicktest.yaml -- that step *provides* the
  interpreter which later reads pyproject.toml.

Without these tests, bumping the key leaves those copies behind silently. The failure mode is a
confusing build rather than a red test, which is exactly what this module prevents.
"""

import re
import sys
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent
PYPROJECT = FASTSURFER_HOME / "pyproject.toml"
DOCKERFILE = FASTSURFER_HOME / "tools" / "Docker" / "Dockerfile"
BUILD_PY = FASTSURFER_HOME / "tools" / "Docker" / "build.py"
QUICKTEST_YAML = FASTSURFER_HOME / ".github" / "workflows" / "quicktest.yaml"
UNITTEST_YAML = FASTSURFER_HOME / ".github" / "workflows" / "unittest.yaml"


def _load_pyproject() -> dict:
    """
    Parse pyproject.toml, tolerating the absence of a toml parser.

    tomllib is stdlib only from python 3.11, and the unittest CI job deliberately runs on the
    oldest supported version, so neither tomllib nor tomli is guaranteed. Fall back to a narrow
    regex over just the two keys this module needs rather than adding a test-only dependency.

    Returns
    -------
    dict
        A mapping with the "python_version" and "requires_python" keys.
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib
        except ImportError:
            tomllib = None

    if tomllib is not None:
        with open(PYPROJECT, "rb") as fp:
            parsed = tomllib.load(fp)
        return {
            "python_version": parsed["tool"]["python"]["version"],
            "requires_python": parsed["project"]["requires-python"],
        }

    text = PYPROJECT.read_text()
    # scope to the [tool.python] section so an unrelated `version =` cannot match
    section = re.search(r"^\[tool\.python]$(.*?)^\[", text, re.M | re.S)
    assert section is not None, f"no [tool.python] section in {PYPROJECT}"
    version = re.search(r"^version\s*=\s*[\"']([^\"']+)[\"']", section.group(1), re.M)
    assert version is not None, f"no version key in the [tool.python] section of {PYPROJECT}"
    requires = re.search(r"^requires-python\s*=\s*[\"']([^\"']+)[\"']", text, re.M)
    assert requires is not None, f"no requires-python in {PYPROJECT}"
    return {"python_version": version.group(1), "requires_python": requires.group(1)}


@pytest.fixture(scope="module")
def config() -> dict:
    """
    Provide the declared python versions.

    Returns
    -------
    dict
        A mapping with the "python_version" and "requires_python" keys.
    """
    return _load_pyproject()


def test_shipped_version_is_a_bare_minor_version(config: dict) -> None:
    """Check the key is an exact "major.minor", since consumers interpolate it verbatim."""
    version = config["python_version"]
    assert re.fullmatch(r"\d+\.\d+", version), (
        f"tool.python.version is {version!r}, but it is interpolated straight into "
        f"'python<version>' and 'brew install python@<version>', so it must be a bare "
        f"major.minor with no specifier or patch level"
    )


def test_shipped_version_satisfies_the_support_floor(config: dict) -> None:
    """Check the shipped version is not older than the declared support floor."""
    floor_spec = config["requires_python"]
    floor = re.sub(r"^[><=~!]+", "", floor_spec.split(",")[0]).strip()
    shipped = tuple(int(part) for part in config["python_version"].split("."))
    assert shipped >= tuple(int(part) for part in floor.split(".")), (
        f"tool.python.version ({config['python_version']}) is older than the "
        f"project.requires-python floor ({floor_spec})"
    )


def test_dockerfile_arg_default_matches(config: dict) -> None:
    """Check the Dockerfile ARG fallback agrees with the key it cannot read."""
    match = re.search(r"^ARG PYTHON_VERSION=\"([^\"]+)\"", DOCKERFILE.read_text(), re.M)
    assert match is not None, f"no 'ARG PYTHON_VERSION=\"...\"' found in {DOCKERFILE}"
    assert match.group(1) == config["python_version"], (
        f"{DOCKERFILE.name} pins python {match.group(1)} but tool.python.version is "
        f"{config['python_version']}; the ARG default is the fallback for a direct `docker build` "
        f"and has to be updated alongside the key"
    )


def test_build_py_forwards_the_key() -> None:
    """Check build.py passes the key through instead of hardcoding a version."""
    text = BUILD_PY.read_text()
    assert "PYTHON_VERSION={pyproject_python['version']}" in text, (
        f"{BUILD_PY.name} no longer forwards PYTHON_VERSION from pyproject.toml; the docker image "
        f"would silently fall back to the Dockerfile ARG default"
    )


def test_quicktest_workflow_matches(config: dict) -> None:
    """Check the quicktest runner interpreter tracks the shipped default."""
    versions = re.findall(r"^\s*python-version:\s*['\"]([^'\"]+)['\"]", QUICKTEST_YAML.read_text(), re.M)
    assert versions, f"no python-version found in {QUICKTEST_YAML}"
    for version in versions:
        assert version == config["python_version"], (
            f"{QUICKTEST_YAML.name} pins python {version} but tool.python.version is "
            f"{config['python_version']}; this value cannot be read from pyproject.toml because "
            f"it provides the interpreter that reads it, so it must be bumped by hand"
        )


def test_unittest_workflow_pins_the_support_floor(config: dict) -> None:
    """Check the unittest job still tests the oldest supported version, not the shipped one."""
    floor_spec = config["requires_python"]
    floor = re.sub(r"^[><=~!]+", "", floor_spec.split(",")[0]).strip()
    versions = re.findall(r"^\s*python-version:\s*['\"]([^'\"]+)['\"]", UNITTEST_YAML.read_text(), re.M)
    assert versions, f"no python-version found in {UNITTEST_YAML}"
    for version in versions:
        assert version == floor, (
            f"{UNITTEST_YAML.name} pins python {version}, but it should pin the "
            f"project.requires-python floor ({floor}). This job is the only thing that tests the "
            f"oldest-supported claim; pointing it at the shipped default would drop that coverage"
        )
