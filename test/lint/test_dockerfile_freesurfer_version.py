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
Guard the copy of the FreeSurfer version in tools/Docker/Dockerfile.

``tool.freesurfer.version`` in pyproject.toml decides which FreeSurfer the build_freesurfer stage
installs, the stage reads it from the build context. The labels of that stage cannot, so they read
``ARG FREESURFER_VERSION``, which tools/Docker/build.py passes and whose default applies to a direct
``docker build``. The stage fails once the two disagree, so a stale default surfaces only as a
failed build of the FreeSurfer image, long after the commit that bumped the key.
"""

import re
import sys
from pathlib import Path

FASTSURFER_HOME = Path(__file__).parent.parent.parent
PYPROJECT = FASTSURFER_HOME / "pyproject.toml"
DOCKERFILE = FASTSURFER_HOME / "tools" / "Docker" / "Dockerfile"


def _freesurfer_version() -> str:
    """
    Read tool.freesurfer.version from pyproject.toml, tolerating the absence of a toml parser.

    The code-style job runs test/lint on python 3.10 without dependencies, so neither tomllib nor
    tomli is guaranteed; fall back to a regex scoped to the [tool.freesurfer] section, like
    test_python_version.py does for [tool.python].
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
            return tomllib.load(fp)["tool"]["freesurfer"]["version"]

    text = PYPROJECT.read_text()
    # scope to the [tool.freesurfer] section so an unrelated `version =` cannot match
    section = re.search(r"^\[tool\.freesurfer]$(.*?)^\[", text, re.M | re.S)
    assert section is not None, f"no [tool.freesurfer] section in {PYPROJECT}"
    version = re.search(r"^version\s*=\s*[\"']([^\"']+)[\"']", section.group(1), re.M)
    assert version is not None, f"no version key in the [tool.freesurfer] section of {PYPROJECT}"
    return version.group(1)


def test_dockerfile_arg_default_matches() -> None:
    """Check the default of ARG FREESURFER_VERSION is the version the image installs."""
    match = re.search(r"^ARG FREESURFER_VERSION=\"([^\"]+)\"", DOCKERFILE.read_text(), re.M)
    assert match is not None, f"no 'ARG FREESURFER_VERSION=\"...\"' found in {DOCKERFILE}"
    version = _freesurfer_version()
    assert match.group(1) == version, (
        f"{DOCKERFILE.name} defaults FREESURFER_VERSION to {match.group(1)}, but pyproject.toml "
        f"installs FreeSurfer {version} (tool.freesurfer.version); a direct `docker build` fails "
        f"in the build_freesurfer stage until the ARG default is updated alongside the key"
    )
