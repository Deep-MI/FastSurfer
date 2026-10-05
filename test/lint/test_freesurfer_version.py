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
Guard the copies of the supported FreeSurfer version.

``tool.freesurfer.version`` in pyproject.toml is the version FastSurfer ships and supports.
tools/Docker/build.py, the build_freesurfer stage of tools/Docker/Dockerfile and
tools/macos_build/build_release_package.sh read it directly, but four places hold copies, because
they run where pyproject.toml cannot be read or before python is available:

* ``FS_VERSION_SUPPORT`` in recon_surf/recon-surf.sh and recon_surf/recon-surfreg.sh, the version
  check against ``$FREESURFER_HOME/build-stamp.txt``,
* the fallback of ``get_supported_freesurfer_version`` in recon_surf/long_compat_segmentHA.py,
* the ``ARG FREESURFER_VERSION`` default in tools/Docker/Dockerfile, which only feeds the image
  labels; the build_freesurfer stage fails when it differs from the version it installs.

A copy left behind on a version bump makes the surface pipeline refuse the FreeSurfer it ships
with, or accept the wrong one, and a direct ``docker build`` of the FreeSurfer image fail.
"""

import re
import tomllib
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent
PYPROJECT = FASTSURFER_HOME / "pyproject.toml"


def _pyproject_freesurfer() -> dict:
    """
    Read the [tool.freesurfer] table of pyproject.toml.

    Returns
    -------
    dict
        A mapping with the "version" key and the "urls" mapping.
    """
    with open(PYPROJECT, "rb") as fp:
        return tomllib.load(fp)["tool"]["freesurfer"]


# each copy: the file and a pattern whose first group is the version it holds
COPIES = {
    "recon-surf.sh": ("recon_surf/recon-surf.sh", r"^FS_VERSION_SUPPORT=\"([^\"]+)\""),
    "recon-surfreg.sh": ("recon_surf/recon-surfreg.sh", r"^FS_VERSION_SUPPORT=\"([^\"]+)\""),
    "long_compat_segmentHA.py": (
        "recon_surf/long_compat_segmentHA.py",
        r"^def get_supported_freesurfer_version\(.*?^    return \"([^\"]+)\"",
    ),
    "Dockerfile": ("tools/Docker/Dockerfile", r"^ARG FREESURFER_VERSION=\"([^\"]+)\""),
}


@pytest.fixture(scope="module")
def freesurfer() -> dict:
    """
    Provide the [tool.freesurfer] table of pyproject.toml.

    Returns
    -------
    dict
        A mapping with the "version" key and the "urls" mapping.
    """
    return _pyproject_freesurfer()


@pytest.mark.parametrize("name", list(COPIES))
def test_copy_matches_pyproject(freesurfer: dict, name: str) -> None:
    """Check a hardcoded copy of the FreeSurfer version agrees with tool.freesurfer.version."""
    relative_path, pattern = COPIES[name]
    path = FASTSURFER_HOME / relative_path
    match = re.search(pattern, path.read_text(), re.M | re.S)
    assert match is not None, f"no FreeSurfer version found in {relative_path} (pattern {pattern!r})"
    assert match.group(1) == freesurfer["version"], (
        f"{relative_path} holds FreeSurfer {match.group(1)}, but tool.freesurfer.version in "
        f"pyproject.toml is {freesurfer['version']}; update the copy along with the key"
    )


def test_urls_follow_the_version(freesurfer: dict) -> None:
    """Check the download urls take the version from the key, so a bump cannot leave one behind."""
    for platform, url in freesurfer["urls"].items():
        assert "{version}" in url and freesurfer["version"] not in url, (
            f"tool.freesurfer.urls.{platform} is {url!r}; it has to use the {{version}} "
            f"placeholder rather than a literal version, or a bump of tool.freesurfer.version "
            f"downloads the old FreeSurfer"
        )
