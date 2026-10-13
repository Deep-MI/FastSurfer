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
Guard the copy of the CUDA images in tools/fastsurfer_container.py.

The launcher runs with any system Python and without FastSurfer installed, so it cannot import
FastSurferCNN/gpu_support.py and keeps its own copy of the CUDA builds and their minimum drivers.
A copy left behind when the CUDA versions change makes it pick an image that is not released.
"""

import importlib.util
import sys
import tomllib
from pathlib import Path

FASTSURFER_HOME = Path(__file__).parent.parent.parent


def _load(name: str, path: Path):
    # from the path, because the lint job installs nothing but pytest; registered in sys.modules,
    # which dataclasses need
    spec = importlib.util.spec_from_file_location(name, path)
    module = sys.modules.setdefault(name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


launcher = _load("fastsurfer_container", FASTSURFER_HOME / "tools" / "fastsurfer_container.py")
gpu_support = _load("gpu_support_for_lint", FASTSURFER_HOME / "FastSurferCNN" / "gpu_support.py")


def test_launcher_builds_match_gpu_support():
    assert launcher.NEWEST_BUILD == gpu_support.NEWEST_BUILD
    assert launcher.LEGACY_BUILD == gpu_support.LEGACY_BUILD
    for (major, _), _ in (launcher.NEWEST_BUILD, launcher.LEGACY_BUILD):
        assert launcher.MIN_DRIVER[major] == gpu_support.MIN_DRIVER[major]


def test_launcher_capabilities_cover_builds():
    # the GPU range of each build comes from its PyTorch, so a new build needs its entry checked by hand
    assert set(launcher.MIN_CAPABILITY) == {launcher.NEWEST_BUILD[1], launcher.LEGACY_BUILD[1]}


def test_launcher_default_build_matches_pyproject():
    with open(FASTSURFER_HOME / "pyproject.toml", "rb") as fp:
        cuda = tomllib.load(fp)["tool"]["cuda"]["version"]
    assert launcher.NEWEST_BUILD[0] == tuple(int(v) for v in cuda.split("."))
