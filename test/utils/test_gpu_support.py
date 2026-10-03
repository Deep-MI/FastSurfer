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
Guard the explanation given when a GPU cannot be used.

None of the development machines or CI runners has a GPU the images do not support, so these
branches are never reached by a real run. The architecture lists are the ones the PyTorch
wheels report.
"""

import pytest

from FastSurferCNN import gpu_support

# torch.cuda.get_arch_list() of the wheels
CU118_27 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_37", "sm_90"]
CU126_214 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
CU128_27 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]
CU132_214 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


class TestSupportsCapability:
    @pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1)])
    def test_cu132_runs_turing_to_blackwell(self, capability):
        """A cubin covers the later minor versions of its major, so sm_86 runs Ada and sm_120 runs GB10."""
        assert gpu_support.supports_capability(capability, CU132_214)

    @pytest.mark.parametrize("capability", [(5, 2), (6, 1), (7, 0)])
    def test_cu132_rejects_maxwell_to_volta(self, capability):
        assert not gpu_support.supports_capability(capability, CU132_214)

    @pytest.mark.parametrize("capability", [(10, 0), (12, 0)])
    def test_cu126_rejects_blackwell(self, capability):
        """There is no compute_ entry, so nothing newer than Hopper can be compiled by the driver."""
        assert not gpu_support.supports_capability(capability, CU126_214)

    def test_ptx_covers_newer_majors(self):
        assert gpu_support.supports_capability((13, 0), CU128_27)
        assert not gpu_support.supports_capability((13, 0), CU132_214)

    def test_cubin_does_not_cover_a_newer_major(self):
        assert not gpu_support.supports_capability((10, 0), ["sm_90"])

    def test_architecture_specific_variant_runs_only_on_its_own(self):
        assert gpu_support.supports_capability((9, 0), ["sm_90a"])
        assert not gpu_support.supports_capability((9, 1), ["sm_90a"])

    def test_unknown_entries_are_ignored(self):
        assert not gpu_support.supports_capability((8, 0), ["gfx90a", ""])


class TestSuggestion:
    """Against the builds FastSurfer points to, whichever these are; only the oldest arch decides old or new."""

    ARCHS = ["sm_75", "sm_120"]
    OLD, NEW = (6, 1), (99, 0)
    LEGACY, NEWEST = gpu_support.LEGACY_BUILD, gpu_support.NEWEST_BUILD

    @pytest.fixture(autouse=True)
    def in_container(self, monkeypatch):
        monkeypatch.setattr(gpu_support, "in_container", lambda: True)

    def test_old_gpu_on_the_newest_build_points_to_the_legacy_image(self):
        lines = gpu_support._arch_problem("old", self.OLD, self.ARCHS, self.NEWEST[0])
        assert "compute capability 6.1" in lines[0]
        assert f"deepmi/fastsurfer:{self.LEGACY[1]}-v" in lines[-1]

    def test_new_gpu_on_the_legacy_build_points_to_the_newest_image(self):
        lines = gpu_support._arch_problem("new", self.NEW, self.ARCHS, self.LEGACY[0])
        assert f"deepmi/fastsurfer:{self.NEWEST[1]}-v" in lines[-1]

    def test_gpu_newer_than_any_build(self):
        lines = gpu_support._arch_problem("new", self.NEW, self.ARCHS, self.NEWEST[0])
        assert "No FastSurfer build supports this GPU yet" in lines[-1]

    def test_gpu_older_than_any_build(self):
        lines = gpu_support._arch_problem("old", self.OLD, self.ARCHS, self.LEGACY[0])
        assert "No FastSurfer build supports this GPU any more" in lines[-1]

    def test_native_install_points_to_the_wheels(self, monkeypatch):
        monkeypatch.setattr(gpu_support, "in_container", lambda: False)
        lines = gpu_support._arch_problem("old", self.OLD, self.ARCHS, self.NEWEST[0])
        assert lines[-1].endswith(f"https://download.pytorch.org/whl/{self.LEGACY[1]}.")


class TestDriverMajor:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.65.06  Release Build  (dvs-builder)\n", 580),
            ("NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.54.03  Tue Jun  6 22:20:39 UTC 2023\n", 535),
            ("unexpected\n", None),
        ],
    )
    def test_parses_the_proc_file(self, tmp_path, monkeypatch, text, expected):
        version = tmp_path / "version"
        version.write_text(text)
        monkeypatch.setattr(gpu_support, "NVIDIA_VERSION_FILE", version)
        assert gpu_support.driver_major() == expected

    def test_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gpu_support, "NVIDIA_VERSION_FILE", tmp_path / "absent")
        assert gpu_support.driver_major() is None
