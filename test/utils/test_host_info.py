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
Guard the host description written into every run log.

The functions here read files and syscalls that differ per platform and are absent on the
development machines, so a wrong branch fails silently: the log keeps a plausible-looking
line that describes the wrong machine. These tests pin each branch against a fake.
"""

import subprocess
import sys

import pytest

from FastSurferCNN import host_info


class TestCpuModel:
    def test_reads_proc_cpuinfo(self, tmp_path, monkeypatch):
        """On linux the model comes from the first 'model name' line."""
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text(
            "processor\t: 0\n"
            "vendor_id\t: GenuineIntel\n"
            "model name\t: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz\n"
            "model name\t: a second core, which must not win\n"
        )
        real_open = open
        monkeypatch.setattr(
            "builtins.open",
            lambda file, *a, **kw: real_open(cpuinfo if file == "/proc/cpuinfo" else file, *a, **kw),
        )
        assert host_info.cpu_model() == "Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz"

    def test_falls_back_when_there_is_no_model_name(self, tmp_path, monkeypatch):
        """ARM linux has no 'model name' field, so the platform fallback has to carry it."""
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text("processor\t: 0\nCPU implementer\t: 0x41\n")
        real_open = open
        monkeypatch.setattr(
            "builtins.open",
            lambda file, *a, **kw: real_open(cpuinfo if file == "/proc/cpuinfo" else file, *a, **kw),
        )
        monkeypatch.setattr(host_info.platform, "system", lambda: "Linux")
        monkeypatch.setattr(host_info.platform, "processor", lambda: "")
        assert host_info.cpu_model() == "unknown"

    def test_uses_sysctl_on_darwin(self, monkeypatch):
        monkeypatch.setattr("builtins.open", _raise_oserror)
        monkeypatch.setattr(host_info.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(host_info.subprocess, "check_output", lambda *a, **kw: "Apple M2\n")
        assert host_info.cpu_model() == "Apple M2"

    def test_survives_a_missing_sysctl(self, monkeypatch):
        """A missing binary must not take the run down over a log line."""
        monkeypatch.setattr("builtins.open", _raise_oserror)
        monkeypatch.setattr(host_info.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(host_info.subprocess, "check_output", _raise_oserror)
        monkeypatch.setattr(host_info.platform, "processor", lambda: "")
        assert host_info.cpu_model() == "unknown"


class TestCpuQuota:
    def test_cgroup_v2_quota(self, tmp_path, monkeypatch):
        cpu_max = tmp_path / "cpu.max"
        cpu_max.write_text("200000 100000\n")
        monkeypatch.setattr(host_info, "CGROUP_V2_CPU_MAX", cpu_max)
        assert host_info.cpu_quota() == 2.0

    def test_cgroup_v2_unlimited(self, tmp_path, monkeypatch):
        cpu_max = tmp_path / "cpu.max"
        cpu_max.write_text("max 100000\n")
        monkeypatch.setattr(host_info, "CGROUP_V2_CPU_MAX", cpu_max)
        monkeypatch.setattr(host_info, "CGROUP_V1_CPU_QUOTA", tmp_path / "absent")
        assert host_info.cpu_quota() is None

    def test_cgroup_v1_quota(self, tmp_path, monkeypatch):
        quota, period = tmp_path / "quota", tmp_path / "period"
        quota.write_text("150000\n")
        period.write_text("100000\n")
        monkeypatch.setattr(host_info, "CGROUP_V2_CPU_MAX", tmp_path / "absent")
        monkeypatch.setattr(host_info, "CGROUP_V1_CPU_QUOTA", quota)
        monkeypatch.setattr(host_info, "CGROUP_V1_CPU_PERIOD", period)
        assert host_info.cpu_quota() == 1.5

    def test_cgroup_v1_uncapped_is_negative(self, tmp_path, monkeypatch):
        """cgroup v1 writes -1 rather than removing the file when there is no cap."""
        quota, period = tmp_path / "quota", tmp_path / "period"
        quota.write_text("-1\n")
        period.write_text("100000\n")
        monkeypatch.setattr(host_info, "CGROUP_V2_CPU_MAX", tmp_path / "absent")
        monkeypatch.setattr(host_info, "CGROUP_V1_CPU_QUOTA", quota)
        monkeypatch.setattr(host_info, "CGROUP_V1_CPU_PERIOD", period)
        assert host_info.cpu_quota() is None

    def test_no_cgroup_at_all(self, tmp_path, monkeypatch):
        monkeypatch.setattr(host_info, "CGROUP_V2_CPU_MAX", tmp_path / "absent")
        monkeypatch.setattr(host_info, "CGROUP_V1_CPU_QUOTA", tmp_path / "absent")
        assert host_info.cpu_quota() is None


class TestCpuCount:
    def test_plain_host(self, monkeypatch):
        monkeypatch.setattr(host_info.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(host_info.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
        monkeypatch.setattr(host_info, "cpu_quota", lambda: None)
        assert host_info.cpu_count() == "8"

    def test_affinity_narrower_than_the_host(self, monkeypatch):
        monkeypatch.setattr(host_info.os, "cpu_count", lambda: 16)
        monkeypatch.setattr(host_info.os, "sched_getaffinity", lambda _: set(range(4)), raising=False)
        monkeypatch.setattr(host_info, "cpu_quota", lambda: None)
        assert host_info.cpu_count() == "4 of 16"

    def test_cgroup_cap_is_reported(self, monkeypatch):
        """`docker run --cpus=2` leaves affinity at every core, so the cap has to be named."""
        monkeypatch.setattr(host_info.os, "cpu_count", lambda: 16)
        monkeypatch.setattr(host_info.os, "sched_getaffinity", lambda _: set(range(16)), raising=False)
        monkeypatch.setattr(host_info, "cpu_quota", lambda: 2.0)
        assert host_info.cpu_count() == "16 capped at 2 by the cgroup"

    def test_without_sched_getaffinity(self, monkeypatch):
        """macOS has no sched_getaffinity, so the count falls back to the host total."""
        monkeypatch.setattr(host_info.os, "cpu_count", lambda: 8)
        monkeypatch.delattr(host_info.os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(host_info, "cpu_quota", lambda: None)
        assert host_info.cpu_count() == "8"

    def test_unknown_core_count(self, monkeypatch):
        """os.cpu_count may return None, which must not print as the string 'None'."""
        monkeypatch.setattr(host_info.os, "cpu_count", lambda: None)
        monkeypatch.delattr(host_info.os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(host_info, "cpu_quota", lambda: None)
        assert host_info.cpu_count() == "unknown"


class TestHostInfo:
    def test_reports_the_thread_limits_that_are_set(self, monkeypatch):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.setenv("MKL_NUM_THREADS", "3")
        limits = [line for line in host_info.host_info() if line.startswith("Thread limits:")]
        assert limits == ["Thread limits: MKL_NUM_THREADS=3"]

    def test_says_so_when_no_limit_is_set(self, monkeypatch):
        for var in host_info.THREAD_VARS:
            monkeypatch.delenv(var, raising=False)
        limits = [line for line in host_info.host_info() if line.startswith("Thread limits:")]
        assert limits == ["Thread limits: none set"]

    def test_every_fact_is_a_single_line(self):
        """The shell caller joins these with newlines, so an embedded newline breaks the block."""
        assert all("\n" not in line for line in host_info.host_info(with_torch=True))

    def test_torch_is_off_by_default(self):
        assert not any(line.startswith("Torch") for line in host_info.host_info())


class TestTorchInfo:
    def test_reports_the_cpu_capability(self):
        line = host_info.torch_info()[0]
        assert line.startswith("Torch ") and "CPU capability" in line

    def test_threads_can_be_left_out(self):
        assert "intra-op" not in host_info.torch_info(with_threads=False)[0]

    def test_the_hostname_is_not_reported(self):
        """It says nothing in a container, and four other files already carry it."""
        assert not any(host_info.platform.node() in line for line in host_info.host_info())


def test_runs_as_a_script_from_an_unrelated_directory(tmp_path):
    """
    The shell callers invoke this by file path, like FastSurferCNN/version.py.

    That only works while the module sits outside FastSurferCNN/utils, whose logging.py
    shadows the standard library module torch needs. Running it from a foreign directory
    with no PYTHONPATH is what keeps that property honest.
    """
    result = subprocess.run(
        [sys.executable, "-s", str(host_info.Path(host_info.__file__)), "--torch"],
        capture_output=True, text=True, cwd=tmp_path, check=True,
    )
    fields = [line.split(" ")[0].rstrip(":") for line in result.stdout.splitlines()]
    assert fields == ["Platform", "CPU", "CPU", "Torch", "Thread"]
    assert "not importable" not in result.stdout


def _raise_oserror(*args, **kwargs):
    raise OSError("absent")


if __name__ == "__main__":
    pytest.main([__file__])
