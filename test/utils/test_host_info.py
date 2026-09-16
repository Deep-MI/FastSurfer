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
Guard the host description written into every run log.

The functions here read files and syscalls that differ per platform and are absent on the
development machines, so a wrong branch fails silently: the log keeps a plausible-looking
line that describes the wrong machine. These tests pin each branch against a fake.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from FastSurferCNN import host_info

FASTSURFER_HOME = Path(host_info.__file__).parent.parent

# The CI job for this directory installs pytest and nothing else, because host_info is standard
# library underneath. Its torch branch is therefore unreachable there, and its ImportError branch
# is unreachable anywhere torch is present, so each is skipped where it cannot run.
HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed in this job")


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

    def test_dispatch_overrides_are_reported(self, monkeypatch):
        """
        A kernel override changes what every number below it means, so the header records it.

        Reported even when absent, so "none set" is on the record rather than being the same
        as a line that failed to print.
        """
        monkeypatch.delenv("OPENBLAS_CORETYPE", raising=False)
        monkeypatch.delenv("NPY_DISABLE_CPU_FEATURES", raising=False)
        for var in host_info.DISPATCH_VARS:
            monkeypatch.delenv(var, raising=False)
        assert "Dispatch overrides: none set" in host_info.host_info()

        monkeypatch.setenv("OPENBLAS_CORETYPE", "Nehalem")
        line = next(ln for ln in host_info.host_info() if ln.startswith("Dispatch overrides:"))
        assert "OPENBLAS_CORETYPE=Nehalem" in line

    def test_thread_limits_and_dispatch_are_separate_lines(self):
        """They answer different questions: how many threads, and which kernel."""
        lines = host_info.host_info()
        assert sum(ln.startswith("Thread limits:") for ln in lines) == 1
        assert sum(ln.startswith("Dispatch overrides:") for ln in lines) == 1

    def test_the_hostname_is_not_reported(self):
        """It says nothing in a container, and four other files already carry it."""
        assert not any(host_info.platform.node() in line for line in host_info.host_info())


class TestTorchInfo:
    @needs_torch
    def test_reports_the_cpu_capability(self):
        line = host_info.torch_info()[0]
        assert line.startswith("Torch ") and "CPU capability" in line

    @needs_torch
    def test_threads_can_be_left_out(self):
        assert "intra-op" not in host_info.torch_info(with_threads=False)[0]

    def test_says_so_plainly_when_torch_is_absent(self, monkeypatch):
        """
        The line a user without torch sees, so it is worth asserting wherever this runs.

        `import torch` raises when sys.modules holds None for it, which makes this testable with
        torch installed. It used to be gated on torch being absent, which stopped covering
        anything once every CI job installed the project.

        It has to stay one line and name the reason, because the shell caller joins these with
        newlines and a reader has to be able to tell a missing install from a broken one.
        """
        monkeypatch.setitem(sys.modules, "torch", None)
        line = host_info.torch_info()[0]
        assert line.startswith("Torch: not importable (")
        assert "\n" not in line


class TestNumericalFingerprint:
    @needs_torch
    def test_is_stable_within_a_process(self):
        """
        Two calls must agree, or the fingerprint says nothing about the host.

        It is the whole point that a difference means the hosts differ, so any variation
        from call to call would make every comparison a false positive.
        """
        assert host_info.numerical_fingerprint() == host_info.numerical_fingerprint()

    @needs_torch
    def test_reports_the_two_operations_the_networks_use(self):
        fp = host_info.numerical_fingerprint()
        assert re.fullmatch(r"conv=[0-9a-f]{12} soft=[0-9a-f]{12}", fp), fp

    @needs_torch
    def test_is_a_single_line(self):
        """The shell caller joins these with newlines, so an embedded one breaks the block."""
        assert "\n" not in host_info.numerical_fingerprint()

    @needs_torch
    def test_does_not_depend_on_the_thread_count(self):
        """
        The fingerprint has to describe the host, not the invocation.

        It is emitted both from log headers, where torch still has its default thread count,
        and from the networks, after `--threads` has been applied. Those have to agree, or
        the same machine reports two classes and every comparison is a false positive.
        """
        import torch

        before = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            one = host_info.numerical_fingerprint()
            torch.set_num_threads(4)
            four = host_info.numerical_fingerprint()
        finally:
            torch.set_num_threads(before)
        assert one == four, f"thread count changed the fingerprint: {one} against {four}"

    @needs_torch
    def test_tracks_the_selected_kernels(self):
        """
        Capping the ISA has to change the hash, or it cannot detect the split it exists for.

        Skipped where there are no wider kernels compiled in to cap, notably arm64, since
        there the cap is correctly a no-op and proves nothing either way.
        """
        import torch

        if torch.backends.cpu.get_cpu_capability() == "NO AVX":
            pytest.skip("no wider kernels on this host, so the cap cannot change anything")

        code = "import FastSurferCNN.host_info as h; print(h.numerical_fingerprint())"
        env = {**os.environ, "PYTHONPATH": str(FASTSURFER_HOME)}
        capped = {**env, "ATEN_CPU_CAPABILITY": "default", "ONEDNN_MAX_CPU_ISA": "SSE41"}
        run = [sys.executable, "-c", code]
        native = subprocess.run(run, env=env, capture_output=True, text=True, check=True).stdout
        lowered = subprocess.run(run, env=capped, capture_output=True, text=True, check=True).stdout
        assert native != lowered, f"capping the ISA did not change the fingerprint: {native!r}"

    def test_says_so_plainly_when_torch_is_absent(self, monkeypatch):
        """Testable with torch installed, so it covers the no-torch line wherever this runs."""
        monkeypatch.setitem(sys.modules, "torch", None)
        assert host_info.numerical_fingerprint().startswith("not available (")


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
    assert fields == ["Platform", "CPU", "CPU", "Torch", "Thread", "Dispatch"]
    if HAS_TORCH:
        # the point of running it from elsewhere: as a file rather than -m it used to import the
        # FastSurferCNN/utils/logging.py next to it and report torch as missing
        assert "not importable" not in result.stdout


def _raise_oserror(*args, **kwargs):
    raise OSError("absent")


if __name__ == "__main__":
    pytest.main([__file__])
