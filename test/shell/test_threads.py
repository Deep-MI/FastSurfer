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
Check how recon_surf/threads.sh turns flags and environment into a thread budget, and how
brun_fastsurfer.sh shares the machine between cases that run at the same time.

The order is: --threads, else OMP_NUM_THREADS, else the default, with OMP_THREAD_LIMIT as a cap.
Every library variable is then exported from that budget, except that one the user set lower
stays a ceiling.

The machine size for brun comes from nproc, getconf and lscpu on PATH, so the share is large
enough to show the caps on a small CI runner. The cgroup files are temporary ones, so the cgroup of
a CI container cannot decide the outcome.
"""

import os
import subprocess
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent
THREADS = FASTSURFER_HOME / "recon_surf" / "threads.sh"

THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OMP_THREAD_LIMIT",
)
SCHEDULER_VARS = ("SLURM_CPUS_PER_TASK", "NSLOTS", "NCPUS", "LSB_DJOB_NUMPROC")


@pytest.fixture
def machine(tmp_path):
    """A machine with 16 CPUs and no cgroup quota, to be adjusted per test."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    class Machine:
        cgroup_self = tmp_path / "proc_self_cgroup"
        cgroup_v2_root = tmp_path / "cgroup2"
        cgroup_v1_root = tmp_path / "cgroup1_cpu"

        def cgroup_v2(self, cpu_max: str, at: str = "/", own: str = "/"):
            """A cgroup v2 tree with this process in <own>, and <cpu_max> on the cgroup <at>."""
            directory = self.cgroup_v2_root / at.strip("/")
            directory.mkdir(parents=True, exist_ok=True)
            (self.cgroup_v2_root / "cgroup.controllers").write_text("cpu memory\n")
            (directory / "cpu.max").write_text(f"{cpu_max}\n")
            self.cgroup_self.write_text(f"0::{own}\n")

        def cpus(self, total: int, physical: int):
            cores = "\n".join(f"{i},0" for i in range(physical))
            for name, body in (
                ("getconf", f"echo {total}"),
                ("nproc", f"echo {total}"),
                # lscpu -p=Core,Socket: one line per logical CPU, so hyperthreads repeat a core
                ("lscpu", f'echo "# Core,Socket"\nfor i in 1 2 ; do cat <<END\n{cores}\nEND\ndone'),
            ):
                fake = bin_dir / name
                fake.write_text(f"#!/bin/bash\n{body}\n")
                fake.chmod(0o755)

        def environment(self, **env: str) -> dict[str, str]:
            clean = {k: v for k, v in os.environ.items() if k not in THREAD_VARS + SCHEDULER_VARS}
            clean["PATH"] = f"{bin_dir}{os.pathsep}{clean['PATH']}"
            return clean | env

        def run(self, body: str, real_cgroup: bool = False, **env: str) -> str:
            cgroup = "" if real_cgroup else (
                f'cgroup_self="{self.cgroup_self}"\n'
                f'cgroup_v2_root="{self.cgroup_v2_root}"\n'
                f'cgroup_v1_cpu_root="{self.cgroup_v1_root}"\n'
            )
            script = f'source "{THREADS}"\n{cgroup}{body}\n'
            result = subprocess.run(
                ["bash", "-c", script], capture_output=True, text=True, env=self.environment(**env), check=True,
            )
            return result.stdout

    m = Machine()
    m.cpus(total=16, physical=16)
    return m


def budget(machine, flag: str = "", **env: str) -> str:
    """The budget and its source, as resolve_threads leaves them, with a default of 2."""
    return machine.run(f'resolve_threads "{flag}" 2 ; echo "$threads_budget|$threads_source"', **env).strip()


def exported(machine, flag: str = "", **env: str) -> dict[str, str]:
    """What set_thread_env exports for the resolved budget."""
    body = f'resolve_threads "{flag}" 2 ; set_thread_env "$threads_budget" ; env'
    lines = machine.run(body, **env).splitlines()
    return {k: v for k, _, v in (line.partition("=") for line in lines) if k in THREAD_VARS}


class TestBudgetOrder:
    def test_the_flag_wins_over_omp_num_threads(self, machine):
        assert budget(machine, flag="8", OMP_NUM_THREADS="4") == "8|--threads"

    def test_omp_num_threads_replaces_the_default(self, machine):
        assert budget(machine, OMP_NUM_THREADS="4") == "4|OMP_NUM_THREADS"

    def test_the_default_applies_without_either(self, machine):
        assert budget(machine) == "2|default"

    def test_an_openmp_list_uses_its_first_level(self, machine):
        assert budget(machine, OMP_NUM_THREADS="4,2") == "4|OMP_NUM_THREADS"

    def test_an_invalid_omp_num_threads_falls_back_with_a_warning(self, machine):
        out = machine.run('resolve_threads "" 2 ; describe_threads Test', OMP_NUM_THREADS="abc")
        assert "uses 2 threads (from default)" in out
        assert "Ignoring OMP_NUM_THREADS='abc'" in out

    def test_omp_thread_limit_caps_even_the_flag(self, machine):
        assert budget(machine, flag="8", OMP_THREAD_LIMIT="3") == "3|--threads, capped by OMP_THREAD_LIMIT"

    def test_an_invalid_flag_fails(self, machine):
        out = machine.run('resolve_threads "x7" 2 ; echo "status=$? $threads_note"')
        assert "status=1" in out and "Invalid number of threads 'x7'" in out


class TestExportedVariables:
    def test_omp_num_threads_alone_reaches_every_library(self, machine):
        assert set(exported(machine, OMP_NUM_THREADS="4").values()) == {"4"}

    def test_a_lower_library_value_stays_a_ceiling(self, machine):
        values = exported(machine, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
        assert values["OPENBLAS_NUM_THREADS"] == "1"
        assert values["MKL_NUM_THREADS"] == "4"

    def test_a_higher_library_value_follows_the_budget(self, machine):
        values = exported(machine, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="16")
        assert values["OPENBLAS_NUM_THREADS"] == "4"

    def test_the_flag_replaces_omp_num_threads_itself(self, machine):
        assert exported(machine, flag="8", OMP_NUM_THREADS="4")["OMP_NUM_THREADS"] == "8"

    def test_restore_returns_the_users_environment(self, machine):
        body = (
            'set_thread_env 8 ; restore_thread_env ; '
            'echo "omp=${OMP_NUM_THREADS-unset} blas=${OPENBLAS_NUM_THREADS-unset}"'
        )
        assert machine.run(body, OMP_NUM_THREADS="4").strip() == "omp=4 blas=unset"


class TestSourcedTwice:
    def test_the_users_environment_is_captured_once(self, machine):
        """A second source must not take this process's own exports for the user's."""
        body = 'set_thread_env 6 ; source "$THREADS_SH" ; resolve_threads "" 2 ; echo "$threads_budget|$threads_source"'
        assert machine.run(body, THREADS_SH=str(THREADS)).strip() == "2|default"


class TestCgroupQuota:
    """The quota is read from the process's own cgroup up to the root, as /proc/self/cgroup names it."""

    def test_a_quota_on_a_parent_cgroup(self, machine):
        machine.cgroup_v2("300000 100000", at="/system.slice", own="/system.slice/fs.service")
        assert machine.run("cgroup_cpu_quota").strip() == "3"

    def test_the_smallest_quota_on_the_way_up_wins(self, machine):
        machine.cgroup_v2("800000 100000", at="/system.slice", own="/system.slice/fs.service")
        machine.cgroup_v2("300000 100000", at="/system.slice/fs.service", own="/system.slice/fs.service")
        assert machine.run("cgroup_cpu_quota").strip() == "3"


@pytest.fixture
def brun(machine, tmp_path):
    """Run brun_fastsurfer.sh on three cases with a stub that records the options each case got.

    Returns the options per subject and brun's output. <lines> adds options to subject lines.
    """
    stub = tmp_path / "stub.sh"
    stub.write_text('#!/bin/bash\necho "$*" >> "$STUB_LOG"\n')
    stub.chmod(0o755)
    log = tmp_path / "stub.log"
    images = {}
    for name in ("subj1", "subj2", "subj3"):
        images[name] = tmp_path / f"{name}.mgz"
        images[name].write_bytes(b"")

    def run(*args: str, lines: dict[str, str] | None = None, **env: str) -> tuple[dict[str, str], str]:
        listfile = tmp_path / "subjects.txt"
        listfile.write_text("".join(f"{n}={p} {(lines or {}).get(n, '')}\n" for n, p in images.items()))
        result = subprocess.run(
            ["bash", str(FASTSURFER_HOME / "brun_fastsurfer.sh"), "--sd", str(tmp_path / "out"),
             "--run_fastsurfer", str(stub), "--subject_list", str(listfile), *args],
            capture_output=True, text=True, env=machine.environment(STUB_LOG=str(log), **env),
            cwd=FASTSURFER_HOME, timeout=120, check=True,
        )
        received = {line.split("--sid ")[1].split()[0]: line for line in log.read_text().splitlines()}
        return received, result.stdout

    return run


def shared(machine, processes: int, cap: int = 8) -> int:
    """What brun should pass, from share_threads in the same environment, so the host's cgroup cannot decide."""
    return min(int(machine.run(f"share_threads {processes}", real_cgroup=True)), cap)


class TestBrunSharesTheMachine:
    def test_parallel_cases_share(self, machine, brun):
        machine.cpus(total=32, physical=16)
        received, _ = brun("--parallel", "3", "--device", "cpu")
        expected = f"--threads_seg {shared(machine, 3)} --threads_surf {shared(machine, 3)}"
        assert all(expected in args for args in received.values())

    def test_more_parallel_slots_than_cases_count_the_cases(self, machine, brun):
        machine.cpus(total=32, physical=16)
        received, _ = brun("--parallel", "max", "--device", "cpu")
        assert all(f"--threads_seg {shared(machine, 3)}" in args for args in received.values())

    def test_two_pipelines_count_both(self, machine, brun):
        machine.cpus(total=32, physical=16)
        received, _ = brun("--parallel_seg", "2", "--parallel_surf", "2", "--device", "cpu")
        assert all(f"--threads_seg {shared(machine, 4)}" in args for args in received.values())

    def test_a_gpu_segmentation_gets_the_gpu_cap(self, machine, brun):
        machine.cpus(total=64, physical=32)
        received, _ = brun("--parallel", "2", "--device", "mps")
        expected = f"--threads_seg {shared(machine, 2, cap=4)} --threads_surf {shared(machine, 2)}"
        assert all(expected in args for args in received.values())

    def test_one_case_at_a_time_is_left_alone(self, brun):
        received, _ = brun("--parallel", "1", "--device", "cpu")
        assert not any("--threads" in args for args in received.values())

    def test_a_threads_flag_is_left_alone(self, brun):
        received, _ = brun("--parallel", "3", "--device", "cpu", "--threads", "2")
        assert not any("--threads_seg" in args for args in received.values())

    def test_omp_num_threads_is_left_alone(self, brun):
        received, _ = brun("--parallel", "3", "--device", "cpu", OMP_NUM_THREADS="2")
        assert not any("--threads" in args for args in received.values())

    def test_a_subject_line_with_its_own_threads_keeps_them_and_is_named(self, brun):
        received, out = brun("--parallel", "3", "--device", "cpu", lines={"subj2": "--threads 8"})
        # after the shared values, so that run_fastsurfer.sh takes the subject's own
        assert received["subj2"].index("--threads 8") > received["subj2"].index("--threads_surf")
        assert "1 subject line(s) set their own threads (subj2)" in out

    def test_passed_on_options_do_not_overwrite_each_other(self, brun):
        """brun collects --py apart from the other options, and each case must receive all of them."""
        received, _ = brun("--py", "python3", "--threads", "2", "--3T")
        for args in received.values():
            assert "--py python3" in args and "--threads 2" in args and "--3T" in args


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
