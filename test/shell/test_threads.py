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
Check how recon_surf/threads.sh turns flags, environment and machine into a thread budget.

The order is: --threads, else OMP_NUM_THREADS, else the default, with OMP_THREAD_LIMIT as a cap.
Every library variable is then exported from that budget, except that one the user set lower
stays a ceiling. "max" counts the CPUs the process may use, which a cgroup quota or a scheduler
allocation can lower without the CPU affinity showing it.

nproc and getconf are replaced by fakes on PATH and the cgroup files by temporary ones, so the
machine the tests run on, and the cgroup of a CI container, cannot decide the outcome.
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
    """A machine with 16 CPUs, an affinity of 16 and no cgroup quota, to be adjusted per test."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    class Machine:
        cgroup_v2 = tmp_path / "cpu.max"
        cgroup_v1_quota = tmp_path / "cpu.cfs_quota_us"
        cgroup_v1_period = tmp_path / "cpu.cfs_period_us"

        def cpus(self, total: int, affinity: int, physical: int | None = None):
            cores = "\n".join(f"{i},0" for i in range(physical or total))
            for name, body in (
                ("getconf", f"echo {total}"),
                ("nproc", f"echo {affinity}"),
                # lscpu -p=Core,Socket: one line per logical CPU, so hyperthreads repeat a core
                ("lscpu", f'echo "# Core,Socket"\nfor i in 1 2 ; do cat <<END\n{cores}\nEND\ndone'),
            ):
                fake = bin_dir / name
                fake.write_text(f"#!/bin/bash\n{body}\n")
                fake.chmod(0o755)

        def run(self, body: str, **env: str) -> str:
            script = (
                f'source "{THREADS}"\n'
                f'cgroup_v2_cpu_max="{self.cgroup_v2}"\n'
                f'cgroup_v1_cpu_quota="{self.cgroup_v1_quota}"\n'
                f'cgroup_v1_cpu_period="{self.cgroup_v1_period}"\n'
                f"{body}\n"
            )
            clean = {k: v for k, v in os.environ.items() if k not in THREAD_VARS + SCHEDULER_VARS}
            clean["PATH"] = f"{bin_dir}{os.pathsep}{clean['PATH']}"
            result = subprocess.run(
                ["bash", "-c", script], capture_output=True, text=True, env=clean | env, check=True,
            )
            return result.stdout

    m = Machine()
    m.cpus(total=16, affinity=16)
    return m


def budget(machine, flag: str = "", default: int | str = 2, **env: str) -> str:
    """The budget and its source, as resolve_threads leaves them."""
    return machine.run(
        f'resolve_threads "{flag}" {default} ; echo "$threads_budget|$threads_source"', **env,
    ).strip()


def exported(machine, flag: str = "", **env: str) -> dict[str, str]:
    """What set_thread_env exports for the resolved budget."""
    body = f'resolve_threads "{flag}" 2 ; set_thread_env "$threads_budget" ; env'
    lines = machine.run(body, **env).splitlines()
    return {k: v for k, _, v in (line.partition("=") for line in lines) if k in THREAD_VARS}


def available(machine, **env: str) -> str:
    out = machine.run('available_cpus ; echo "$cpus_available|$cpus_reason|$cpus_allocated"', **env)
    return out.strip()


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


class TestAvailableCpus:
    def test_the_whole_machine_is_not_an_allocation(self, machine):
        assert available(machine) == "16||false"

    def test_a_lower_affinity_counts_but_is_not_an_allocation(self, machine):
        machine.cpus(total=16, affinity=4)
        assert available(machine) == "4|the CPU affinity|false"

    def test_omp_num_threads_does_not_shrink_the_count(self, machine):
        """GNU nproc reads OMP_NUM_THREADS itself, which must not leak into max."""
        fake = Path(machine.cgroup_v2).parent / "bin" / "nproc"
        fake.write_text('#!/bin/bash\necho "${OMP_NUM_THREADS:-16}"\n')
        assert budget(machine, flag="max", OMP_NUM_THREADS="2") == "16|--threads max"

    def test_a_cgroup_v2_quota_lowers_the_count(self, machine):
        machine.cgroup_v2.write_text("400000 100000\n")
        assert available(machine) == "4|the cgroup CPU quota|true"

    def test_a_fractional_quota_rounds_down_but_not_below_one(self, machine):
        machine.cgroup_v2.write_text("150000 100000\n")
        assert available(machine).startswith("1|")
        machine.cgroup_v2.write_text("50000 100000\n")
        assert available(machine).startswith("1|")

    def test_an_unlimited_cgroup_v2_is_no_allocation(self, machine):
        machine.cgroup_v2.write_text("max 100000\n")
        assert available(machine) == "16||false"

    def test_a_cgroup_v1_quota_lowers_the_count(self, machine):
        machine.cgroup_v1_quota.write_text("200000\n")
        machine.cgroup_v1_period.write_text("100000\n")
        assert available(machine) == "2|the cgroup CPU quota|true"

    def test_an_unlimited_cgroup_v1_is_no_allocation(self, machine):
        machine.cgroup_v1_quota.write_text("-1\n")
        machine.cgroup_v1_period.write_text("100000\n")
        assert available(machine) == "16||false"

    @pytest.mark.parametrize("var", SCHEDULER_VARS)
    def test_a_scheduler_allocation_lowers_the_count(self, machine, var):
        assert available(machine, **{var: "6"}) == f"6|{var}|true"

    def test_a_scheduler_job_is_an_allocation_even_when_it_lowers_nothing(self, machine):
        """Slurm usually restricts the affinity too, so the variable matches the count."""
        machine.cpus(total=16, affinity=6)
        assert available(machine, SLURM_CPUS_PER_TASK="6") == "6|the CPU affinity|true"

    def test_max_reports_what_limited_it(self, machine):
        machine.cgroup_v2.write_text("400000 100000\n")
        assert budget(machine, flag="max") == "4|--threads max, limited by the cgroup CPU quota"


class TestAuto:
    def test_a_workstation_keeps_one_physical_core_free(self, machine):
        machine.cpus(total=8, affinity=8, physical=4)
        assert budget(machine, default="auto") == "3|auto, 4 cores with one kept free"

    def test_hyperthreads_do_not_count(self, machine):
        machine.cpus(total=32, affinity=32, physical=16)
        assert budget(machine, default="auto").startswith("8|auto, 16 cores with one kept free, capped at 8")

    def test_the_cap_is_the_one_passed(self, machine):
        machine.cpus(total=32, affinity=32, physical=16)
        out = machine.run('resolve_threads "" auto "$thread_auto_cap_gpu" ; echo "$threads_budget"').strip()
        assert out == "4"

    def test_a_single_core_runs_single_threaded(self, machine):
        machine.cpus(total=1, affinity=1, physical=1)
        assert budget(machine, default="auto").startswith("1|")

    def test_a_lower_affinity_limits_the_cores(self, machine):
        machine.cpus(total=32, affinity=4, physical=16)
        assert budget(machine, default="auto").startswith("3|auto, 4 cores")

    def test_an_allocation_is_used_whole(self, machine):
        machine.cgroup_v2.write_text("400000 100000\n")
        assert budget(machine, default="auto") == "4|auto, the allocation of 4 CPUs"

    def test_a_large_allocation_is_capped(self, machine):
        machine.cpus(total=64, affinity=32, physical=32)
        assert budget(machine, default="auto", SLURM_CPUS_PER_TASK="32").startswith("8|auto, the allocation")

    def test_omp_num_threads_still_wins_over_auto(self, machine):
        assert budget(machine, default="auto", OMP_NUM_THREADS="2") == "2|OMP_NUM_THREADS"

    def test_auto_can_be_asked_for_explicitly(self, machine):
        machine.cpus(total=8, affinity=8, physical=4)
        assert budget(machine, flag="auto", OMP_NUM_THREADS="2") == "3|--threads auto, 4 cores with one kept free"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
