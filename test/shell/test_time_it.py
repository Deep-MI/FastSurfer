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
Check the failure contract of time_it, the per-step wrapper in recon_surf/functions.sh.

time_it returns the wrapped command's status and leaves the decision to the caller, which is
what lets run_fastsurfer.sh name the step that failed before it stops. It used to call `exit`
instead. A function's exit ends the shell it runs in, which is the caller for a plain call but
only a subshell inside a pipeline, so 14 of the 19 call sites had an error message that could
never print while the other 5 worked. The two forms have to behave the same, which is what
these tests pin.

The other half of the contract is that the status still reaches the caller unchanged, because
every call site reads it from PIPESTATUS and exits with it. Losing that would turn a failed
step into a run that reports success.

Returning is only safe while every call site really does check. That half only reads the script
and never runs it, so it is a lint: test/lint/test_time_it_call_sites.py.
"""

import subprocess
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent
FUNCTIONS = FASTSURFER_HOME / "recon_surf" / "functions.sh"

# the caller's shape, both as it appears with and without a pipe in run_fastsurfer.sh
CALLER = """
source "{functions}" > /dev/null 2>&1
{timecmd_override}
wrap=("time_it" "/dev/null")
"${{wrap[@]}}" {command} {pipe}
exit_code="${{PIPESTATUS[0]}}"
if [[ "$exit_code" != 0 ]] ; then
  echo "ERROR: the step failed"
  exit "$exit_code"
fi
echo "CONTINUED"
"""


def run_caller(command: str, piped: bool, timed: bool) -> subprocess.CompletedProcess:
    """Run one call site of the shape run_fastsurfer.sh uses."""
    script = CALLER.format(
        functions=FUNCTIONS,
        # functions.sh probes fs_time and leaves timecmd empty where it does not work, so both
        # branches have to be forced rather than left to the platform
        timecmd_override="" if timed else 'timecmd=""',
        command=command,
        pipe="2>&1 | tee /dev/null" if piped else "",
    )
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


@pytest.mark.parametrize("piped", [False, True], ids=["unpiped", "piped"])
@pytest.mark.parametrize("timed", [False, True], ids=["untimed", "timed"])
class TestFailureIsReported:
    def test_the_caller_message_is_reached(self, piped, timed):
        """The whole point: exiting inside the function skipped this line when unpiped."""
        result = run_caller("false", piped, timed)
        assert "ERROR: the step failed" in result.stdout
        assert "CONTINUED" not in result.stdout

    def test_the_caller_stops(self, piped, timed):
        """Returning must not turn a failure into a run that carries on."""
        result = run_caller("false", piped, timed)
        assert result.returncode != 0

    @pytest.mark.parametrize("code", [1, 7, 42])
    def test_the_exit_code_survives(self, piped, timed, code):
        result = run_caller(f'bash -c "exit {code}"', piped, timed)
        assert result.returncode == code


@pytest.mark.parametrize("piped", [False, True], ids=["unpiped", "piped"])
@pytest.mark.parametrize("timed", [False, True], ids=["untimed", "timed"])
def test_success_continues(piped, timed):
    result = run_caller("true", piped, timed)
    assert result.returncode == 0
    assert "CONTINUED" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
