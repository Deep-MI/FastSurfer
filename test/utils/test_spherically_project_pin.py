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
Check the CPU dispatch pin used by the steps that have to reproduce across machines.

numpy and OpenBLAS each choose kernels from the CPU features they detect at import, so the same
wheel computes slightly different numbers on different machines. In the spherical projection that
moved the projected sphere enough for the topology correction to make a different retessellation,
which every later surface inherited. In talairach-reg.sh it moved the registration transforms in
their last digits, and everything concatenated from them with it.

`pin_cpu_dispatch` takes the environment to pin as an argument, so these tests hand it a plain dict
and read back what it did. Stub interpreters stand in for numpy, so this runs in the light CI job.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from recon_surf.pin_cpu_dispatch import enabled_simd_features, pin_cpu_dispatch

FASTSURFER_HOME = Path(__file__).parent.parent.parent
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def stub(tmp_path, body, name="fakepython"):
    """An executable standing in for a python that reports SIMD features."""
    path = tmp_path / name
    path.write_text(f"#!/bin/bash\n{body}\n")
    path.chmod(0o755)
    return str(path)


class TestEnabledSimdFeatures:
    def test_reads_what_the_interpreter_reports(self, tmp_path):
        """The list has to come from the numpy in use, not from a table in our source."""
        assert enabled_simd_features(stub(tmp_path, "echo 'X86_V3 X86_V4 AVX512_ICL'")) == [
            "X86_V3", "X86_V4", "AVX512_ICL",
        ]

    def test_nothing_to_disable_is_an_empty_list(self, tmp_path):
        """
        A numpy with no dispatchable features enabled is already pinned.

        Distinct from a failure, because reporting it as one would warn that reproducibility
        was lost when it was not.
        """
        assert enabled_simd_features(stub(tmp_path, "echo ''")) == []

    def test_only_the_last_line_is_read(self, tmp_path):
        """
        Anything else on stdout, a sitecustomize or a chatty build, is not a feature name.

        numpy ignores names it does not know, so passing them through would silently leave the
        projection unpinned.
        """
        assert enabled_simd_features(
            stub(tmp_path, "printf 'build chatter\\nX86_V3 X86_V4\\n'")
        ) == ["X86_V3", "X86_V4"]

    def test_a_failing_interpreter_is_none_not_empty(self, tmp_path):
        """None and [] mean different things to the caller, and only None warrants a warning."""
        assert enabled_simd_features(stub(tmp_path, "echo boom >&2 ; exit 1")) is None

    def test_a_missing_interpreter_is_none(self):
        assert enabled_simd_features("/nonexistent/python") is None

    def test_it_never_raises(self, tmp_path):
        """
        It runs before any work starts.

        Aborting the projection because the feature list could not be read would trade a
        reproducibility property for a broken run.
        """
        for body in ("exit 3", "kill -9 $$", "echo x ; exit 1"):
            assert enabled_simd_features(stub(tmp_path, body)) in (None, ["x"])


class TestPinCpuDispatch:
    def test_pins_both_variables(self, tmp_path):
        env = {}
        pin_cpu_dispatch(env, stub(tmp_path, "echo 'X86_V3 X86_V4'"))
        assert env["OPENBLAS_CORETYPE"] == "Nehalem"
        assert env["NPY_DISABLE_CPU_FEATURES"] == "X86_V3 X86_V4"

    def test_the_core_type_is_never_an_upgrade(self, tmp_path):
        """
        Forcing a core type the CPU cannot execute faults rather than falling back.

        So the value has to be one any x86-64 machine can run, which rules out the faster
        Haswell that would otherwise have been enough.
        """
        env = {}
        pin_cpu_dispatch(env, stub(tmp_path, "echo ''"))
        assert env["OPENBLAS_CORETYPE"] == "Nehalem"

    def test_nothing_to_disable_leaves_the_numpy_variable_unset(self, tmp_path):
        """Setting it to an empty string would be a different thing to say than not setting it."""
        env = {}
        pin_cpu_dispatch(env, stub(tmp_path, "echo ''"))
        assert "NPY_DISABLE_CPU_FEATURES" not in env

    def test_a_value_the_user_set_is_kept(self, tmp_path):
        """
        An explicitly set variable wins, and is the only escape hatch from a faulting core type.

        Each is independent: overriding one must not stop the other being pinned.
        """
        env = {"OPENBLAS_CORETYPE": "Haswell"}
        pin_cpu_dispatch(env, stub(tmp_path, "echo 'X86_V4'"))
        assert env["OPENBLAS_CORETYPE"] == "Haswell"
        assert env["NPY_DISABLE_CPU_FEATURES"] == "X86_V4"

        env = {"NPY_DISABLE_CPU_FEATURES": "AVX512_ICL"}
        pin_cpu_dispatch(env, stub(tmp_path, "echo 'X86_V4'"))
        assert env["NPY_DISABLE_CPU_FEATURES"] == "AVX512_ICL"
        assert env["OPENBLAS_CORETYPE"] == "Nehalem"

    def test_keeping_a_user_value_says_so(self, tmp_path, capsys):
        env = {"OPENBLAS_CORETYPE": "Haswell"}
        pin_cpu_dispatch(env, stub(tmp_path, "echo ''"))
        out = capsys.readouterr().out
        assert "OPENBLAS_CORETYPE" in out and "Haswell" in out

    def test_it_does_not_claim_the_run_will_not_reproduce(self, tmp_path, capsys):
        """
        A user-set value can be perfectly reproducible if it is valid everywhere they compare.

        Predicting failure would be wrong in that case, and a warning that is sometimes wrong
        gets ignored when it is right.
        """
        pin_cpu_dispatch({"OPENBLAS_CORETYPE": "Haswell"}, stub(tmp_path, "echo ''"))
        out = capsys.readouterr().out
        assert "requires that" in out
        assert "will not reproduce" not in out

    def test_pinning_quietly_when_there_is_nothing_to_report(self, tmp_path, capsys):
        """A warning on every ordinary run is a warning nobody reads."""
        pin_cpu_dispatch({}, stub(tmp_path, "echo 'X86_V3'"))
        assert capsys.readouterr().out == ""

    def test_nothing_to_disable_is_not_a_warning(self, tmp_path, capsys):
        """
        An empty list is the pinned state, not a failure.

        This is the case that separates 'could not read the features' from 'there are none',
        and warning about the second would make the first easy to miss.
        """
        pin_cpu_dispatch({}, stub(tmp_path, "echo ''"))
        assert capsys.readouterr().out == ""

    def test_an_unreadable_list_warns_but_still_pins_openblas(self, tmp_path, capsys):
        env = {}
        pin_cpu_dispatch(env, stub(tmp_path, "exit 1"))
        assert "could not read" in capsys.readouterr().out
        assert env["OPENBLAS_CORETYPE"] == "Nehalem"


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy is not installed in this job")
def test_the_pinned_environment_actually_silences_numpy_dispatch():
    """
    The end the whole thing exists for: numpy started with this environment dispatches to nothing.

    Uses the real interpreter and the real numpy rather than a stub, so it also catches the
    feature names being read in a form numpy does not accept.
    """
    env = {}
    pin_cpu_dispatch(env, sys.executable)
    code = (
        "try:\n"
        "    from numpy._core._multiarray_umath import __cpu_dispatch__ as d, __cpu_features__ as f\n"
        "except ImportError:\n"
        "    from numpy.core._multiarray_umath import __cpu_dispatch__ as d, __cpu_features__ as f\n"
        "print(' '.join(x for x in d if f[x]))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, **env},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", "numpy still dispatches despite the pin"


def test_the_wrapper_imports_without_numpy():
    """
    The module must stay importable in the light CI job, which installs pytest and nothing else.

    It is also what lets the pin happen before numpy loads: a numpy import at module level
    would make the whole exercise pointless.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['numpy'] = None; "
         "import recon_surf.spherically_project_wrapper as w; print(w.__name__)"],
        env={**os.environ, "PYTHONPATH": str(FASTSURFER_HOME)},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestShellForm:
    """
    The `python pin_cpu_dispatch.py` form, which talairach-reg.sh and run_fastsurfer.sh eval.

    Nothing else covers it: a regression here produces output that eval ignores or chokes on, and
    both callers treat a zero exit as success, so the step would run unpinned while the log says it
    succeeded. That is the failure this pinning exists to remove.
    """

    SCRIPT = FASTSURFER_HOME / "recon_surf" / "pin_cpu_dispatch.py"

    def run(self, env_extra=None):
        env = {**os.environ, "PYTHONPATH": str(FASTSURFER_HOME)}
        for key in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES"):
            env.pop(key, None)
        env.update(env_extra or {})
        return subprocess.run(
            [sys.executable, str(self.SCRIPT)], env=env, capture_output=True, text=True, check=True,
        ).stdout

    def eval_in_bash(self, stdout, env_extra=None):
        """
        What the callers do with it: eval, then report what landed in the environment.

        env_extra has to match what the script was run with, since a variable it deliberately left
        alone is only visible here if this shell inherited it too.
        """
        script = f'{stdout}\nprintf "%s|%s" "${{OPENBLAS_CORETYPE-}}" "${{NPY_DISABLE_CPU_FEATURES-}}"'
        env = {**os.environ}
        for key in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES"):
            env.pop(key, None)
        env.update(env_extra or {})
        done = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
        assert done.returncode == 0, f"eval of the emitted text failed: {done.stderr}"
        return done.stdout.split("|")

    @pytest.mark.skipif(not HAS_NUMPY, reason="needs a numpy to read features from")
    def test_emits_a_core_type_that_survives_eval(self):
        coretype, _ = self.eval_in_bash(self.run())
        assert coretype == "Nehalem"

    @pytest.mark.skipif(not HAS_NUMPY, reason="needs a numpy to read features from")
    def test_an_already_set_value_is_left_alone_and_only_commented(self):
        already = {"OPENBLAS_CORETYPE": "Haswell"}
        stdout = self.run(already)
        assert "export OPENBLAS_CORETYPE" not in stdout
        coretype, _ = self.eval_in_bash(stdout, already)
        assert coretype == "Haswell"

    @pytest.mark.skipif(not HAS_NUMPY, reason="needs a numpy to read features from")
    def test_a_value_carrying_a_newline_cannot_break_out_of_the_comment(self):
        """A `#` comment ends at the newline, so an unquoted value would leave a line to run."""
        hostile = {"OPENBLAS_CORETYPE": "Haswell\ntouch pwned"}
        stdout = self.run(hostile)
        # must not raise, so the emitted text is still valid shell despite the newline
        self.eval_in_bash(stdout, hostile)
        assert not Path("pwned").exists(), "a value with a newline escaped into an executed line"

    def test_every_non_comment_line_is_an_export(self):
        """eval gets only assignments, whatever the interpreter reports."""
        for line in self.run().splitlines():
            assert line.startswith(("#", "export ")), f"not evalable: {line!r}"
