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
Check the CPU dispatch pin in the spherical projection wrapper.

numpy and OpenBLAS each choose kernels from the CPU features they detect at import, so the same
wheel computes slightly different numbers on different machines. In this step that moved the
projected sphere enough for the topology correction to make a different retessellation, which every
later surface inherited. The wrapper pins both before importing numpy.

These tests use stub interpreters rather than numpy, so they run in the light CI job, and they pin
the two properties that make the real thing work: the feature list is read from the interpreter
rather than hardcoded, and an unreadable list degrades loudly instead of silently.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from recon_surf.spherically_project_wrapper import enabled_simd_features

FASTSURFER_HOME = Path(__file__).parent.parent.parent
WRAPPER = FASTSURFER_HOME / "recon_surf" / "spherically_project_wrapper.py"


def stub(tmp_path, body):
    """An executable standing in for a python that reports SIMD features."""
    path = tmp_path / "fakepython"
    path.write_text(f"#!/bin/bash\n{body}\n")
    path.chmod(0o755)
    return str(path)


def test_reads_the_features_the_interpreter_reports(tmp_path):
    """The list has to come from the numpy in use, not from a table in our source."""
    assert enabled_simd_features(stub(tmp_path, "echo 'X86_V3 X86_V4 AVX512_ICL'")) == [
        "X86_V3", "X86_V4", "AVX512_ICL",
    ]


def test_no_dispatchable_features_is_an_empty_list(tmp_path):
    assert enabled_simd_features(stub(tmp_path, "echo ''")) == []


def test_a_failing_interpreter_yields_an_empty_list(tmp_path):
    """
    Never raise out of this.

    It runs before any work starts, and aborting the projection because the feature list
    could not be read would trade a reproducibility property for a broken run.
    """
    assert enabled_simd_features(stub(tmp_path, "echo boom >&2 ; exit 1")) == []


def test_a_missing_interpreter_yields_an_empty_list():
    assert enabled_simd_features("/nonexistent/python") == []


def test_the_real_interpreter_answers():
    """Whatever this platform dispatches on, reading it must not fail."""
    assert isinstance(enabled_simd_features(sys.executable), list)


class TestTheWrapperPinsBeforeImportingNumpy:
    """
    Both variables are read once, when numpy and OpenBLAS load.

    Setting them after the import silently does nothing, which is the failure this class
    exists to catch.
    """

    @staticmethod
    def _source():
        return WRAPPER.read_text()

    def test_both_variables_are_set(self):
        src = self._source()
        assert 'environ["OPENBLAS_CORETYPE"]' in src
        assert 'environ["NPY_DISABLE_CPU_FEATURES"]' in src

    def test_the_core_type_is_a_baseline_one(self):
        """
        It must be a downgrade on every target, never an upgrade.

        Forcing a core type the CPU cannot execute is not a fallback, it is an illegal
        instruction, so this must stay at a level any x86-64 machine can run.
        """
        assert 'environ["OPENBLAS_CORETYPE"] = "Nehalem"' in self._source()

    def test_the_pin_precedes_the_numpy_import(self):
        src = self._source()
        pin = src.index('environ["OPENBLAS_CORETYPE"]')
        # the projection is imported lazily inside __main__, and that is what pulls in numpy
        use = src.index("from recon_surf.spherically_project import")
        assert pin < use, "the dispatch pin must be set before numpy is imported"

    def test_an_unreadable_feature_list_warns(self):
        """Losing the pin silently would leave a run that cannot be reproduced and says so nowhere."""
        assert "WARNING: could not read numpy's SIMD features" in self._source()


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
