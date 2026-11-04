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
Guard the build log that version.py writes for every run.

A section that cannot be determined used to discard the whole output, so the log of such a run
recorded no version at all.
"""

import os
import re
import subprocess
import sys
import venv
from pathlib import Path

FASTSURFER_HOME = Path(__file__).parent.parent.parent


def test_a_failed_section_keeps_the_others(tmp_path):
    # an interpreter without pip, so the python packages section fails for real
    venv.create(tmp_path / "venv", with_pip=False)
    python = tmp_path / "venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    output = tmp_path / "BUILD.log"
    result = subprocess.run(
        [python, FASTSURFER_HOME / "FastSurferCNN" / "version.py", "--sections", "+pip", "-o", output],
        env={**os.environ, "PYTHONPATH": str(FASTSURFER_HOME)},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "pypackages" in result.stderr
    version_line, *rest = output.read_text().splitlines()
    assert re.match(r"\d+\.\d+\.\d+", version_line)
    assert "python packages:" not in rest

    # appended to a build log, the report is recorded and the failed section only warns
    build_log = tmp_path / "build.log"
    result = subprocess.run(
        [python, FASTSURFER_HOME / "FastSurferCNN" / "version.py", "--sections", "+pip", "--append", build_log],
        env={**os.environ, "PYTHONPATH": str(FASTSURFER_HOME)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "pypackages" in result.stdout
    assert re.match(r"\d+\.\d+\.\d+", build_log.read_text())


INVOCATION = "\n==========\ninvocation:\n==========\nrun_fastsurfer.sh --sid s\nMon, 2 Nov 2026 10:00:00 +0100\n"


def version_info(version_line: str, packages: str) -> str:
    return f"{version_line}\n==========\npython packages:\n==========\n{packages}\n"


def test_build_log_keeps_every_version(tmp_path):
    from FastSurferCNN.version import append_build_log, last_version_block

    build_log = tmp_path / "build.log"
    first = version_info("2.6.0+1dc02473 (stable)", "numpy 2.5.3")

    def run(info: str) -> str | None:
        warning = append_build_log(build_log, info)
        with open(build_log, "a") as file:
            file.write(INVOCATION)
        return warning

    # the first run writes the version info, a second run of the same version only its invocation
    assert run(first) is None
    assert run(first) is None
    content = build_log.read_text()
    assert content.splitlines()[0] == "2.6.0+1dc02473 (stable)"
    assert content.count("python packages:") == 1 and content.count("invocation:") == 2
    # the same version in another environment adds its version info, with a warning
    drifted = version_info("2.6.0+1dc02473 (stable)", "numpy 2.6.0")
    assert "another environment" in run(drifted)
    # the version line alone, recorded when the full version info failed, tells no environment
    assert run("2.6.0+1dc02473 (stable)\n") is None
    # another version adds its version info, which the next run compares with
    other = version_info("2.6.1+019ee29a (stable)", "numpy 2.6.0")
    assert run(other) is None
    assert build_log.read_text().count("python packages:") == 3
    assert build_log.read_text().count("invocation:") == 5
    assert last_version_block(build_log.read_text()) == other.rstrip("\n")


def test_version_difference():
    from FastSurferCNN.version import version_difference

    # the branch and the length of the abbreviated commit hash do not matter
    assert version_difference("2.6.0+1dc02473 (stable)", "2.6.0+1dc0247") == ""
    # another patch release, development commit or development version of the same minor version warns
    assert version_difference("2.6.0+1dc02473 (stable)", "2.6.1+019ee29a (stable)") == "patch"
    assert version_difference("2.7.0.dev0+019ee29a (dev)", "2.7.0.dev0+48e42be9 (dev)") == "patch"
    assert version_difference("2.7.0.dev0+019ee29a (dev)", "2.7.0+5f3c2a10 (stable)") == "patch"
    # another major or minor version stops, also on a branch whose name has parentheses
    assert version_difference("2.5.4+aa11bb22 (stable)", "2.6.1+019ee29a (stable)") == "minor"
    assert version_difference("2.5.4+aa11bb22 (fix(mask))", "2.6.1+019ee29a (stable)") == "minor"
    assert version_difference("2.6.1+019ee29a (stable)", "3.0.0+5f3c2a10 (stable)") == "minor"
