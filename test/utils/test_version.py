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
