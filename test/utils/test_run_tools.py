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
Guard Popen.finish against children that write more than a pipe holds.

version.py runs `pip list --verbose` through finish; with several hundred packages its output
exceeds the pipe buffer, and a finish that waits before reading never sees the child exit.
"""

import subprocess
import sys

from FastSurferCNN.utils.run_tools import Popen

# well above the 64 KiB pipe buffer of Linux and macOS
SIZE = 1 << 20


def test_finish_reads_output_larger_than_a_pipe():
    script = f"import sys; sys.stdout.write('o' * {SIZE}); sys.stderr.write('e' * {SIZE})"
    process = Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = process.finish(timeout=30.0)
    assert result.retcode == 0
    assert result.out == b"o" * SIZE
    assert result.err == b"e" * SIZE
    assert result.runtime < 30.0


def test_finish_stops_a_child_that_outlives_the_timeout():
    process = Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdout=subprocess.PIPE)
    result = process.finish(timeout=0.5)
    assert result.retcode != 0
    assert result.runtime < 10.0
