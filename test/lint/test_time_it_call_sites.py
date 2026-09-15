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
Keep every time_it call site in run_fastsurfer.sh guarded by its own status check.

time_it returns the wrapped command's status rather than exiting, so that the caller can name
the step that failed before stopping. That is only safe while every caller actually tests the
status: a new call site without a check would carry on past a failed step and report success.

This reads the script and never runs it, so it is a lint rather than a test of behaviour, which
is why it lives here. The functional half, that the contract holds in both the piped and the
unpiped form, is test/shell/test_time_it.py.
"""

import re
from pathlib import Path

import pytest

FASTSURFER_HOME = Path(__file__).parent.parent.parent
RUN_FASTSURFER = FASTSURFER_HOME / "run_fastsurfer.sh"

# how many lines below a call site still count as its check
_CHECK_WINDOW = 8
# the check has to read the status, not merely mention an error: a nearby `echo "ERROR: ..."`
# belonging to something else would otherwise satisfy this scan while the failure goes unhandled
_READS_STATUS = re.compile(r"PIPESTATUS\[0\]|exit_code")


def test_the_script_was_found() -> None:
    """A missing file would make the scan below pass while checking nothing."""
    assert RUN_FASTSURFER.is_file(), f"{RUN_FASTSURFER} is not there"


def test_every_call_site_checks_the_status() -> None:
    """
    Returning is only safe while every caller tests the status itself.

    A new call site without a check would silently carry on past a failed step, which is the
    one regression the return could introduce and the one a runtime test cannot catch.
    """
    lines = RUN_FASTSURFER.read_text().splitlines()
    call_sites = [i for i, line in enumerate(lines) if '"${wrap[@]}"' in line]
    assert call_sites, 'no time_it call sites found; has the `wrap` idiom been renamed?'
    unchecked = [
        i + 1
        for i in call_sites
        if not any(_READS_STATUS.search(line) for line in lines[i + 1: i + 1 + _CHECK_WINDOW])
    ]
    assert not unchecked, f"time_it call sites that never read the status, at lines: {unchecked}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
