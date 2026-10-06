"""
Check tools/Docker/fastsurfer_startup.py, which names a user without passwd entry at interpreter
start.

``docker run --user $(id -u):$(id -g)``, the documented way to run the image, starts processes whose
uid has no passwd entry in the image, and getpass.getuser() then fails inside nibabel and monai.
The hook runs through the .pth file (python up to 3.14) or the .start file (from 3.15) next to it,
so these tests load tools/Docker the way site loads site-packages, with site.addsitedir, in an
interpreter started with -S, so that no installed copy of the hook runs first.
"""

import os
import pwd
import subprocess
import sys
from pathlib import Path

import pytest

STARTUP_DIR = Path(__file__).parent.parent.parent / "tools" / "Docker"


def getuser_after_startup(*, passwd_entry: bool) -> subprocess.CompletedProcess:
    """Process the startup files in a new interpreter, then print getuser() and USERNAME."""
    lines = ["import getpass, os, pwd, site"]
    if not passwd_entry:
        lines += [
            "def no_entry(uid):",
            "    raise KeyError(f'getpwuid(): uid not found: {uid}')",
            "pwd.getpwuid = no_entry",
        ]
    lines += [
        f"site.addsitedir({str(STARTUP_DIR)!r})",
        "print(getpass.getuser(), os.environ.get('USERNAME'))",
    ]
    return subprocess.run(
        [sys.executable, "-S", "-c", "\n".join(lines)],
        # none of the variables getuser() reads, as in a container started with --user
        env={"PATH": os.environ.get("PATH", "")},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_uid_without_passwd_entry_is_named_unknown():
    """Without the hook, getuser() raises here: KeyError, and OSError from python 3.13 on."""
    result = getuser_after_startup(passwd_entry=False)
    assert result.returncode == 0, result.stderr
    # the whole of stdout: the hook itself must print nothing, pin_cpu_dispatch.py parses it
    assert result.stdout.split() == ["UNKNOWN", "UNKNOWN"]


def test_known_user_keeps_their_name():
    """
    A user the system knows is not renamed.

    Setting USERNAME unconditionally, as an ENV in the Dockerfile once did, names every user
    UNKNOWN in the headers FastSurfer writes.
    """
    try:
        name = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        pytest.skip("the uid running the tests has no passwd entry itself")
    result = getuser_after_startup(passwd_entry=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [name, "None"]
