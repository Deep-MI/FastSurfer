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
Pin which vectorised kernels numpy and OpenBLAS select, for steps that have to reproduce.

Both read their setting once, at import, so a caller in Python must pin before importing numpy
and a caller in shell must pin before starting the process. Run as a script this prints the
assignments for the shell form:

    eval "$(python3 recon_surf/pin_cpu_dispatch.py)"

Used by the spherical projection and by talairach-reg.sh. Those are the steps where a kernel
difference has been measured to survive into the output: both end in a decomposition, which
turns a last-bit difference in the input into a visible one in the result.
"""

# IMPORTS
from collections.abc import MutableMapping

__all__ = ["enabled_simd_features", "pin_cpu_dispatch", "pins_for"]


def enabled_simd_features(python: str) -> list[str] | None:
    """
    Ask a numpy which SIMD extensions it would dispatch to.

    In a throwaway process, because the answer has to be known before this one imports
    numpy. Read rather than hardcoded: the names differ by numpy version and platform,
    unknown ones are ignored, but disabling a baseline feature raises.

    Parameters
    ----------
    python : str
        The interpreter to ask, normally `sys.executable`.

    Returns
    -------
    list[str], None
        The dispatchable features this numpy has enabled. An empty list means there are
        none to disable, which is a pinned state rather than a failure. None means the
        list could not be read at all, so nothing can be pinned.
    """
    import subprocess

    code = (
        "try:\n"
        "    from numpy._core._multiarray_umath import __cpu_dispatch__ as d, __cpu_features__ as f\n"
        "except ImportError:\n"
        "    from numpy.core._multiarray_umath import __cpu_dispatch__ as d, __cpu_features__ as f\n"
        "print(' '.join(x for x in d if f[x]))\n"
    )
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    # the last line only: anything else that writes to stdout, a sitecustomize or a chatty
    # build, would otherwise be passed to numpy as feature names
    lines = out.stdout.strip().splitlines()
    return lines[-1].split() if lines else []


def pins_for(python: str) -> tuple[dict[str, str], str | None]:
    """
    The variables to pin, and a warning if the numpy half could not be worked out.

    Parameters
    ----------
    python : str
        The interpreter whose numpy decides the feature list, normally `sys.executable`.

    Returns
    -------
    dict[str, str]
        Variable names and the values to pin them to.
    str, None
        A warning to show the user, or None when there is nothing to report.
    """
    pins = {"OPENBLAS_CORETYPE": "Nehalem"}
    simd = enabled_simd_features(python)
    warning = None
    if simd is None:
        # only a failure to read is worth warning about; an empty list means that numpy has
        # nothing dispatchable enabled, which is already the pinned state
        warning = (
            "WARNING: could not read numpy's SIMD features, so numpy is not pinned here and\n"
            "  this step may not reproduce on other hardware."
        )
    elif simd:
        pins["NPY_DISABLE_CPU_FEATURES"] = " ".join(simd)
    return pins, warning


def pin_cpu_dispatch(env: MutableMapping[str, str], python: str) -> None:
    """
    Pin which vectorised kernels numpy and OpenBLAS choose.

    Both read their setting once, when they are imported, so this has to run before numpy
    reaches the interpreter. Without it the same wheel computes slightly different numbers
    on different machines.

    A value already in `env` is left alone. An explicitly set variable should take effect,
    and it is the only escape hatch: forcing a core type the CPU cannot execute faults
    rather than falling back, so someone on unusual hardware needs a way to override this.
    Such a value may well be reproducible, it is simply not the one tested here, hence a
    note about the condition rather than a prediction of trouble.

    Parameters
    ----------
    env : MutableMapping[str, str]
        The environment to pin, normally `os.environ`.
    python : str
        The interpreter whose numpy decides the feature list, normally `sys.executable`.
    """
    pins, warning = pins_for(python)
    if warning:
        print(warning)

    for var, value in pins.items():
        if var in env:
            print(f"WARNING: {var} is already set to '{env[var]}', so FastSurfer is not")
            print("  pinning it. Reproducing this step on another machine then requires that")
            print("  value to be supported and identical there.")
        else:
            env[var] = value


if __name__ == "__main__":
    import os
    import shlex
    import sys

    # Shell form. Everything goes to stdout, warnings as `#` comments, so that one stream is both
    # safe to eval and worth appending to the log: eval ignores comments, and the caller keeps the
    # record of what was pinned without having to interleave two streams.
    # An already-set value is left alone here too, and says so rather than emitting an assignment.
    _pins, _warning = pins_for(sys.executable)
    if _warning:
        for _line in _warning.splitlines():
            print(f"# {_line}")
    for _var, _value in _pins.items():
        if _var in os.environ:
            print(f"# WARNING: {_var} is already set to '{os.environ[_var]}', so it is left alone.")
            print("#   Reproducing this step elsewhere then needs that value supported and identical there.")
        else:
            print(f"export {_var}={shlex.quote(_value)}")
