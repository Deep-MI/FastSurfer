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
        if env.get(var) == value:
            # already what this would set, so there is nothing to do and nothing to report. The
            # pipeline test pins the whole container, which is how this arises in CI.
            continue
        if var in env:
            print(f"WARNING: {var} is already set to '{env[var]}', so FastSurfer is not")
            print("  pinning it. Reproducing this step on another machine then requires that")
            print("  value to be supported and identical there.")
        else:
            env[var] = value


if __name__ == "__main__":
    import argparse
    import os
    import shlex
    import sys

    _parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    _parser.add_argument(
        "--env",
        action="store_true",
        help="print bare VAR=value lines for `env` or `docker run --env` instead of shell "
             "assignments for `eval`. Warnings then go to stderr, since every stdout line is "
             "consumed as an assignment.",
    )
    _args = _parser.parse_args()

    # Two forms, because the callers differ. Default is shell: assignments plus warnings as `#`
    # comments, so one stream is both safe to eval and worth appending to the log. --env is for a
    # caller that turns each line into an argument, where a comment would be read as a variable.
    #
    # A value that already matches is passed over in silence: the pipeline test pins the whole
    # container, so every step would otherwise report it.
    _pins, _warning = pins_for(sys.executable)

    def _note(text: str) -> None:
        if _args.env:
            print(text, file=sys.stderr)
        else:
            for _line in text.splitlines():
                print(f"# {_line}")

    if _warning:
        _note(_warning)
    for _var, _value in _pins.items():
        if os.environ.get(_var) == _value:
            continue
        if _var in os.environ:
            # quoted, because a value carrying a newline would otherwise end a `#` comment and
            # leave the rest of it as a line the caller's eval would try to run
            _note(
                f"WARNING: {_var} is already set to {shlex.quote(os.environ[_var])!r}, left alone.\n"
                f"  Reproducing this step elsewhere then needs that value supported and identical there."
            )
        elif _args.env:
            # one argv element per line, so no quoting: the caller must not word-split these
            print(f"{_var}={_value}")
        else:
            print(f"export {_var}={shlex.quote(_value)}")
