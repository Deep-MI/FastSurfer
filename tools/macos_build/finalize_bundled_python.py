#!/usr/bin/env python3

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
Make a bundled python distribution independent of where it was built.

The macOS installer builds its python distribution in a staging directory and ships it to
/Applications/FastSurfer<version>. The interpreter itself needs no help -- it derives its prefix
from its own location, so the tree can be moved anywhere -- but two kinds of file record the build
path and have to be corrected:

* **console scripts** in ``bin/``. pip and uv write these as ``#!/bin/sh`` wrappers that ``exec``
  the interpreter by absolute path (the workaround for shebangs being length-limited), so each of
  the ~57 of them -- including ``pip`` itself and neuroreg's ``coreg``/``robreg`` -- would exec an
  interpreter that does not exist on the user's machine.
* **compiled bytecode**. A ``.pyc`` stores the absolute path of its source for tracebacks, and it
  is binary, so it cannot be rewritten as text. These are simply removed; the installer's
  postinstall regenerates them, with correct paths, once the files are in place.

Rewriting text in place is safe here because the staging path is a long, unambiguous absolute path
that cannot appear in these files for any other reason.
"""

import argparse
import sys
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    """
    Create the command line interface.

    Returns
    -------
    argparse.ArgumentParser
        The argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Retarget a bundled python distribution from its build path to its install path",
    )
    parser.add_argument(
        "--dist",
        type=Path,
        required=True,
        help="the bundled python distribution (the directory holding bin/ and lib/)",
    )
    parser.add_argument(
        "--from",
        dest="from_prefix",
        type=str,
        required=True,
        help="the build-time prefix to replace (the staged FastSurfer directory)",
    )
    parser.add_argument(
        "--to",
        dest="to_prefix",
        type=str,
        required=True,
        help="the prefix the package is installed to (e.g. /Applications/FastSurfer2.6.0)",
    )
    return parser


def is_text(path: Path) -> bool:
    """
    Report whether a file looks like text rather than a binary.

    Parameters
    ----------
    path : Path
        The file to inspect.

    Returns
    -------
    bool
        True if the first block decodes as UTF-8 and holds no NUL byte.
    """
    try:
        block = path.open("rb").read(8192)
    except OSError:
        return False
    if b"\0" in block:
        return False
    try:
        block.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def rewrite(path: Path, from_prefix: str, to_prefix: str) -> bool:
    """
    Replace every occurrence of a prefix in a text file, preserving its mode.

    Parameters
    ----------
    path : Path
        The file to rewrite.
    from_prefix : str
        The string to replace.
    to_prefix : str
        The replacement.

    Returns
    -------
    bool
        True if the file was changed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if from_prefix not in text:
        return False
    mode = path.stat().st_mode
    path.write_text(text.replace(from_prefix, to_prefix), encoding="utf-8")
    path.chmod(mode)
    return True


def main() -> int:
    """
    Retarget the distribution and verify that nothing still points at the build path.

    Returns
    -------
    int
        0 on success, 1 if the distribution looks wrong or a stale path remains.
    """
    args = make_parser().parse_args()
    dist: Path = args.dist
    # strip trailing slashes so the prefixes compare and concatenate consistently
    from_prefix = args.from_prefix.rstrip("/")
    to_prefix = args.to_prefix.rstrip("/")

    if not (dist / "bin").is_dir() or not (dist / "lib").is_dir():
        print(f"ERROR: {dist} does not look like a python distribution", file=sys.stderr)
        return 1

    # bytecode first: it is unrewritable and would otherwise trip the verification below
    caches = [p for p in dist.rglob("__pycache__") if p.is_dir()]
    for cache in caches:
        for entry in cache.iterdir():
            entry.unlink()
        cache.rmdir()
    print(f"  removed {len(caches)} __pycache__ director{'y' if len(caches) == 1 else 'ies'}")

    if from_prefix == to_prefix:
        print(f"nothing to retarget: build and install prefix are both {to_prefix}")
        return 0

    changed = []
    # bin/ holds the console-script wrappers. Skip binaries: the interpreter lives here too and
    # must not be touched.
    for entry in sorted((dist / "bin").iterdir()):
        if entry.is_file() and not entry.is_symlink() and is_text(entry):
            if rewrite(entry, from_prefix, to_prefix):
                changed.append(entry)
    # .pth files can also carry absolute paths
    for pth in sorted(dist.rglob("*.pth")):
        if rewrite(pth, from_prefix, to_prefix):
            changed.append(pth)
    print(f"  retargeted {len(changed)} file(s): {from_prefix} -> {to_prefix}")

    # Verify. A leftover would surface as a broken installed package rather than a failed build,
    # so fail here instead.
    stale = []
    for entry in dist.rglob("*"):
        if entry.is_file() and not entry.is_symlink() and is_text(entry):
            try:
                if from_prefix in entry.read_text(encoding="utf-8"):
                    stale.append(entry)
            except (OSError, UnicodeDecodeError):
                continue
    if stale:
        print(f"ERROR: {len(stale)} file(s) still reference the build path {from_prefix}:", file=sys.stderr)
        for entry in stale[:20]:
            print(f"  {entry.relative_to(dist)}", file=sys.stderr)
        return 1

    print("  verified: no file references the build path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
