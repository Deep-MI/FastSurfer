#!/usr/bin/env python3

"""Check that a FreeSurfer surface has valid volume metadata."""

from __future__ import annotations

import argparse
import sys

from nibabel.freesurfer.io import read_geometry


def options_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("surface", help="FreeSurfer surface to check")
    return parser.parse_args()


def main() -> int:
    args = options_parse()
    info = read_geometry(args.surface, read_metadata=True)[2]
    head = list(info.get("head", []))
    valid = str(info.get("valid", "")).startswith("1")
    if valid and head == [2, 0, 20]:
        return 0
    print(
        f"Invalid surface volume metadata in {args.surface}: "
        f"valid={info.get('valid')!r}, head={head!r}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
