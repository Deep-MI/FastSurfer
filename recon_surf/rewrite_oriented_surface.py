# Copyright 2024 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
import argparse
import shutil

# IMPORTS
import sys
from pathlib import Path

import lapy

__version__ = "1.0"


def make_parser() -> argparse.ArgumentParser:
    """
    Create a command line interface and return command line options.

    Returns
    -------
    options
        Namespace object holding options.
    """
    parser = argparse.ArgumentParser(
        description="Script to load and safe surface (that are guaranteed to be "
                    "correctly oriented) under a given name",
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        dest="file",
        help="path to surface to check and fix",
        required=True,
    )
    parser.add_argument(
        "--backup",
        type=Path,
        dest="backup",
        help="if the surface is corrupted, create a backup of the original surface. "
             "Default: no backup.",
        default=None,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__version__} 2024/08/08 12:20:10 kueglerd",
    )
    return parser


def resafe_surface(
        surface_file: Path | str,
        surface_backup: Path | str | None = None,
) -> bool:
    """
    Take path to surface_file and rewrite it to fix improperly oriented triangles.

    If the surface is not oriented and surface_backup is set, rename the old
    surface_file to surface_backup. Else just overwrite with the corrected surface.

    Parameters
    ----------
    surface_file : Path, str
        Path and name of input surface.
    surface_backup : Path, str, optional
        Path and name of output surface.

    Returns
    -------
    bool
        Whether the surface was rewritten.
    """
    import getpass
    try:
        getpass.getuser()
    except Exception:
        # nibabel crashes in write_geometry, if getpass.getuser does not return
        # make sure the process has a username
        from os import environ
        environ.setdefault("USERNAME", "UNKNOWN")

    triamesh = lapy.TriaMesh.read_fssurf(str(surface_file))
    fsinfo = triamesh.fsinfo

    # make sure the triangles are oriented (normals pointing to the same direction
    if not triamesh.is_oriented():
        if surface_backup is not None:
            print(f"Renaming {surface_file} to {surface_backup}")
            shutil.move(surface_file, surface_backup)

        print("Surface was not oriented, flipping corrupted normals.")
        triamesh.orient_()

        from packaging.version import Version
        if Version(lapy.__version__) <= Version("1.0.1"):
            print(f"lapy version {lapy.__version__}<=1.0.1 detected, fixing fsinfo.")
            triamesh.fsinfo = fsinfo

        triamesh.write_fssurf(str(surface_file))
        return True
    else:
        print("Surface was already oriented.")
        return False


if __name__ == "__main__":
    # Command Line options are error checking done here
    parser = make_parser()
    args = parser.parse_args()

    print(f"Reading in surface: {args.file} ...")
    if resafe_surface(args.file, args.backup):
        print(f"Outputting surface as: {args.file}")
    sys.exit(0)
