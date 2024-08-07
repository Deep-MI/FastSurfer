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

# IMPORTS
import sys
import argparse
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
        "--input", "-i",
        type=Path,
        dest="input_surf",
        help="path to input surface",
        required=True,
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        dest="output_surf",
        help="path to output surface",
        required=True,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__version__} 2024/08/08 12:20:10 kueglerd",
    )
    return parser


def resafe_surface(insurf: Path | str, outsurf: Path | str) -> None:
    """
    Take path to insurf and rewrite it to outsurf thereby fixing improperly oriented triangles.

    Parameters
    ----------
    insurf : Path, str
        Path and name of input surface.
    outsurf : Path, str
        Path and name of output surface.
    """
    import getpass
    try:
        getpass.getuser()
    except Exception:
        # nibabel crashes in write_geometry, if getpass.getuser does not return
        # make sure the process has a username
        from os import environ
        environ.setdefault("USERNAME", "UNKNOWN")

    triamesh = lapy.TriaMesh.read_fssurf(str(insurf))

    # make sure the triangles are oriented (normals pointing to the same direction
    if not triamesh.is_oriented():
        print("Surface was not oriented, flipping corrupted normals.")
        triamesh.orient_()
    else:
        print("Surface was oriented.")

    triamesh.write_fssurf(str(outsurf))


if __name__ == "__main__":
    # Command Line options are error checking done here
    parser = make_parser()
    args = parser.parse_args()
    surf_in = args.input_surf
    surf_out = args.output_surf

    print(f"Reading in surface: {surf_in} ...")
    resafe_surface(surf_in, surf_out)
    print(f"Outputting surface as: {surf_out}")
    sys.exit(0)
