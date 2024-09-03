# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
import optparse
import nibabel.freesurfer.io as fs
from nibabel import load as nibload


def options_parse():
    """
    Create a command line interface and return command line options.

    Returns
    -------
    options
        Namespace object holding options.
    """
    parser = optparse.OptionParser(
        version="$Id: rewrite_mc_surface,v 1.1 2020/06/23 15:42:08 henschell $",
        usage="Function to load and resafe surface under a given name",
    )
    parser.add_option("--input", "-i", dest="input_surf", help="path to input surface")
    parser.add_option(
        "--output", "-o", dest="output_surf", help="path to output surface"
    )
    parser.add_option(
        "--filename_pretess",
        "-p",
        dest="in_pretess",
        default=None,
        help="path and name of pretess file used (info missing when using marching cube).",
    )
    (options, args) = parser.parse_args()

    if options.input_surf is None or options.output_surf is None:
        sys.exit("ERROR: Please specify input and output surfaces")

    return options


def resafe_surface(insurf: str, outsurf: str, pretess: str) -> None:
    """
    Take path to insurf and rewrite it to outsurf thereby fixing vertex locs flag error.

    This function fixes header information not properly saved in marching cubes. 
    It makes sure the file header correctly references the scannerRAS instead of the surfaceRAS,
    i.e. filename and volume is set to the correct data in the header.

    Parameters
    ----------
    insurf : str
        Path and name of input surface.
    outsurf : str
        Path and name of output surface.
    pretess : str
        Path and name of file the input surface was created on (e.g. filled-pretess127.mgz).
    """
    surf = fs.read_geometry(insurf, read_metadata=True)

    if not surf[2]["filename"]:
        # Set information with file used for surface construction (volume info and name)
        surf[2]["filename"] = pretess
        surf[2]["volume"] = nibload(pretess).header.get_data_shape()

    fs.write_geometry(outsurf, surf[0], surf[1], volume_info=surf[2])


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()
    surf_in = options.input_surf
    surf_out = options.output_surf
    vol_in = options.in_pretess

    print("Reading in surface: {} ...".format(surf_in))
    resafe_surface(surf_in, surf_out, vol_in)
    print("Outputting surface as: {}".format(surf_out))
    sys.exit(0)
