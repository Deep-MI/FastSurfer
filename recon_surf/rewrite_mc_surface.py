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
from lapy.read_geometry import read_geometry


def options_parse():
    """
    Command line option parser for spherically_project.py
    """
    parser = optparse.OptionParser(version='$Id: rewrite_mc_surface,v 1.1 2020/06/23 15:42:08 henschell $',
                                   usage='Function to load and resafe surface under a given name')
    parser.add_option('--input', '-i', dest='input_surf', help='path to input surface')
    parser.add_option('--output', '-o', dest='output_surf', help='path to ouput surface')
    parser.add_option('--filename_pretess', '-p', dest='in_pretess', default=None,
                      help='path and name of pretess file used (info missing when using marching cube).')
    (options, args) = parser.parse_args()

    if options.input_surf is None or options.output_surf is None:
        sys.exit('ERROR: Please specify input and output surfaces')

    return options


def resafe_surface(insurf, outsurf, pretess):
    """
    takes path to insurf and rewrites it to outsurf thereby fixing vertex locs flag error
    (scannerRAS instead of surfaceRAS after marching cube)
    :param str insurf: path and name of input surface
    :param str outsurf: path and name of output surface
    :param str pretess: path and name of file the input surface was created on (e.g. filled-pretess127.mgz)
    """
    surf = read_geometry(insurf, read_metadata=True)

    if not surf[2]['filename']:
        # Set information with file used for surface construction (volume info and name)
        surf[2]['filename'] = pretess
        surf[2]['volume'] = nibload(pretess).header.get_data_shape()

    fs.write_geometry(outsurf, surf[0], surf[1], volume_info=surf[2])


if __name__ == "__main__":
    # Command Line options are error checking done here
    options = options_parse()
    surf_in = options.input_surf
    surf_out = options.output_surf
    vol_in = options.in_pretess

    print("Reading in surface: {} ...".format(surf_in))
    resafe_surface(surf_in, surf_out, vol_in)
    print ("Outputing surface as: {}".format(surf_out))
    sys.exit(0)
