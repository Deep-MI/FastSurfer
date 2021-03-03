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
from subprocess import Popen, PIPE
import shlex
import argparse


def setup_options():
    # Validation settings
    parser = argparse.ArgumentParser(description='Wrapper for spherical projection')

    parser.add_argument('--hemi', type=str, help="Hemisphere to analyze.")
    parser.add_argument('--sdir', type=str, help="Surface directory of subject.")
    parser.add_argument('--subject', type=str, help="Name (ID) of subject.")
    parser.add_argument('--threads', type=int, help="Number of threads to use.")
    parser.add_argument('--py', type=str, help="which python version to use.")
    parser.add_argument('--binpath', type=str, help="directory of spherically_project.py script.")

    args = parser.parse_args()
    return args


def call(command, **kwargs):
    """Run command with arguments. Wait for command to complete. Sends
    output to logging module. The arguments are the same as for the Popen
    constructor."""

    kwargs['stdout'] = PIPE
    kwargs['stderr'] = PIPE
    command_split = shlex.split(command)

    p = Popen(command_split, **kwargs)
    stdout = p.communicate()[0]

    if stdout:
        for line in stdout.decode('utf-8').split("\n"):
            print(line)

    return p.returncode


def spherical_wrapper(command1, command2, **kwargs):
    # First try to run standard spherical project
    print("Running command: {}".format(command1))
    code_1 = call(command1, **kwargs)

    if code_1 != 0:
        print("Command {} failed.\nRunning fallback command: {}".format(command1, command2))
        code_1 = call(command2, **kwargs)

    return code_1


if __name__ == "__main__":

    opts = setup_options()
    cmd1 = opts.py + " " + opts.binpath + "spherically_project.py -i " + opts.sdir + "/" + opts.hemi \
                   + ".smoothwm.nofix -o " + opts.sdir + "/" + opts.hemi + ".qsphere.nofix"

    if opts.threads > 1:
        threading = "-threads " + str(opts.threads) + " -itkthreads " + str(opts.threads)
    else:
        threading = ""

    cmd2 = "recon-all -s " + opts.subject + " -hemi " + opts.hemi + " -qsphere -no-isrunning " + threading
    spherical_wrapper(cmd1, cmd2)
