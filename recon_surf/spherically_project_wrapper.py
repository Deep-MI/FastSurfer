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
import argparse
from pathlib import Path


def setup_options():
    """
    Create a command line interface and return command line options.

    Returns
    -------
    options : argparse.Namespace
        Namespace object holding options.
    """
    from os import environ
    # Validation settings
    parser = argparse.ArgumentParser(description="Wrapper for spherical projection")

    parser.add_argument("--hemi", choices=("lh", "rh"), help="Hemisphere to analyze.", required=True)
    parser.add_argument(
        "--sd",
        type=Path,
        help="Subjects directory $SUBJECTS_DIR.",
        default=Path(environ.get("SUBJECTS_DIR", Path.cwd())),
        required="SUBJECTS_DIR" not in environ,
    )
    parser.add_argument("--subject", type=str, help="Name (ID) of subject.", required=True)
    parser.add_argument("--threads", type=int, help="Number of threads to use.", default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import sys
    opts = setup_options()

    # identify whether sksparse is installed (in which case we can use_cholmod in LaPy
    try:
        # ignore ruff F401 (unused import)
        from sksparse import cholmod  # noqa F401
        has_sksparse = True
    except ImportError:
        has_sksparse = False
        # First try to run standard spherical project
    try:
        from os import environ

        from recon_surf.spherically_project import spherically_project_surface

        source_surface = opts.sd / opts.subject / "surf" / f"{opts.hemi}.smoothwm.nofix"
        projected_surface = opts.sd / opts.subject / "surf" / f"{opts.hemi}.qsphere.nofix"
        print(f"Reading in surface: {source_surface} ...")

        # make sure the process has a username, so nibabel does not crash in write_geometry
        environ.setdefault("USERNAME", "UNKNOWN")

        # only switch cholmod on if we have scikit sparse cholmod (cholmod on will be faster)
        spherically_project_surface(source_surface, projected_surface, use_cholmod=has_sksparse)
        print(f"Spherically projected surface output to: {projected_surface}")

    except Exception as e:
        import shutil
        from os import umask
        from traceback import print_exception

        from FastSurferCNN.utils.run_tools import Popen

        print_exception(e)

        # get the umask (for some reason this can only be returned if it is also set, so we set it to 2 just to get the
        # current value)
        umask(_umask := umask(0o02))

        # run the FreeSurfer fallback command
        recon_all = shutil.which("recon-all")
        static_args = ("-qsphere", "-no-isrunning", "-umask", f"{_umask:o}")
        fallback = (recon_all, "-s", opts.subject, " -hemi ", opts.hemi) + static_args
        if opts.threads > 1:
            fallback += ("-threads", str(opts.threads), "-itkthreads", str(opts.threads))

        print(f"spherical_project.py failed.\nRunning fallback command: {' '.join(fallback)}")
        process = Popen(fallback, env=dict(environ, SUBJECTS_DIR=str(opts.sd)))
        done = process.forward_output(encoding="utf-8", timeout=None)
        sys.exit(done.retcode)
