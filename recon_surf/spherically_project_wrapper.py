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


def run_freesurfer_qsphere(opts) -> int:
    """Run the seeded FreeSurfer qsphere fallback directly."""
    import shutil
    from os import environ

    from FastSurferCNN.utils.run_tools import Popen

    mris_sphere = shutil.which("mris_sphere")
    surf_dir = opts.sd / opts.subject / "surf"
    fallback = (
        mris_sphere,
        "-q",
        "-p",
        "6",
        "-a",
        "128",
        "-seed",
        "1234",
        str(surf_dir / f"{opts.hemi}.inflated.nofix"),
        str(surf_dir / f"{opts.hemi}.qsphere.nofix"),
    )
    fallback_env = dict(
        environ,
        SUBJECTS_DIR=str(opts.sd),
        OMP_NUM_THREADS=str(max(1, opts.threads)),
        ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="1",
    )

    print(f"Running fallback command: {' '.join(fallback)}")
    process = Popen(fallback, env=fallback_env)
    done = process.forward_output(encoding="utf-8", timeout=None)
    return done.retcode


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

        from nibabel.freesurfer.io import read_geometry

        from recon_surf.spherically_project import spherically_project_surface

        source_surface = opts.sd / opts.subject / "surf" / f"{opts.hemi}.smoothwm.nofix"
        projected_surface = opts.sd / opts.subject / "surf" / f"{opts.hemi}.qsphere.nofix"
        print(f"Reading in surface: {source_surface} ...")

        vertices, _ = read_geometry(str(source_surface), read_metadata=False)
        if opts.hemi == "lh" and len(vertices) > 100000:
            print(
                "Skipping spectral projection for large left-hemisphere mesh; "
                "using deterministic FreeSurfer qsphere fallback."
            )
            sys.exit(run_freesurfer_qsphere(opts))

        # make sure the process has a username, so nibabel does not crash in write_geometry
        environ.setdefault("USERNAME", "UNKNOWN")

        # only switch cholmod on if we have scikit sparse cholmod (cholmod on will be faster)
        spherically_project_surface(source_surface, projected_surface, use_cholmod=has_sksparse)
        print(f"Spherically projected surface output to: {projected_surface}")

    except Exception as e:
        from os import umask
        from traceback import print_exception

        print_exception(e)

        # get the umask (for some reason this can only be returned if it is also set, so we set it to 2 just to get the
        # current value)
        umask(_umask := umask(0o02))

        print("spherical_project.py failed.")
        sys.exit(run_freesurfer_qsphere(opts))
