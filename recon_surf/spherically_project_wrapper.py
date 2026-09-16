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
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Accepted for interface compatibility, but ignored: both the projection and its "
             "FreeSurfer fallback are pinned to one thread so the result is reproducible.",
    )

    args = parser.parse_args()
    return args


def enabled_simd_features(python: str) -> list[str]:
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
    list[str]
        The dispatchable features this numpy has enabled, empty if they cannot be read.
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
        return []
    return out.stdout.split()


if __name__ == "__main__":
    import sys
    from os import environ
    opts = setup_options()

    # The spectral projection is sensitive to tiny threaded BLAS/eigensolver
    # differences, and those can be amplified by later topology correction.
    # ITK is in the list for the fallback rather than for the projection: recon-surf.sh exports
    # ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS per hemisphere, and the child would inherit that.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ):
        environ[var] = "1"

    # Thread counts are not the only thing that decides the arithmetic here. numpy and OpenBLAS
    # both choose kernels from the CPU features they detect at import, so the same wheel computes
    # slightly different numbers on different machines. Measured across eight runners and four CPU
    # models, that moves the projected sphere by about 1e-5, and the topology correction turns that
    # into a different retessellation and a different vertex count, which every later surface and
    # every number derived from one inherits.
    #
    # Pinning both to a level every x86-64 machine can reach collapsed those runners to one result.
    # It costs about 20% of this step, seconds against a pipeline measured in tens of minutes.
    # Set before numpy is imported below, because both are read once at import.
    environ["OPENBLAS_CORETYPE"] = "Nehalem"
    simd = enabled_simd_features(sys.executable)
    if simd:
        environ["NPY_DISABLE_CPU_FEATURES"] = " ".join(simd)
    else:
        print("WARNING: could not read numpy's SIMD features, so this projection is not pinned")
        print("  and may not reproduce on other hardware.")

    # identify whether sksparse is installed (in which case we can use_cholmod in LaPy
    try:
        # ignore ruff F401 (unused import)
        from sksparse import cholmod  # noqa F401
        has_sksparse = True
    except ImportError:
        has_sksparse = False
        # First try to run standard spherical project
    try:
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
        if recon_all is None:
            # without this the tuple below starts with None and Popen raises TypeError, which would
            # replace the projection error above with something unrelated
            print(
                "spherical_project.py failed, and recon-all is not on PATH, so the FreeSurfer "
                "fallback cannot run either. Is FREESURFER_HOME set and sourced?",
                file=sys.stderr,
            )
            sys.exit(1)

        # -hemi, not " -hemi ": Popen takes a list, so nothing splits the argument. It happens to
        # work today only because recon-all is tcsh and `switch ($flag)` word-splits the unquoted
        # variable; any other callee would reject it.
        # No -threads either: this process pinned the thread count to 1 above for reproducibility,
        # and passing -threads would call omp_set_num_threads in the child and undo that. The
        # pinned variables reach it through env below.
        static_args = ("-qsphere", "-no-isrunning", "-umask", f"{_umask:o}")
        fallback = (recon_all, "-s", opts.subject, "-hemi", opts.hemi) + static_args

        print(f"spherical_project.py failed.\nRunning fallback command: {' '.join(fallback)}")
        process = Popen(fallback, env=dict(environ, SUBJECTS_DIR=str(opts.sd)))
        done = process.forward_output(encoding="utf-8", timeout=None)
        sys.exit(done.retcode)
