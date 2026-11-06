# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
Create the topologically correct ``?h.orig`` of one hemisphere from the segmentation.

This replaces the orig.nofix, qsphere and fix steps of recon-surf.sh with ``--topofix``. It
writes ``surf/<hemi>.orig.nofix``, ``surf/<hemi>.orig.premesh`` and ``surf/<hemi>.orig``, and a QC
record in ``scripts/<hemi>.topofix.json``. If it fails, the record says why and the exit status
is 1, so that recon-surf.sh runs the standard steps instead.
"""

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

    parser = argparse.ArgumentParser(description="Topologically correct white matter surface from the segmentation")
    parser.add_argument("--hemi", choices=("lh", "rh"), help="Hemisphere to process.", required=True)
    parser.add_argument(
        "--sd",
        type=Path,
        help="Subjects directory $SUBJECTS_DIR.",
        default=Path(environ.get("SUBJECTS_DIR", Path.cwd())),
        required="SUBJECTS_DIR" not in environ,
    )
    parser.add_argument("--subject", type=str, help="Name (ID) of subject.", required=True)
    parser.add_argument(
        "--seg",
        type=Path,
        help="Segmentation; default: <subject_dir>/mri/aparc.DKTatlas+aseg.deep.mgz.",
    )
    parser.add_argument(
        "--hires",
        action="store_true",
        help="Remesh to the coarser density of high-resolution surfaces on the standard route.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import json
    import sys
    from os import environ
    from traceback import format_exc

    opts = setup_options()

    # One thread, and the CPU dispatch pinned, as in spherically_project_wrapper.py: the sphere
    # maps that find folds feed discrete decisions, so tiny arithmetic differences between machines
    # could change which surface is removed.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
    ):
        environ[var] = "1"
    # imported here, after the pins and before numpy, see spherically_project_wrapper.py
    from pin_cpu_dispatch import pin_cpu_dispatch

    pin_cpu_dispatch(environ, sys.executable)

    subject_dir = opts.sd / opts.subject
    seg = opts.seg or subject_dir / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    record = subject_dir / "scripts" / f"{opts.hemi}.topofix.json"
    try:
        from recon_surf.topofix.surfaces import TARGET_LENGTH, TARGET_LENGTH_HIRES, create_orig_surfaces

        target = TARGET_LENGTH_HIRES if opts.hires else TARGET_LENGTH
        qc = create_orig_surfaces(seg, opts.hemi, subject_dir / "surf", target_length=target)
        qc["status"] = "ok"
        status = 0
    except Exception:
        print(format_exc())
        qc = {"hemi": opts.hemi, "segmentation": str(seg), "status": "failed", "error": format_exc()}
        status = 1
    record.write_text(json.dumps(qc, indent=2) + "\n")
    if status == 0:
        n = len(qc["handles"])
        print(f"{opts.hemi}: genus {qc['genus']}, removed {n} handles, wrote surf/{opts.hemi}.orig and {record}")
    else:
        print(f"{opts.hemi}: topofix failed, see {record}")
    sys.exit(status)
