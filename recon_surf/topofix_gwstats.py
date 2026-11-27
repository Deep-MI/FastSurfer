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
Replace the integer gray and white modes in ``surf/autodet.gw.stats.<hemi>.dat`` by smooth modes.

recon-surf.sh runs this with ``--topofix`` after mris_autodet_gwstats, see
:mod:`recon_surf.topofix.gwstats`. The original file is kept as
``surf/autodet.gw.stats.<hemi>.integer.dat``. If the file cannot be refined, it is left as it is.
"""

import argparse
import shutil
import sys
from pathlib import Path


def options() -> argparse.Namespace:
    """
    Parse the command line.

    Returns
    -------
    argparse.Namespace
        The hemisphere, subjects directory and subject.
    """
    from os import environ

    parser = argparse.ArgumentParser(description="Smooth gray and white modes for surface placement")
    parser.add_argument("--hemi", choices=("lh", "rh"), help="Hemisphere to process.", required=True)
    parser.add_argument(
        "--sd",
        type=Path,
        help="Subjects directory $SUBJECTS_DIR.",
        default=Path(environ.get("SUBJECTS_DIR", Path.cwd())),
        required="SUBJECTS_DIR" not in environ,
    )
    parser.add_argument("--subject", type=str, help="Name (ID) of subject.", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    opts = options()
    from recon_surf.topofix.gwstats import read_stats, refine_stats, sample_white_gray, write_stats

    subject_dir = opts.sd / opts.subject
    stats_file = subject_dir / "surf" / f"autodet.gw.stats.{opts.hemi}.dat"
    stats = read_stats(stats_file)
    # mris_autodet_gwstats writes integer modes, so other modes come from an earlier run of this script
    if any(stats[k] != round(stats[k]) for k in ("white_mode", "gray_mode")):
        print(f"{opts.hemi}: {stats_file.name} is already refined")
        sys.exit(0)
    white, gray = sample_white_gray(
        subject_dir / "mri" / "brain.finalsurfs.mgz",
        subject_dir / "mri" / "wm.mgz",
        subject_dir / "surf" / f"{opts.hemi}.orig.premesh",
    )
    try:
        refined = refine_stats(stats, white, gray)
    except ValueError as e:
        print(f"{opts.hemi}: keeping {stats_file.name}: {e}")
        sys.exit(0)
    shutil.copyfile(stats_file, stats_file.with_suffix(".integer.dat"))
    write_stats(refined, stats_file)
    print(
        f"{opts.hemi}: white mode {stats['white_mode']:.0f} -> {refined['white_mode']:.2f}, "
        f"gray mode {stats['gray_mode']:.0f} -> {refined['gray_mode']:.2f}"
    )
