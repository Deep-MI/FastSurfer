#!/bin/python

# Copyright 2024 Image Analysis Lab, German Center for Neurodegenerative Diseases
# (DZNE), Bonn
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
from os import environ as env
from pathlib import Path

from FastSurferCNN.segstats import HelpFormatter, main, VERSION
from FastSurferCNN.mri_segstats import print_and_exit

DEFAULT_MEASURES_STRINGS = [
   (False, "BrainSeg"),
   (False, "BrainSegNotVent"),
   (False, "SupraTentorial"),
   (False, "SupraTentorialNotVent"),
   (False, "SubCortGray"),
   (False, "lhCortex"),
   (False, "rhCortex"),
   (False, "Cortex"),
   (False, "TotalGray"),
   (False, "lhCerebralWhiteMatter"),
   (False, "rhCerebralWhiteMatter"),
   (False, "CerebralWhiteMatter"),
   (False, "Mask"),
   (False, "SupraTentorialNotVentVox"),
   (False, "BrainSegNotVentSurf"),
   (False, "VentricleChoroidVol"),
]
DEFAULT_MEASURES = list((False, m) for m in DEFAULT_MEASURES_STRINGS)

USAGE = "python mri_brainvol_stats.py -s <subject>"
HELPTEXT = f"""
Dependencies:

    Python 3.10

    Numpy
    http://www.numpy.org

    Nibabel to read images
    http://nipy.org/nibabel/

    Pandas to read/write stats files etc.
    https://pandas.pydata.org/

Original Author: David Kügler
Date: Jan-23-2024

Revision: {VERSION}
"""
DESCRIPTION = """
Translates mri_brainvol_stats options for segstats.py. Options not listed here have no 
equivalent representation in segstats.py. """


def make_arguments() -> argparse.ArgumentParser:
    """Make the argument parser."""
    parser = argparse.ArgumentParser(
        usage=USAGE,
        epilog=HELPTEXT.replace("\n", "<br>"),
        description=DESCRIPTION,
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--print",
        action="append_const",
        dest="parse_actions",
        default=[],
        const=print_and_exit,
        help="Print the equivalent native segstats.py options and exit.",
    )
    default_sd = Path(env["SUBJECTS_DIR"]) if "SUBJECTS_DIR" in env else None
    parser.add_argument(
        "--sd",
        dest="out_dir", metavar="subjects_dir", type=Path,
        default=default_sd,
        required=not bool(default_sd),
        help="set SUBJECTS_DIR, defaults to environment SUBJECTS_DIR, required to find "
             "several files used by measures, e.g. surfaces.")
    parser.add_argument(
        "-s",
        "--subject",
        "--sid",
        dest="sid", metavar="subject_id",
        help="set subject_id, required to find several files used by measures, e.g. "
             "surfaces.")
    parser.add_argument(
        "-o",
        "--segstatsfile",
        dest="segstatsfile",
        default=Path("stats/brainvol.stats"),
        help="Where to save the brainvol.stats, if relative path, this will be "
             "relative to the subject directory."
    )
    fs_home = "FREESURFER_HOME"
    default_lut = Path(env[fs_home]) / "ASegStatsLUT.txt" if fs_home in env else None
    parser.set_defaults(
        segfile=Path("mri/aseg.mgz"),
        measures=DEFAULT_MEASURES,
        lut=default_lut,
        measure_only=True,
    )
    advanced = parser.add_argument_group(
        "FastSurfer options (no equivalence with FreeSurfer's mri_brainvol_stats)",
    )
    advanced.add_argument(
        "--no_legacy",
        action="store_false",
        dest="legacy_freesurfer",
        help="use FastSurfer algorithms instead of FastSurfer.",
    )
    advanced.add_argument(
        "--pvfile",
        "-pv",
        type=Path,
        dest="pvfile",
        help="Path to image used to compute the partial volume effects. This file is "
             "only used in the FastSurfer algoritms (--no_legacy).",
    )
    return parser


if __name__ == "__main__":
    import sys

    args = make_arguments().parse_args()
    parse_actions = getattr(args, "parse_actions", [])
    for parse_action in parse_actions:
        parse_action(args)
    sys.exit(main(args))
