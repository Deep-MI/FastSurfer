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
from typing import Literal

from FastSurferCNN.segstats import HelpFormatter, VERSION, _check_arg_path
from FastSurferCNN.mri_segstats import print_and_exit
from FastSurferCNN.utils.threads import get_num_threads

DEFAULT_MEASURES = [
    "BrainSeg",
    "BrainSegNotVent",
    "SupraTentorial",
    "SupraTentorialNotVent",
    "SubCortGray",
    "lhCortex",
    "rhCortex",
    "Cortex",
    "TotalGray",
    "lhCerebralWhiteMatter",
    "rhCerebralWhiteMatter",
    "CerebralWhiteMatter",
    "Mask",
    "SupraTentorialNotVentVox",
    "BrainSegNotVentSurf",
    "VentricleChoroidVol",
]

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
        computed_measures=DEFAULT_MEASURES,
        lut=default_lut,
    )
    parser.add_argument(
        "--no_legacy",
        action="store_false",
        dest="legacy_freesurfer",
        help="use fastsurfer algorithms instead of fastsurfer."
    )
    return parser


def main(args: argparse.Namespace) -> Literal[0] | str:
    """
    Main segstats function, based on mri_segstats.

    Parameters
    ----------
    args : object
        Parameter object as defined by `make_arguments().parse_args()`

    Returns
    -------
    Literal[0], str
        Either as a successful return code or a string with an error message
    """
    from time import perf_counter_ns
    from concurrent.futures import ThreadPoolExecutor

    from FastSurferCNN.utils.common import assert_no_root
    from FastSurferCNN.utils.brainvolstats import Manager, PVMeasure

    start = perf_counter_ns()
    getattr(args, "allow_root", False) or assert_no_root()

    subjects_dir = getattr(args, "out_dir", None)
    subject_id = getattr(args, "sid", None)

    # the manager object calculates the measure
    manager = Manager(args)
    # load these files in different threads to avoid waiting on IO
    # (not parallel due to GIL though)
    with manager.with_subject(subjects_dir, subject_id):
        segstatsfile = _check_arg_path(
            args,
            "segstatsfile",
            subjects_dir=subjects_dir,
            subject_id=subject_id,
            require_exist=False,
        )
        # lutfile = _check_arg_path(
        #     args,
        #     "lut",
        #     subjects_dir=subjects_dir,
        #     subject_id=subject_id,
        #     require_exist=False,
        # )
        # lut = manager.make_read_hook(read_classes_from_lut)(lutfile)

        threads = getattr(args, "threads", 0)
        if threads <= 0:
            threads = get_num_threads()
        compute_threads = ThreadPoolExecutor(threads)

        if any(isinstance(m, PVMeasure) for m in manager.values()):
            return "mri_brainvol_stats does not support PVMeasures."
        manager.default_measures = DEFAULT_MEASURES

    # finished manager io here
    # manager.lut = lut
    manager.compute_non_derived_pv(compute_threads)
    # wait for computation of measures and return an error message if errors occur
    try:
        manager.wait_write_brainvolstats(segstatsfile)
    except RuntimeError as e:
        return e.args[0]
    print(f"Brain volume stats written to {segstatsfile}.")
    duration = (perf_counter_ns() - start) / 1e9
    print(f"Calculation took {duration:.2f} seconds using up to {threads} threads.")
    return 0


if __name__ == "__main__":
    import sys

    args = make_arguments().parse_args()
    parse_actions = getattr(args, "parse_actions", [])
    for parse_action in parse_actions:
        parse_action(args)
    sys.exit(main(args))