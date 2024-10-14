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
from itertools import pairwise, chain
from pathlib import Path
from typing import TypeVar, Sequence, Any, Iterable

from FastSurferCNN.segstats import (
    main,
    HelpFormatter,
    add_two_help_messages,
    VERSION,
    empty,
)

_T = TypeVar("_T")


USAGE = "python mri_segstats.py --seg segvol [optional arguments]"
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
Date: Jan-04-2024

Revision: {VERSION}
"""
DESCRIPTION = """
Translates mri_segstats options for segstats.py. Options not listed here have no 
equivalent representation in segstats.py. <br>
IMPORTANT NOTES
mri_segstats uses a legacy version for the computation of measures (from FreeSurfer 6).
But mri_segstats.py implements the behavior if first mri_brainvol_stats
and then mri_segstats is run (which uses the stats/brainvol.stats generated by 
mri_brainvol_stats). This reflects the output of stats files as created by FreeSurfer's 
recon-all.
"""
ETIV_RATIO_KEY = "eTIV-ratios"
ETIV_RATIOS = {"BrainSegVol-to-eTIV": "BrainSeg", "MaskVol-to-eTIV": "Mask"}
ETIV_FROM_TAL = "EstimatedTotalIntraCranialVol"


class _ExtendConstAction(argparse.Action):
    """Helper class to allow action='extend_const' by action=_ExtendConstAction."""
    def __init__(
            self,
            option_strings: Sequence[str],
            dest: str,
            const: _T | None = None,
            default: _T | str | None = None,
            required: bool = False,
            help: str | None = None,
            metavar: str | tuple[str, ...] | None = None,
    ) -> None:
        super(_ExtendConstAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=const,
            default=default,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: str | Sequence[Any],
            option_string: str | None = None,
    ) -> None:
        """
        Extend attribute `self.dest` of `namespace` with the values in `self.const`.
        """
        items = getattr(namespace, self.dest, None)
        if items is None:
            items = []
        elif type(items) is list:
            items = items[:]
        else:
            import copy
            items = copy.copy(items)
        items.extend(self.const)
        setattr(namespace, self.dest, items)


def make_arguments() -> argparse.ArgumentParser:
    """Create an argument parser object with all parameters of the script."""
    parser = argparse.ArgumentParser(
        usage=USAGE,
        epilog=HELPTEXT.replace("\n", "<br>"),
        description=DESCRIPTION,
        formatter_class=HelpFormatter,
        add_help=False,
    )

    def add_etiv_measures(args: argparse.Namespace) -> None:
        measures: list[tuple[bool, str]] = getattr(args, "measures", [])
        measure_strings = list(map(lambda x: x[1], measures))
        if all(m in measure_strings for m in (ETIV_RATIO_KEY, ETIV_FROM_TAL)):

            measures = [m for m in measures if m[1] == ETIV_RATIO_KEY]
            for k, v in ETIV_RATIOS.items():
                for is_imported, m in measures:
                    if m == v or m.startswith(v + "("):
                        measures.append((False, k))
                        continue
            setattr(args, "measures", measures)

    def _update_what_to_import(args: argparse.Namespace) -> argparse.Namespace:
        """
        Update the Namespace object based on the existence of the brainvol.stats file.
        """
        cachefile = Path(args.measurefile)
        if not cachefile.is_absolute():
            cachefile = args.out_dir / args.sid / cachefile
        if not args.explicit_no_cached and cachefile.is_file():
            from FastSurferCNN.utils.brainvolstats import read_measure_file

            measure_data = read_measure_file(cachefile)

            def update_key(measure: tuple[bool, str]) -> tuple[bool, str]:
                is_imported, measure_string = measure
                measure_key, _, _ = measure_string.partition("(")
                return measure_key in measure_data.keys(), measure_string

            # for each measure to be computed, replace it with the value in
            # brainvol.stats, if available
            args.measures = list(map(update_key, getattr(args, "measures", [])))
        return args

    parser.set_defaults(
        measurefile="stats/brainvol.stats",
        parse_actions=[(1, add_etiv_measures), (10, _update_what_to_import)],
        volume_precision=1,
    )

    if "--help" in sys.argv:
        from FastSurferCNN.utils.brainvolstats import Manager
        manager = Manager([])

        def help_text(keys: Iterable[str]) -> Iterable[str]:
            return (manager[k].help() for k in keys)
    else:
        help_text = None

    def help_add_measures(message: str, keys: list[str]) -> str:
        if help_text:
            _keys = (k.split(' ')[0] for k in keys)
            keys = [f"{k}: {text}" for k, text in zip(keys, help_text(_keys))]
        return "<br>- ".join([message] + list(keys))

    add_two_help_messages(parser)
    parser.add_argument(
        "--print",
        action="append_const",
        dest="parse_actions",
        const=(0, print_and_exit),
        help="Print the equivalent native segstats.py options and exit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
        help="Print the version of the mri_segstats.py script",
    )
    parser.add_argument(
        "--seg",
        type=Path,
        metavar="segvol",
        dest="segfile",
        help="Specify the segmentation file.",
    )
    # --annot subject hemi parc
    # --surf whitesurfname -- used with annot
    # --slabel subject hemi full_path_to_label
    # --label-thresh threshold
    # --seg-from-input
    parser.add_argument(
        "--o",
        "--sum",
        type=Path,
        metavar="file",
        dest="segstatsfile",
        help="Specifiy the output summary statistics file.",
    )
    parser.add_argument(
        "--pv",
        type=Path,
        metavar="pvvol",
        dest="pvfile",
        help="file to compensate for partial volume effects.",
    )
    parser.add_argument(
        "--i",
        "--in",
        type=Path,
        metavar="invol",
        dest="normfile",
        help="file to compute intensity values.",
    )

    # --seg-erode Nerodes
    # --frame frame
    def _percent(__value) -> float:
        return float(__value) / 50

    parser.add_argument(
        "--robust",
        type=_percent,
        metavar="percent",
        dest="robust",
        help="Compute stats after excluding percent from high and and low values, e.g. "
             "with --robust 2, min and max are the 2nd and the 98th percentiles.",
    )

    def _add_invol_op(*flags: str, op: str, metavar: str | None = None) -> None:
        if metavar:
            def _optype(_a) -> str:
                # test the argtype for float as well
                return f"{flags[0].lstrip('-')}={float(_a)}"
            kwargs = {
                "action": "append",
                "type": _optype,
                "dest": "pvfile_preproc",
                "help": f"Apply the {op} with `{metavar}` to `invol` (--in)",
            }
        else:
            kwargs = {
                "action": "append_const",
                "const": flags[0].lstrip("-"),
                "dest": "pvfile_preproc",
                "help": f"Apply {op} to `invol` (--in)",
            }
        parser.add_argument(*flags, **kwargs)

    def _no_import(*args: str) -> list[tuple[bool, str]]:
        return list((False, a) for a in args)

    _add_invol_op("--sqr", op="squaring")
    _add_invol_op("--sqrt", op="the square root")
    _add_invol_op("--mul", op="multiplication", metavar="val")
    _add_invol_op("--div", op="division", metavar="val")
    # --snr
    _add_invol_op("--abs", op="absolute value")
    # --accumulate
    parser.add_argument(
        "--ctab",
        type=Path,
        metavar="ctabfile",
        dest="lut",
        help="load the Color Lookup Table.",
    )
    import os
    env = os.environ
    if "FREESURFER_HOME" in env:
        default_lut = Path(env["FREESURFER_HOME"]) / "FreeSurferColorLUT.txt"
    elif "FASTSURFER_HOME" in env:
        default_lut = (
            Path(env["FASTSURFER_HOME"]) / "FastSurferCNN/config/FreeSurferColorLUT.txt"
        )
    else:
        default_lut = None
    parser.add_argument(
        "--ctab-default",
        metavar="ctabfile",
        dest="lut",
        const=default_lut,
        action="store_const",
        help="load default Color Lookup Table (from FREESURFER_HOME or "
             "FASTSURFER_HOME).",
    )
    # --ctab-gca gcafile
    parser.add_argument(
        "--id",
        type=int,
        nargs="+",
        metavar="segid",
        action="extend",
        dest="ids",
        default=[],
        help="Specify segmentation Exclude segmentation ids from report.",
    )
    parser.add_argument(
        "--excludeid",
        type=int,
        nargs="+",
        metavar="segid",
        dest="excludeid",
        help="Exclude segmentation ids from report.",
    )
    parser.add_argument(
        "--no-cached",
        action="store_true",
        dest="explicit_no_cached",
        help="Do not try to load stats/brainvol.stats.",
    )
    parser.add_argument(
        "--excl-ctxgmwm",
        dest="excludeid",
        action=_ExtendConstAction,
        const=[2, 3, 41, 42],
        help="Exclude cortical gray and white matter regions from volume stats.",
    )
    surf_wm = ["rhCerebralWhiteMatter", "lhCerebralWhiteMatter", "CerebralWhiteMatter"]
    parser.add_argument(
        "--surf-wm-vol",
        action=_ExtendConstAction,
        dest="measures",
        const=_no_import(*surf_wm),
        help=help_add_measures(
            "Compute cortical white matter based on the surface:",
            surf_wm,
        ),
    )
    surf_ctx = ["rhCortex", "lhCortex", "Cortex"]
    parser.add_argument(
        "--surf-ctx-vol",
        action=_ExtendConstAction,
        dest="measures",
        const=_no_import(*surf_ctx),
        help=help_add_measures(
            "compute cortical gray matter based on the surface:",
            surf_ctx,
        ),
    )
    parser.add_argument(
        "--no_global_stats",
        action="store_const",
        dest="measures",
        const=[],
        help="Resets the computed global stats.",
    )
    parser.add_argument(
        "--empty",
        action="store_true",
        dest="empty",
        help="Report all segmentation labels in ctab, even if they are not in seg.",
    )
    # --ctab-out ctaboutput
    # --mask maskvol
    # --maskthresh thresh
    # --masksign sign
    # --maskframe frame
    # --maskinvert
    # --maskerode nerode
    brainseg = ["BrainSeg", "BrainSegNotVent"]
    parser.add_argument(
        "--brain-vol-from-seg",
        action=_ExtendConstAction,
        dest="measures",
        const=_no_import(*brainseg),
        help=help_add_measures("Compute measures BrainSeg measures:", brainseg),
    )

    def _mask(__value):
        return False, "Mask(" + str(__value) + ")"

    parser.add_argument(
        "--brainmask",
        type=_mask,
        metavar="brainmask",
        action="append",
        dest="measures",
        help="Report the Volume of the brainmask",
    )
    supratent = ["SupraTentorial", "SupraTentorialNotVent"]
    parser.add_argument(
        "--supratent",
        action=_ExtendConstAction,
        dest="measures",
        const=_no_import(*supratent),
        help=help_add_measures("Compute supratentorial measures:", supratent),
    )
    parser.add_argument(
        "--subcortgray",
        action="append_const",
        dest="measures",
        const=(False, "SubCortGray"),
        help=help_add_measures("Compute measure SubCortGray:", ["SubCortGray"]),
    )
    parser.add_argument(
        "--totalgray",
        action="append_const",
        dest="measures",
        const=(False, "TotalGray"),
        help=help_add_measures("Compute measure TotalGray:", ["TotalGray"]),
    )
    etiv_measures = [f"{k} (if also --brain-vol-from-seg)" for k in ETIV_RATIOS]
    parser.add_argument(
        "--etiv",
        action=_ExtendConstAction,
        dest="measures",
        const=_no_import(ETIV_FROM_TAL, ETIV_RATIO_KEY),
        help=help_add_measures("Compute eTIV:", [ETIV_FROM_TAL] + etiv_measures),
    )
    # --etiv-only
    # --old-etiv-only
    # --xfm2etiv xfm outfile
    surf_holes = ["rhSurfaceHoles", "lhSurfaceHoles", "SurfaceHoles"]
    parser.add_argument(
        "--euler",
        action=_ExtendConstAction,
        dest="measures",
        const=_no_import(*surf_holes),
        help=help_add_measures("Compute surface holes measures:", surf_holes),
    )
    # --avgwf textfile
    # --sumwf testfile
    # --avgwfvol mrivol
    # --avgwf-remove-mean
    # --sfavg textfile
    # --vox C R S
    # --replace ID1 ID2
    # --replace-file file
    # --gtm-default-seg-merge
    # --gtm-default-seg-merge-choroid
    # --ga-stats subject statsfile
    default_sd = Path(env["SUBJECTS_DIR"]) if "SUBJECTS_DIR" in env else None
    parser.add_argument(
        "--sd",
        dest="out_dir", metavar="subjects_dir", type=Path,
        default=default_sd,
        help="set SUBJECTS_DIR, defaults to environment SUBJECTS_DIR, required to find "
             "several files used by measures, e.g. surfaces.")
    parser.add_argument(
        "--subject",
        dest="sid", metavar="subject_id",
        help="set subject_id, required to find several files used by measures, e.g. "
             "surfaces.")
    parser.add_argument(
        "--seed",
        nargs=1, metavar="N", help="The seed has no effect")
    parser.add_argument(
        "--in-intensity-name",
        type=str,
        dest="norm_name",
        default="",
        help="name of the intensity image"
    )
    parser.add_argument(
        "--in-intensity-units",
        type=str,
        dest="norm_unit",
        default="",
        help="unit of the intensity image"
    )
    parser.add_argument(
        "--no_legacy",
        action="store_false",
        dest="legacy_freesurfer",
        help="use fastsurfer algorithms instead of fastsurfer."
    )
    return parser


def print_and_exit(args: object):
    """Print the commandline arguments of the segstats script to stdout and exit."""
    print(" ".join(format_cmdline_args(args)))
    import sys
    sys.exit(0)


def format_cmdline_args(args: object) -> list[str]:
    """Format the commandline arguments of the segstats script."""
    arglist = ["python", str(Path(__file__).parent / "segstats.py")]
    # this entry has nargs="+" and should therefore be up top
    if not empty(__ids := getattr(args, "ids", [])):
        arglist.extend(["--id"] + list(map(str, __ids)))

    def _append_storetrue(name: str, flag: str = ""):
        if flag == "":
            flag = "--" + name
        if getattr(args, name, False):
            arglist.append(flag)

    def _extend_arg(name: str, flag: str = None):
        if flag == "":
            flag = "--" + name
        if (value := getattr(args, name, None)) is not None:
            arglist.extend([flag, str(value)])

    _append_storetrue("legacy_freesurfer")
    _extend_arg("segfile")
    _extend_arg("normfile")
    _extend_arg("pvfile")
    _extend_arg("segstatsfile")
    _extend_arg("out_dir", "--sd")
    _extend_arg("sid")
    _extend_arg("threads")
    _extend_arg("lut")
    _extend_arg("volume_precision")

    measures: list[tuple[bool, str]] = getattr(args, "measures", [])
    if not empty(measures):
        arglist.append("measures")
        _extend_arg("measurefile", "--file")
        _flag = {True: "--import", False: "--compute"}
        blank_measure = (not measures[0][0], "")
        flag_measure_iter = ((_flag[i], m) for i, m in [blank_measure, *measures])
        arglist.extend(chain(
            (*((flag,) if flag != last_flag else ()), str(measure))
            for (last_flag, _), (flag, measure) in pairwise(flag_measure_iter)
        ))

    return arglist


if __name__ == "__main__":
    import sys

    args = make_arguments().parse_args()
    parse_actions = getattr(args, "parse_actions", [])
    for i, parse_action in sorted(parse_actions, key=lambda x: x[0], reverse=True):
        parse_action(args)
    sys.exit(main(args))
