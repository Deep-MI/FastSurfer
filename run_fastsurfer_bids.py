#!/usr/bin/env python3
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
BIDS-App entrypoint for FastSurfer.

Discovers subjects and sessions in a BIDS dataset and hands them to the existing FastSurfer
entrypoints: it writes a subject list and calls brun_fastsurfer.sh, or srun_fastsurfer.sh with
--slurm. It does not reimplement any part of the pipeline, and every option it does not define
itself is passed through unchanged.

Each session is processed on its own, as one cross-sectional case. Longitudinal processing, where
the timepoints of a subject are conditioned on a person-specific template, is a different
scientific method rather than a different spelling of this one, and is run with long_fastsurfer.sh.
"""

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from FastSurferCNN.utils.bids import BidsSession

FASTSURFER_HOME = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)

# set from bids_dir and output_dir, so a second one after the -- would silently win instead
RESERVED_PASSTHROUGH = ("--subject_list", "--subjects_list", "--sd", "--sid", "--t1", "--t2")


def make_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser for run_fastsurfer_bids.py.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Any options after a literal '--' are passed through unchanged to "
               "brun_fastsurfer.sh (or srun_fastsurfer.sh), e.g.:\n"
               "  run_fastsurfer_bids.py /bids /out participant -- --seg_only --3T",
    )
    parser.add_argument("bids_dir", type=Path, help="Path to the BIDS-valid input dataset.")
    parser.add_argument(
        "output_dir", type=Path,
        help="Output directory, used as FastSurfer's SUBJECTS_DIR. Each session becomes the "
             "directory <output_dir>/sub-<label>_ses-<label>.",
    )
    parser.add_argument(
        "analysis_level", choices=["participant", "group"],
        help="Level of analysis. Only 'participant' performs processing; 'group' is a no-op.",
    )
    parser.add_argument(
        "--participant_label", "--participant-label", dest="participant_label", nargs="+",
        default=None, metavar="LABEL",
        help="Restrict processing to these participant labels (with or without 'sub-' prefix). "
             "Default: process all subjects found.",
    )
    parser.add_argument(
        "--session_label", "--session-label", dest="session_label", nargs="+", default=None,
        metavar="LABEL",
        help="Restrict processing to these session labels (with or without 'ses-' prefix). "
             "Default: process all sessions found.",
    )
    parser.add_argument(
        "--use_t2", action="store_true",
        help="Also pass the T2w image of each session, which switches the hypothalamus module "
             "to its multimodal mode. Off by default, because whether a T2 is used changes the "
             "result, so it is a choice for the whole study rather than something the presence "
             "of a file decides. Sessions without a T2w are still processed, without one.",
    )
    parser.add_argument(
        "--skip_bids_validator", action="store_true",
        help="Skip validation of the input dataset against the BIDS specification. Validation "
             "uses the external bids-validator tool and is skipped with a warning if it is not "
             "installed.",
    )
    parser.add_argument(
        "--fs_license", type=Path, default=None,
        help="Path to the FreeSurfer license file (passed through to run_fastsurfer.sh).",
    )
    parser.add_argument(
        "--slurm", action="store_true",
        help="EXPERIMENTAL. Submit the cases to slurm via "
             "srun_fastsurfer.sh instead of running them locally via brun_fastsurfer.sh. Cluster "
             "options such as --partition or --work are passed through after the literal '--'. "
             "Check the output of --dry before relying on it.",
    )
    parser.add_argument(
        "--dry", "--dry_run", dest="dry", action="store_true",
        help="Print the commands that would be run, without executing them (same spelling as "
             "srun_fastsurfer.sh).",
    )
    return parser


def split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    """
    Split an argument vector at the first literal '--' into own and passed-through arguments.

    Parameters
    ----------
    argv : list[str]
        The argument vector, without the program name.

    Returns
    -------
    tuple[list[str], list[str]]
        The arguments this script parses itself, and those handed on unchanged.
    """
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1:]
    return argv, []


def subject_list_lines(sessions: "list[BidsSession]", quote: bool = True) -> list[str]:
    """
    Format discovered sessions as brun_fastsurfer.sh/srun_fastsurfer.sh subject list lines.

    Parameters
    ----------
    sessions : list[FastSurferCNN.utils.bids.BidsSession]
        The sessions to process.
    quote : bool, default=True
        Whether to shell-quote the paths. True for brun_fastsurfer.sh, False for
        srun_fastsurfer.sh, see the notes.

    Returns
    -------
    list[str]
        One ``<subject_id>=<t1 path>[ --t2 <t2 path>]`` line per session. Both scripts read this
        same format, though they do not read it the same way.

    Notes
    -----
    brun_fastsurfer.sh tokenizes the part after the ``=`` shell-style, so a path holding a space
    has to be quoted or it is read as the image plus a stray argument. BIDS labels cannot hold a
    space, but the directory the dataset sits in can.

    srun_fastsurfer.sh instead rewrites each path with awk, to point at the bind mount inside the
    container, and a leading quote stops that rewrite from matching. Its lines are therefore
    unquoted, and a path with a space is refused before it reaches this point.
    """
    escape = shlex.quote if quote else str
    lines = []
    for session in sessions:
        line = f"{session.output_id}={escape(str(session.t1w))}"
        if session.t2w is not None:
            line += f" --t2 {escape(str(session.t2w))}"
        lines.append(line)
    return lines


def main(argv: list[str] | None = None) -> int:
    """
    Run the BIDS-App entrypoint.

    Parameters
    ----------
    argv : list[str], optional
        Argument vector (defaults to sys.argv[1:]).

    Returns
    -------
    int
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    own_args, passthrough = split_passthrough(sys.argv[1:] if argv is None else argv)
    args = make_parser().parse_args(own_args)

    if args.analysis_level == "group":
        LOGGER.info("analysis_level 'group' is a no-op for FastSurfer, nothing to do.")
        return 0

    # --data only for slurm, where this script sets it from bids_dir; brun does not take it
    candidates = RESERVED_PASSTHROUGH + (("--data",) if args.slurm else ())
    reserved = [flag for flag in candidates if flag in passthrough]
    if reserved:
        LOGGER.error(
            "%s after the '--' would override what this script sets from bids_dir and "
            "output_dir. Remove it and use the positional arguments instead.",
            ", ".join(reserved),
        )
        return 1

    from FastSurferCNN.utils import bids
    from FastSurferCNN.version import read_and_close_version

    bids_dir: Path = args.bids_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not bids_dir.is_dir():
        LOGGER.error("%s is not a directory.", bids_dir)
        return 1
    # the output would land in the input dataset, where the next run also finds it as a subject
    if output_dir == bids_dir or (
        output_dir.is_relative_to(bids_dir)
        and output_dir.relative_to(bids_dir).parts[0].startswith("sub-")
    ):
        LOGGER.error(
            "output_dir %s is the BIDS dataset or inside one of its subjects. Use a directory "
            "outside of it, or %s.",
            output_dir, bids_dir / "derivatives" / "fastsurfer",
        )
        return 1

    # a mistyped label and an invalid dataset are both the user's input being wrong, which is
    # worth an error line rather than a traceback
    try:
        if not args.skip_bids_validator:
            bids.validate_dataset(bids_dir)
        sessions = bids.find_sessions(
            bids_dir,
            participant_labels=args.participant_label,
            session_labels=args.session_label,
            with_t2=args.use_t2,
        )
    except (ValueError, subprocess.CalledProcessError) as error:
        LOGGER.error("%s", error)
        return 1
    if not sessions:
        LOGGER.error("No session with a T1w image found in %s.", bids_dir)
        return 1

    # the hypothalamus module computes something different with a T2 than without one, so a run
    # where only some sessions have one holds two methods side by side
    if args.use_t2:
        without = [session.output_id for session in sessions if session.t2w is None]
        if without:
            LOGGER.warning(
                "--use_t2 was given, but %d of %d sessions have no T2w image and will be "
                "processed without one, so the hypothalamus results will not be comparable "
                "across the study: %s",
                len(without), len(sessions), without,
            )

    recorded = bids.read_processing_mode(output_dir)
    if recorded is not None and recorded != bids.CROSS_SECTIONAL:
        LOGGER.error(
            "%s already holds %s results. A session and a timepoint of that session claim the "
            "same directory name but are results of different methods, so pick another "
            "output_dir rather than mixing them.",
            output_dir, recorded,
        )
        return 1

    if args.slurm:
        spaced = sorted(
            session.output_id for session in sessions
            if any(ch.isspace() for ch in str(session.t1w))
            or (session.t2w is not None and any(ch.isspace() for ch in str(session.t2w)))
        )
        if spaced:
            LOGGER.error(
                "srun_fastsurfer.sh rewrites the image paths with awk, which has no notion of "
                "shell quoting, so until it does --slurm cannot take a path holding a space. "
                "Move the dataset, or run these without --slurm: %s",
                spaced,
            )
            return 1

    lines = subject_list_lines(sessions, quote=not args.slurm)
    # top level rather than in a scripts/ directory: scripts/ is what FreeSurfer calls the
    # per-subject log directory, and this sits beside the subjects rather than inside one
    subject_list = output_dir / "bids_subjects.txt"
    script = "srun_fastsurfer.sh" if args.slurm else "brun_fastsurfer.sh"
    cmd = [
        str(FASTSURFER_HOME / script),
        "--subject_list", str(subject_list),
        "--sd", str(output_dir),
    ]
    if args.slurm:
        # srun rewrites every path in the list relative to --data before binding it into the
        # container, and its default is the working directory, which BIDS paths need not sit under
        cmd += ["--data", str(bids_dir)]
    cmd += passthrough
    if args.fs_license is not None:
        cmd += ["--fs_license", str(args.fs_license)]
    if args.dry:
        # nothing is written either, so the list is printed where it would have gone. The command
        # is printed rather than run with its own --dry, which would read the list that is not there
        print(f"+ (dry) would write {subject_list}:")
        print("".join(f"    {line}\n" for line in lines), end="")
        print("+ " + shlex.join(cmd))
        return 0

    bids.write_derivatives_dataset_description(
        output_dir, read_and_close_version(), bids.CROSS_SECTIONAL
    )
    subject_list.parent.mkdir(parents=True, exist_ok=True)
    subject_list.write_text("\n".join(lines) + "\n")
    print("+ " + shlex.join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
