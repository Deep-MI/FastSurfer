#!/usr/bin/env python3
# Copyright 2026 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

Discovers subjects/sessions in a BIDS-valid dataset and delegates processing to the
existing FastSurfer entrypoints (brun_fastsurfer.sh for cross-sectional subjects,
long_fastsurfer.sh for subjects with multiple sessions).
"""

import argparse
import subprocess
import sys
from pathlib import Path

FASTSURFER_HOME = Path(__file__).resolve().parent


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
               "run_fastsurfer.sh/brun_fastsurfer.sh/long_fastsurfer.sh, e.g.:\n"
               "  run_fastsurfer_bids.py /bids /out participant -- --parallel --3T",
    )
    parser.add_argument("bids_dir", type=Path, help="Path to the BIDS-valid input dataset.")
    parser.add_argument(
        "output_dir", type=Path,
        help="Output directory (used as FastSurfer's SUBJECTS_DIR).",
    )
    parser.add_argument(
        "analysis_level", choices=["participant", "group"],
        help="Level of analysis. Only 'participant' performs processing; 'group' is a no-op.",
    )
    parser.add_argument(
        "--participant_label", "--participant-label", dest="participant_label", nargs="+", default=None,
        metavar="LABEL",
        help="Restrict processing to these participant labels (with or without 'sub-' prefix). "
             "Default: process all subjects found.",
    )
    parser.add_argument(
        "--session_label", "--session-label", dest="session_label", nargs="+", default=None,
        metavar="LABEL",
        help="Restrict processing to these session labels (with or without 'ses-' prefix). "
             "Default: process all sessions found.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--longitudinal", action="store_true",
        help="Force longitudinal processing for all selected subjects (requires >=2 sessions each).",
    )
    mode.add_argument(
        "--cross_sectional", action="store_true",
        help="Force cross-sectional processing of every session independently, even for subjects "
             "with multiple sessions (default: subjects with >=2 sessions are processed longitudinally).",
    )
    parser.add_argument(
        "--skip_bids_validator", action="store_true",
        help="Skip validation of the input dataset against the BIDS specification.",
    )
    parser.add_argument(
        "--fs_license", type=Path, default=None,
        help="Path to the FreeSurfer license file (passed through to run_fastsurfer.sh).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print the commands that would be run, without executing them.",
    )
    return parser


def _split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1:]
    return argv, []


def _write_subject_list(path: Path, lines: list[str]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _run(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


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
    argv = sys.argv[1:] if argv is None else argv
    own_args, passthrough = _split_passthrough(argv)
    args = make_parser().parse_args(own_args)

    if args.analysis_level == "group":
        print("analysis_level 'group' is a no-op for FastSurfer; nothing to do.")
        return 0

    from FastSurferCNN.utils import bids
    from FastSurferCNN.version import read_and_close_version

    bids_dir: Path = args.bids_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    subjects = bids.find_subjects(
        bids_dir,
        participant_labels=args.participant_label,
        session_labels=args.session_label,
        validate=not args.skip_bids_validator,
    )
    if not subjects:
        print(f"ERROR: No subjects with a T1w image found in {bids_dir}.", file=sys.stderr)
        return 1

    if not args.dry_run:
        bids.write_derivatives_dataset_description(output_dir, read_and_close_version())

    common_args = list(passthrough)
    if args.fs_license is not None:
        common_args += ["--fs_license", str(args.fs_license)]

    cross_sectional_lines = []
    for subject in subjects:
        force_long = args.longitudinal
        force_cross = args.cross_sectional
        is_long = subject.is_longitudinal and not force_cross
        if force_long and not subject.is_longitudinal:
            print(
                f"WARNING: --longitudinal requested but {subject.subject_id} only has a single "
                "session, processing cross-sectionally instead.",
                file=sys.stderr,
            )
            is_long = False
        elif force_long:
            is_long = True

        if is_long:
            tpids = [subject.output_id(s) for s in subject.sessions]
            t1s = [str(s.t1w) for s in subject.sessions]
            if any(s.t2w is not None for s in subject.sessions):
                print(
                    f"WARNING: {subject.subject_id} has T2w images, but T2w/HypVINN is not supported "
                    "in the longitudinal pipeline; ignoring T2w for this subject.",
                    file=sys.stderr,
                )
            cmd = [
                str(FASTSURFER_HOME / "long_fastsurfer.sh"),
                "--tid", subject.subject_id,
                "--tpids", *tpids,
                "--t1s", *t1s,
                "--sd", str(output_dir),
                *common_args,
            ]
            _run(cmd, args.dry_run)
        else:
            for session in subject.sessions:
                sid = subject.output_id(session)
                line = f"{sid}={session.t1w}"
                if session.t2w is not None:
                    line += f" --t2 {session.t2w}"
                cross_sectional_lines.append(line)

    if cross_sectional_lines:
        subject_list_path = output_dir / "scripts" / "bids_subjects.txt"
        if not args.dry_run:
            subject_list_path.parent.mkdir(parents=True, exist_ok=True)
            _write_subject_list(subject_list_path, cross_sectional_lines)
        else:
            print(f"+ (dry run) would write subject list to {subject_list_path}:")
            for line in cross_sectional_lines:
                print(f"    {line}")
        cmd = [
            str(FASTSURFER_HOME / "brun_fastsurfer.sh"),
            "--subject_list", str(subject_list_path),
            "--sd", str(output_dir),
            *common_args,
        ]
        _run(cmd, args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
