#!/usr/bin/env python3

# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path
from subprocess import CompletedProcess

import nibabel as nib

VERSION = "1.0 2025-08-18 by kueglerd @ DZNE"


class FastSurferCompatError(Exception):
    """Custom exception for FastSurfer compatibility issues"""
    pass


def validate_existing_file(value: str) -> Path:
    """Validate that the input is an existing file path"""
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File does not exist: {value}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Path is not a file: {value}")
    return path


def validate_existing_subjects_dir(value: str) -> Path:
    """Validate that the input is an existing directory path"""
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {value}")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Path is not a directory: {value}")
    return path

def check_freesurfer(check_version: bool = True) -> None:
    """Check if FreeSurfer is properly installed and version is supported"""
    freesurfer_home = os.environ.get("FREESURFER_HOME")
    if not freesurfer_home:
        raise FastSurferCompatError(
            f"Did not find $FREESURFER_HOME. A working version of FreeSurfer {get_supported_freesurfer_version()} is "
            f"needed to run {Path(__name__).name} locally. Make sure to export and source FreeSurfer before running "
            f"this script:\n"
            f"  export FREESURFER_HOME=/path/to/your/local/fs{get_supported_freesurfer_version()}\n"
            f"  source $FREESURFER_HOME/SetUpFreeSurfer.sh"
        )

    if check_version:
        build_stamp_file = Path(freesurfer_home) / "build-stamp.txt"
        if build_stamp_file.exists():
            with open(build_stamp_file) as f:
                version_content = f.read().strip()
            fs_support = get_supported_freesurfer_version()
            if fs_support not in version_content:
                raise FastSurferCompatError(
                    f"You are trying to run recon-surf with FreeSurfer version {version_content}. We are currently "
                    f"supporting only FreeSurfer {fs_support}. Therefore, make sure to export and source the correct "
                    f"FreeSurfer version before running this script:\n"
                    f"  export FREESURFER_HOME=/path/to/your/local/fs{fs_support}\n"
                    f"  source $FREESURFER_HOME/SetUpFreeSurfer.sh"
                )
        else:
            raise FastSurferCompatError("Could not find/read FreeSurfer build-stamp file.")

def softlink_or_copy(source: str | Path, target: str | Path) -> None:
    """Create soft link or copy file if linking fails"""
    source_path = Path(source)
    target_path = Path(target)

    if target_path.exists():
        target_path.unlink()

    try:
        target_path.symlink_to(source_path)
    except OSError:
        # If symlink fails, copy the file
        from shutil import copy2

        if not target_path.is_absolute():
            target_path = (source.parent / target_path).resolve()
        copy2(source_path, target_path)


def get_voxel_size(image_file: Path) -> float:
    """Get voxel size from aseg file using nibabel."""
    try:
        img = nib.load(str(image_file))
        return float(img.header.get_zooms()[0])
    except Exception as e:
        raise FastSurferCompatError(f"Could not read voxel size from {image_file}: {e}") from e


@lru_cache
def get_supported_freesurfer_version() -> str:
    """Get FreeSurfer version from FreeSurfer from recon-surf.sh."""
    recon_surf_script = Path(__file__).parent / "recon-surf.sh"
    with recon_surf_script.open() as fp:
        for line in fp.readlines():
            if line.lstrip().startswith("FS_VERSION_SUPPORT"):
                return line.strip().removeprefix("FS_VERSION_SUPPORT").lstrip(" =").strip("\"")
    return "7.4.1"


def run(command: list[str], *args, **kwargs) -> CompletedProcess:
    """Run the FreeSurfer command `command`."""
    from shutil import which
    from subprocess import run as _run

    executable, *arguments = command
    # do we have executable in PATH?
    if not (_executable := which(executable)):
        # if not, do we have it in FREESURFER_HOME/bin ?
        _executable = which(f"{os.environ['FREESURFER_HOME']}/bin/{executable}")
    if _executable is None:
        raise FastSurferCompatError(f"Could not find the FreeSurfer executable {executable}.")

    print((args, kwargs))

    return _run([_executable, *arguments], *args, **kwargs)


def make_parser() -> argparse.ArgumentParser:
    """Parse command line arguments"""
    from textwrap import dedent

    def int_gt_zero(value: str | int) -> int:
        val = int(value)
        if val <= 0:
            raise ValueError("Invalid value, must not be negative.")
        return val

    parser = argparse.ArgumentParser(
        description="long_compat_segmentHA.py takes a longitudinally processed subject and creates files "
                    "missing for other longitudinal processing like the hippocampal subfields stream of FreeSurfer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
               REFERENCES:

               If you use this for research publications, please cite:

               Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M, FastSurfer - A fast and accurate deep 
                learning based neuroimaging pipeline, NeuroImage 219 (2020), 117012. 
                https://doi.org/10.1016/j.neuroimage.2020.117012

               Henschel L, Kuegler D (co-first), Reuter M.. FastSurferVINN: Building Resolution-Independence into 
                Deep Learning Segmentation Methods - A Solution for HighRes Brain MRI. NeuroImage 251 (2022), 118933. 
                http://dx.doi.org/10.1016/j.neuroimage.2022.118933
               """)
    )

    # Required arguments
    parser.add_argument("--sid", "--tid", dest="subject", required=True, metavar="subject id",
                        help="Template to create directory inside $SUBJECTS_DIR.")
    parser.add_argument("--sd", dest="subjects_dir", type=validate_existing_subjects_dir,
                        default=Path(os.environ.get("SUBJECTS_DIR", "/")), required="SUBJECTS_DIR" not in os.environ,
                        help=f"Output directory, default: SUBJECTS_DIR env var ({os.environ.get('SUBJECTS_DIR', '')}).")

    # Optional arguments
    parser.add_argument("--threads", type=int_gt_zero, default=1,
                        help="Set openMP and ITK threads to <int>.")
    parser.add_argument("--fs_license", dest="fs_license", type=validate_existing_file, default=Path("."),
                        help="Path to FreeSurfer license key file.")
    parser.add_argument("--version", action="version", version=f"long_compat_segmentHA: {VERSION}")

    # Dev flags
    parser.add_argument("--ignore_fs_version", action="store_true",
                        help=f"Switch on to avoid check for FreeSurfer version. Program will otherwise terminate "
                             f"if FreeSurfer {get_supported_freesurfer_version()} is not sourced.")

    return parser


def validate_inputs(subject_dir: Path):
    """Validate all input arguments and environment"""
    # Check path length
    if len(str(subject_dir)) > 185:
        raise FastSurferCompatError(
            "Subject directory path is very long. This is known to cause errors due to some commands run by "
            "FreeSurfer versions built for Ubuntu. --sd + --sid should be less than 185 characters long."
        )

    # Check if subject exists and has required files
    tpfile = subject_dir / "base-tps.fastsurfer"
    if not tpfile.exists():
        raise FastSurferCompatError(
            f"{subject_dir.name} is either not found in $SUBJECTS_DIR or it is not a longitudinal template directory "
            f"(base), which needs to contain base-tps.fastsurfer file. Please ensure that the base (template) has been "
            f"created with long_prepare_template.sh."
        )

    # Check if already processed
    segfile = subject_dir / "mri" / "aparc.DKTatlas+aseg.mapped.mgz"
    if segfile.exists():
        raise FastSurferCompatError(
            f"{segfile} already exists! The output directory must not contain the generated files! {segfile}, "
            f".../mri/aparc.DKTatlas+aseg.mgz, .../mri/aparc+aseg.mgz, .../mri/wmparc.DKTatlas.mapped.mgz, "
            f".../mri/wmparc.mgz, and .../base-tps ."
        )


def main(subjects_dir: Path, subject: str, fs_license: Path, threads: int = 1, ignore_fs_version: bool = False) -> None:
    """
    Takes a longitudinally processed subject and creates files missing for other longitudinal processing like the
    hippocampal subfields stream of FreeSurfer.

    Parameters
    ----------
    subjects_dir : Path
        The subjects_dir with one subject per subdir.
    subject : str
        The subject id.
    fs_license : Path
        The path to FreeSurfer license file.
    threads : int, default=1
        The number of threads to use.
    ignore_fs_version : bool, default=False
        Ignore FreeSurfer version if True.
    """
    print(f"\nlong_compat_segmentHA: {VERSION}")
    print(f"sid {subject}")
    print()

    try:
        # Check FreeSurfer
        check_freesurfer(not ignore_fs_version)

        # Validate inputs
        subject_dir = subjects_dir / subject
        validate_inputs(subject_dir)

        # Set threading environment
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(threads)
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)

        if Path(".") != fs_license:
            env["FS_LICENSE"] = str(fs_license)
        elif env.get("FS_LICENSE", None) and not Path(env["FS_LICENSE"]).exists():
            raise FastSurferCompatError("No valid --fs_license passed, but environment FS_LICENSE does not exist!")
        # else FS_LICENSE will default to automatic search

        env["SUBJECTS_DIR"] = str(subjects_dir)
        if env["FREESURFER_HOME"] not in env["PATH"]:
            env["PATH"] = env["FREESURFER_HOME"] + "/bin:" + env["PATH"]

        # Create FreeSurfer-compatible timepoint symlinks
        tpfile = subject_dir / "base-tps.fastsurfer"
        print("Copying/symlinking timepoints...")
        with open(subject_dir / tpfile) as f:
            for long_id in filter(lambda x: bool(x), map(str.strip, f.readlines())):
                target = f"{long_id}.long.{subject}"
                if not (subjects_dir / long_id).exists():
                    raise FastSurferCompatError(f"The longitudinal processing {long_id} could not be found!")

                softlink_or_copy(long_id, subjects_dir / target)
                print(f"Created link: {long_id} -> {target}")

        # Setup variables
        mdir = subject_dir / "mri"
        ldir = subject_dir / "label"
        sdir = subject_dir / "surf"
        aseg_segfile = mdir / "aseg.mgz"
        in_segfile = mdir / "aparc.DKTatlas+aseg.deep.mgz"

        # Get voxel size and determine hires flag
        vox_size = get_voxel_size(in_segfile)
        hires_voxsize_threshold = 0.999

        hires = vox_size < hires_voxsize_threshold
        print(f"The voxel size {vox_size} is {'' if hires else 'not '}less than {hires_voxsize_threshold}, so we are "
              f"proceeding with {'hires' if hires else 'standard'} options.")

        # CREATE cortical ribbon (approx 5mins)
        # CREATE ASEG
        # 25 sec hyporelabel run whatever else can be done without sphere, cortical ribbon and segmentations
        # -hyporelabel creates aseg.presurf.hypos.mgz from aseg.presurf.mgz
        # -apas2aseg creates aseg.mgz by editing aseg.presurf.hypos.mgz with surfaces
        print("Creating cortical ribbon...")
        os.umask(_umask := os.umask(0o22))
        cmd = ["recon-all", "-subject", subject, "-cortribbon", "-hyporelabel", "-apas2aseg", "-umask", f"{_umask:o}"]
        if hires:
            cmd.append("-hires")
        if threads > 1:
            cmd.extend(["-threads", str(threads), "-itkthreads", str(threads)])

        print("Running: cortical ribbon creation")

        completed = run(cmd, env=env)
        if completed.returncode != 0:
            raise FastSurferCompatError("Creating cortical ribbon failed!")

        # mapped aparcDKT to vol (1:30 min)
        segfile = mdir / "aparc.DKTatlas+aseg.mapped.mgz"
        wm_segfile = mdir / "wmparc.DKTatlas.mapped.mgz"
        threads_flags = ("--threads", str(threads))

        def hemi_flags(hemi: str, offset: int) -> list:
            return [
                f"--{hemi}-annot", ldir / f"{hemi}.aparc.DKTatlas.mapped.annot", str(offset),
                f"--{hemi}-cortex-mask", ldir / f"{hemi}.cortex.label",
                f"--{hemi}-white", sdir / f"{hemi}.white", f"--{hemi}-pial", sdir / f"{hemi}.pial",
            ]

        print("Mapping aparcDKT to volume...")
        cmd = ["mri_surf2volseg", "--o", segfile, "--label-cortex", "--i", aseg_segfile, *threads_flags]

        for hemi, offset in (("lh", 1000), ("rh", 2000)):
            cmd.extend(hemi_flags(hemi, offset))

        completed = run(list(map(str, cmd)), env=env)
        if completed.returncode != 0:
            raise FastSurferCompatError("Mapping aparcDKT failed!")

        # mapped wmparc (1:30 min)
        print("Mapping wmparc...")
        cmd = ["mri_surf2volseg", "--o", wm_segfile, "--label-wm", "--i", segfile, *threads_flags]

        for hemi, offset in (("lh", 3000), ("rh", 4000)):
            cmd.extend(hemi_flags(hemi, offset))

        completed = run(list(map(str, cmd)), env=env)
        if completed.returncode != 0:
            raise FastSurferCompatError("Mapping wmparc failed!")

        # Create symbolic links
        print("Creating symbolic links...")

        softlink_or_copy(segfile.name, mdir / "aparc.DKTatlas+aseg.mgz")
        softlink_or_copy(segfile.name, mdir / "aparc+aseg.mgz")
        softlink_or_copy("wmparc.DKTatlas.mapped.mgz", mdir / "wmparc.mgz")

        softlink_or_copy("base-tps.fastsurfer", subject_dir / "base-tps")

        print(f"Processing {subject} completed successfully!")

    except FastSurferCompatError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(130)
    except Exception as e:
        from traceback import print_exception
        print("Unexpected error:")
        print_exception(e)
        sys.exit(1)

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    main(
        subjects_dir=args.subjects_dir,
        subject=args.subject,
        fs_license=args.fs_license,
        threads=args.threads,
        ignore_fs_version=args.ignore_fs_version,
    )
