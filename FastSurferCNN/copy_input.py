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
Put the input images into the subject directory: an archival copy, and the rawavg the surfaces need.

``mri/orig/001.<ext>`` is a byte-for-byte copy of the input, in the format it arrived in, so no
header field, scale factor or data type is lost. Nothing in FastSurfer reads it back; it is there so
the subject directory records what was processed, which is also why an archive left by a different
input stops the run. A T2 is copied the same way, as ``mri/orig/T2raw.<ext>``.

``mri/rawavg.mgz`` is the copy the tools read. ``pctsurfcon`` builds that path itself and hardcodes
the name, so it has to be an MGH file whatever the input was. Where the input is already an .mgz it
is a symlink to the archival copy; otherwise it is converted, through ``save_image`` rather than
nibabel directly, because a plain nibabel write leaves ``fov`` at 0 and FreeSurfer reports that
field as the field of view. The T2 equivalent is ``mri/orig/T2raw.mgz``, which for an .mgz input is
the archival copy itself.

The names follow recon-all: it converts a ``-T2`` input to ``mri/orig/T2raw.mgz`` with
``--no_scale 1``, and ``samseg`` and ``-T2pial`` look for it there. 001 is historic, 001, 002 and so
on being the separate runs of one session that FreeSurfer registered and averaged into rawavg;
FastSurfer takes a single input and conforms it instead, so rawavg here is that one input rather
than an average.
"""

import argparse
import filecmp
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from FastSurferCNN.data_loader.data_utils import save_image, storable_dtype
from FastSurferCNN.utils import logging

LOGGER = logging.getLogger(__name__)

# both relative to the subject's mri directory, so the shapes match and neither name lies
RAWAVG_PATH = Path("rawavg.mgz")
T2_RAWAVG_PATH = Path("orig/T2raw.mgz")


def image_suffix(path: Path) -> str:
    """
    The extension to give a copy of `path`, keeping a compound one such as .nii.gz intact.

    Parameters
    ----------
    path : Path
        The file whose extension is wanted.

    Returns
    -------
    str
        The extension, including the leading dot.
    """
    suffixes = path.suffixes
    if len(suffixes) >= 2 and suffixes[-1] == ".gz":
        return "".join(suffixes[-2:])
    return path.suffix


def archive_input(
        source: Path,
        orig_dir: Path,
        stem: str = "001",
        derived: Path | None = None,
) -> Path:
    """
    Copy `source` into `orig_dir` as `stem` plus the source's own extension.

    The copy is byte for byte, so it is the input and not a re-encoding of it. Formats that spread
    one image over several files, such as an Analyze .hdr and .img pair, are copied whole, and every
    part keeps `stem` so they still find each other.

    An archive already in `orig_dir` is compared with `source` byte for byte. Equal means this is a
    re-run and the copy is skipped; anything else is a second input arriving in a directory that
    belongs to the first, which nothing downstream could tell apart, so it raises. Comparing the
    bytes rather than only the file names catches the case where the extension is the same and the
    image is not.

    Parameters
    ----------
    source : Path
        The image the user passed.
    orig_dir : Path
        The `mri/orig` directory of the subject, created if it does not exist.
    stem : str, default="001"
        The name to give the copy, without an extension.
    derived : Path, optional
        A file the caller writes into `orig_dir` itself, the T2 rawavg in practice, which shares
        `stem` and so matches the same names as an archive. It is left out of the check, but only
        once the archive of this very input has been found unchanged, so that switching to a
        different input still reports the old copies rather than overwriting them.

    Returns
    -------
    Path
        The path of the copy, the one nibabel names when it loads the image.

    Raises
    ------
    FileExistsError
        If an archive for `stem` exists and is not this input.
    """
    orig_dir.mkdir(parents=True, exist_ok=True)
    # every file of the image, so a .hdr/.img pair is not split in half
    file_map = nib.load(source).file_map
    parts = {key: Path(holder.filename) for key, holder in file_map.items()}
    destinations = {key: orig_dir / f"{stem}{image_suffix(part)}" for key, part in parts.items()}
    by_name = {destination.name: key for key, destination in destinations.items()}

    present = {p.name for p in orig_dir.glob(f"{stem}.*")}
    unchanged = {
        name for name in present & set(by_name)
        if filecmp.cmp(parts[by_name[name]], destinations[by_name[name]], shallow=False)
    }
    derived_name = derived.name if derived is not None and derived.parent == orig_dir else None
    # the archive of this input is already here, so this is a re-run and whatever else the caller
    # derives into the directory is its own output from the previous one, not a competing archive
    rerun = destinations["image"].name in unchanged
    ignored = {derived_name} if rerun and derived_name is not None else set()
    conflicting = sorted(present - unchanged - ignored)
    if conflicting:
        # one file to point at, and not the rawavg we write ourselves: it is an output, so passing
        # it back would leave the archive beside it and fail again. Either half of a pair loads.
        archived = [name for name in conflicting if name != derived_name] or conflicting
        raise FileExistsError(
            f"{orig_dir} already holds {', '.join(conflicting)}, which is not {source}. One subject "
            f"directory belongs to one input: to reprocess the image archived here, pass "
            f"{orig_dir / archived[0]} as the input, and to process a different image, give it a "
            f"subject id of its own."
        )

    for key, part in parts.items():
        destination = destinations[key]
        if destination.name in unchanged:
            LOGGER.info(f"{destination} is already this input, not copying it again.")
            continue
        LOGGER.info(f"Copying {part} to {destination}")
        shutil.copyfile(part, destination)
    return destinations["image"]


def write_rawavg(source: Path, rawavg: Path, archive: Path | None = None) -> None:
    """
    Provide `rawavg` as an MGH file, by symlink where the input is one already and by conversion else.

    Parameters
    ----------
    source : Path
        The image the user passed, which the voxels are read from. Reading the input rather than the
        archival copy matters when the subject directory is on slower storage.
    rawavg : Path
        The `mri/rawavg.mgz` to create.
    archive : Path, optional
        The archival copy of the input, if one was made. An .mgz one is linked to instead of being
        written a second time.
    """
    rawavg.parent.mkdir(parents=True, exist_ok=True)
    if archive is not None and archive.resolve() == rawavg.resolve():
        # an .mgz T2, whose archival copy is already at the name the tools read
        LOGGER.info(f"{rawavg} is the archival copy, nothing to convert.")
        return
    if rawavg.is_symlink() or rawavg.exists():
        rawavg.unlink()

    if archive is not None and image_suffix(archive) == ".mgz":
        # relative, so the subject directory stays movable; relpath rather than relative_to, which
        # refuses any layout where the archive is not below the link
        target = Path(os.path.relpath(archive, rawavg.parent))
        try:
            rawavg.symlink_to(target)
            LOGGER.info(f"Linking {rawavg} to {target}")
            return
        except OSError as error:
            LOGGER.info(f"Could not link {rawavg} ({error}), copying instead.")
            shutil.copyfile(archive, rawavg)
            return

    if image_suffix(source) == ".mgz":
        # already the right container, and a copy keeps rawavg inside the subject directory rather
        # than linking out to an input that may not stay where it is
        LOGGER.info(f"Copying {source} to {rawavg}")
        shutil.copyfile(source, rawavg)
        return

    LOGGER.info(f"Converting {source} to {rawavg}")
    image = nib.load(source)
    data = np.asanyarray(image.dataobj)
    save_image(image.header, image.affine, data, rawavg, dtype=storable_dtype(data))


def make_parser() -> argparse.ArgumentParser:
    """Create the command line interface."""
    parser = argparse.ArgumentParser(
        description="Copy the input images into the subject directory and provide mri/rawavg.mgz.",
    )
    parser.add_argument("--t1", type=Path, required=True, help="the T1 image the user passed")
    parser.add_argument("--t2", type=Path, default=None, help="the T2 image, if one was passed")
    parser.add_argument("--sd", type=Path, required=True, help="the subjects directory")
    parser.add_argument("--sid", required=True, help="the subject id")
    parser.add_argument(
        "--rawavg_only",
        action="store_true",
        help="write rawavg but no archival copy, for an input the pipeline built itself rather than "
             "one the user passed, such as a longitudinal time point resampled into base space",
    )
    return parser


def main(
        t1: Path,
        sd: Path,
        sid: str,
        t2: Path | None = None,
        rawavg_only: bool = False,
) -> int:
    """
    Copy the inputs into the subject directory and provide rawavg.

    Parameters
    ----------
    t1 : Path
        The T1 image the user passed.
    sd : Path
        The subjects directory.
    sid : str
        The subject id.
    t2 : Path, optional
        The T2 image, if one was passed.
    rawavg_only : bool, default=False
        Write rawavg but no archival copy.

    Returns
    -------
    int
        0 on success, 1 if an input is missing or an archive of a different image is already there.
    """
    if not t1.is_file():
        LOGGER.error(f"The T1 file {t1} does not exist.")
        return 1
    mri_dir = sd / sid / "mri"
    # (modality, source, name for the archival copy, path of the mgz the tools read)
    inputs = [("T1", t1, "001", RAWAVG_PATH)]
    if t2 is not None:
        if not t2.is_file():
            LOGGER.error(f"The T2 file {t2} does not exist.")
            return 1
        inputs.append(("T2", t2, "T2raw", T2_RAWAVG_PATH))

    for _modality, source, stem, rawavg in inputs:
        archive = None
        if not rawavg_only:
            try:
                archive = archive_input(
                    source, mri_dir / "orig", stem=stem, derived=mri_dir / rawavg,
                )
            except FileExistsError as error:
                LOGGER.error(str(error))
                return 1
        write_rawavg(source, mri_dir / rawavg, archive=archive)
    return 0


if __name__ == "__main__":
    logging.setup_logging()
    sys.exit(main(**vars(make_parser().parse_args())))
