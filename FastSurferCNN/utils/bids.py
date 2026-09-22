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
Discovery of subjects, sessions and anatomical images in a BIDS dataset.

Deliberately the standard library and nothing else. What run_fastsurfer_bids.py needs from a
BIDS dataset is the subject label, the session label and the T1w/T2w files, which the BIDS
directory layout spells out in the paths themselves: ``sub-<label>/[ses-<label>/]anat/*_T1w.nii.gz``.
That is a glob and a filename suffix, so pulling pybids (and its pandas, sqlalchemy, formulaic and
num2words) into the Docker image, the macOS bundle and the dependency baseline buys nothing here.

Validation is the one thing that genuinely needs the specification rather than the layout, and it
is delegated to the ``bids-validator`` command line tool if the user has it, rather than made a
hard dependency of FastSurfer.
"""

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# BIDS permits both, and the nifti suffix is what separates an image from its json sidecar
NIFTI_EXTENSIONS = (".nii", ".nii.gz")


@dataclass
class BidsSession:
    """One anatomical session of one subject, i.e. one FastSurfer case."""

    subject_id: str
    session_id: str | None
    t1w: Path
    t2w: Path | None = None

    @property
    def output_id(self) -> str:
        """
        Return the FastSurfer subject id for this session, which names its output directory.

        Returns
        -------
        str
            ``sub-<label>_ses-<label>``, or just ``sub-<label>`` in a dataset that has no
            session level.
        """
        if self.session_id is None:
            return self.subject_id
        return f"{self.subject_id}_{self.session_id}"


def _find_image(anat_dir: Path, suffix: str) -> Path | None:
    """Return the anatomical image with this suffix in anat_dir, or None if there is none."""
    images = sorted(
        path for path in anat_dir.glob(f"*_{suffix}.nii*")
        if path.name.endswith(NIFTI_EXTENSIONS)
    )
    if len(images) > 1:
        LOGGER.warning(
            "%s holds %d %s images, using %s. Pick one explicitly with run_fastsurfer.sh if that "
            "is the wrong one: %s",
            anat_dir, len(images), suffix, images[0].name, [p.name for p in images],
        )
    return images[0] if images else None


def _labels(labels: list[str] | None, prefix: str) -> set[str] | None:
    """Normalise user-supplied labels to full BIDS directory names, e.g. 01 or sub-01 to sub-01."""
    if not labels:
        return None
    return {f"{prefix}{label.removeprefix(prefix)}" for label in labels}


def find_sessions(
    bids_dir: Path,
    participant_labels: list[str] | None = None,
    session_labels: list[str] | None = None,
) -> list[BidsSession]:
    """
    Discover every session with a T1w image in a BIDS dataset.

    Parameters
    ----------
    bids_dir : Path
        Path to the root of the BIDS dataset.
    participant_labels : list[str], optional
        If given, only these subjects, with or without the ``sub-`` prefix.
    session_labels : list[str], optional
        If given, only these sessions, with or without the ``ses-`` prefix.

    Returns
    -------
    list[BidsSession]
        One entry per session that has a T1w image, ordered by subject and then session.
        Sessions without a T1w are skipped with a warning, since FastSurfer cannot process them.

    Raises
    ------
    ValueError
        If a requested participant or session label is not in the dataset. A typo in
        ``--participant_label`` or ``--session_label`` would otherwise silently process nothing,
        or, worse, everything else.
    """
    subject_dirs = sorted(path for path in bids_dir.glob("sub-*") if path.is_dir())
    wanted_subjects = _labels(participant_labels, "sub-")
    if wanted_subjects is not None:
        missing = wanted_subjects - {path.name for path in subject_dirs}
        if missing:
            raise ValueError(
                f"Requested participant label(s) not found in {bids_dir}: {sorted(missing)}"
            )
        subject_dirs = [path for path in subject_dirs if path.name in wanted_subjects]

    session_dirs = {
        subject_dir: sorted(path for path in subject_dir.glob("ses-*") if path.is_dir())
        for subject_dir in subject_dirs
    }
    wanted_sessions = _labels(session_labels, "ses-")
    if wanted_sessions is not None:
        present = {path.name for paths in session_dirs.values() for path in paths}
        missing = wanted_sessions - present
        if missing:
            raise ValueError(
                f"Requested session label(s) not found in {bids_dir}: {sorted(missing)}"
            )

    sessions: list[BidsSession] = []
    for subject_dir, subject_session_dirs in session_dirs.items():
        if subject_session_dirs:
            if wanted_sessions is not None:
                subject_session_dirs = [
                    path for path in subject_session_dirs if path.name in wanted_sessions
                ]
                if not subject_session_dirs:
                    continue  # this subject has none of the requested sessions
            anat_dirs = [(path.name, path / "anat") for path in subject_session_dirs]
        elif wanted_sessions is not None:
            # a dataset may mix the two layouts; a subject with no session level has nothing that
            # the requested session could name, and processing it anyway would ignore the filter
            LOGGER.warning(
                "%s has no session level and --session_label was given, skipping the subject.",
                subject_dir.name,
            )
            continue
        else:
            # a dataset without a session level keeps anat directly under the subject
            anat_dirs = [(None, subject_dir / "anat")]

        found = 0
        for session_id, anat_dir in anat_dirs:
            t1w = _find_image(anat_dir, "T1w")
            if t1w is None:
                LOGGER.warning("No T1w image in %s, skipping it.", anat_dir)
                continue
            sessions.append(
                BidsSession(
                    subject_id=subject_dir.name,
                    session_id=session_id,
                    t1w=t1w,
                    t2w=_find_image(anat_dir, "T2w"),
                )
            )
            found += 1
        if not found:
            LOGGER.warning(
                "%s has no session with a T1w image, skipping the subject.", subject_dir.name
            )

    return sessions


def validate_dataset(bids_dir: Path) -> None:
    """
    Validate bids_dir with the ``bids-validator`` command line tool, if it is installed.

    Parameters
    ----------
    bids_dir : Path
        Path to the root of the BIDS dataset.

    Raises
    ------
    subprocess.CalledProcessError
        If the validator reports the dataset as invalid.
    """
    validator = shutil.which("bids-validator")
    if validator is None:
        LOGGER.warning(
            "bids-validator is not installed, so %s was not validated. Install it "
            "(https://github.com/bids-standard/bids-validator) or pass --skip_bids_validator to "
            "say that this is intended.",
            bids_dir,
        )
        return
    subprocess.run([validator, str(bids_dir)], check=True)


def write_derivatives_dataset_description(output_dir: Path, fastsurfer_version: str) -> None:
    """
    Write a minimal BIDS-derivatives dataset_description.json into output_dir.

    Parameters
    ----------
    output_dir : Path
        Directory to write dataset_description.json into. Created if it does not exist. An
        existing description is left alone, since it may describe a dataset this run is only
        adding subjects to.
    fastsurfer_version : str
        FastSurfer version string to record as GeneratedBy.Version.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    description_file = output_dir / "dataset_description.json"
    if description_file.exists():
        LOGGER.info("%s exists already, keeping it.", description_file)
        return
    description = {
        "Name": "FastSurfer Output",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "FastSurfer", "Version": fastsurfer_version}],
    }
    with open(description_file, "w") as file:
        json.dump(description, file, indent=2)
