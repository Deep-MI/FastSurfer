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

"""Discovery and grouping of subjects/sessions from a BIDS dataset for run_fastsurfer_bids.py."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass
class BidsSession:
    """A single anat session found for a subject (may be the subject's only session)."""

    session_id: str | None
    t1w: Path
    t2w: Path | None = None


@dataclass
class BidsSubject:
    """A subject with one or more sessions, plus its processing mode."""

    subject_id: str
    sessions: list[BidsSession] = field(default_factory=list)

    @property
    def is_longitudinal(self) -> bool:
        """Return True if this subject has more than one session with a T1w image."""
        return len(self.sessions) > 1

    def output_id(self, session: BidsSession) -> str:
        """Return the FastSurfer subject/timepoint id for a given session of this subject."""
        if session.session_id is None:
            return self.subject_id
        return f"{self.subject_id}_ses-{session.session_id}"


def _require_pybids():
    try:
        import bids as pybids
    except ImportError as e:
        raise ImportError(
            "The 'bids' (pybids) package is required for BIDS support but is not installed. "
            "Install it with `pip install fastsurfer[bids]` or `pip install pybids`."
        ) from e
    return pybids


def find_subjects(
    bids_dir: Path,
    participant_labels: list[str] | None = None,
    session_labels: list[str] | None = None,
    validate: bool = True,
) -> list[BidsSubject]:
    """
    Discover subjects, sessions, and their T1w/T2w images in a BIDS dataset.

    Parameters
    ----------
    bids_dir : Path
        Path to the root of a BIDS-valid dataset.
    participant_labels : list[str], optional
        If given, only include these subject labels (without the 'sub-' prefix).
    session_labels : list[str], optional
        If given, only include these session labels (without the 'ses-' prefix).
    validate : bool, default=True
        Whether pybids should validate the dataset against the BIDS spec.

    Returns
    -------
    list[BidsSubject]
        Subjects found, each with the sessions that have a usable T1w image. Subjects
        with no T1w anywhere in the dataset are omitted (with a warning).
    """
    pybids = _require_pybids()
    layout = pybids.BIDSLayout(str(bids_dir), validate=validate)

    subject_ids = layout.get_subjects()
    if participant_labels:
        wanted = {label.removeprefix("sub-") for label in participant_labels}
        missing = wanted - set(subject_ids)
        if missing:
            raise ValueError(f"Requested participant label(s) not found in {bids_dir}: {sorted(missing)}")
        subject_ids = [s for s in subject_ids if s in wanted]

    subjects = []
    for subject_id in subject_ids:
        session_ids = layout.get_sessions(subject=subject_id)
        if session_labels:
            wanted_ses = {label.removeprefix("ses-") for label in session_labels}
            session_ids = [s for s in session_ids if s in wanted_ses]
        # a dataset without a session level yields session_ids == [], represent as [None]
        iter_sessions: list[str | None] = list(session_ids) if session_ids else [None]

        sessions = []
        for session_id in iter_sessions:
            query = {"subject": subject_id, "suffix": "T1w", "extension": [".nii", ".nii.gz"]}
            if session_id is not None:
                query["session"] = session_id
            t1w_files = layout.get(**query)
            if not t1w_files:
                continue
            if len(t1w_files) > 1:
                LOGGER.warning(
                    "Subject sub-%s%s has multiple T1w images, using the first: %s",
                    subject_id,
                    f" session ses-{session_id}" if session_id else "",
                    [f.path for f in t1w_files],
                )
            t1w_path = Path(t1w_files[0].path)

            t2w_query = dict(query, suffix="T2w")
            t2w_files = layout.get(**t2w_query)
            t2w_path = Path(t2w_files[0].path) if t2w_files else None

            sessions.append(BidsSession(session_id=session_id, t1w=t1w_path, t2w=t2w_path))

        if not sessions:
            LOGGER.warning("Subject sub-%s has no T1w image, skipping.", subject_id)
            continue
        subjects.append(BidsSubject(subject_id=f"sub-{subject_id}", sessions=sessions))

    return subjects


def write_derivatives_dataset_description(output_dir: Path, fastsurfer_version: str) -> None:
    """
    Write a minimal BIDS-derivatives dataset_description.json into output_dir.

    Parameters
    ----------
    output_dir : Path
        Directory to write dataset_description.json into. Created if it does not exist.
    fastsurfer_version : str
        FastSurfer version string to record as GeneratedBy.Version.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    description = {
        "Name": "FastSurfer Output",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "FastSurfer", "Version": fastsurfer_version}],
    }
    with open(output_dir / "dataset_description.json", "w") as f:
        json.dump(description, f, indent=2)
