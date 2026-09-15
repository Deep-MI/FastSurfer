import json
from pathlib import Path

import pytest

pytest.importorskip("bids", reason="pybids (fastsurfer[bids] extra) is not installed")

from FastSurferCNN.utils import bids  # noqa: E402


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


@pytest.fixture
def bids_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "bids"
    _touch(root / "dataset_description.json")
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Test Dataset", "BIDSVersion": "1.8.0"})
    )

    # sub-01: single session (no ses- level), T1w only
    _touch(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz")

    # sub-02: two sessions -> longitudinal; ses-01 has a T2w, ses-02 does not
    _touch(root / "sub-02" / "ses-01" / "anat" / "sub-02_ses-01_T1w.nii.gz")
    _touch(root / "sub-02" / "ses-01" / "anat" / "sub-02_ses-01_T2w.nii.gz")
    _touch(root / "sub-02" / "ses-02" / "anat" / "sub-02_ses-02_T1w.nii.gz")

    # sub-03: only a T2w, no T1w anywhere -> must be skipped
    _touch(root / "sub-03" / "anat" / "sub-03_T2w.nii.gz")

    return root


def test_find_subjects_basic_discovery(bids_dataset: Path):
    subjects = bids.find_subjects(bids_dataset, validate=False)
    by_id = {s.subject_id: s for s in subjects}

    # sub-03 has no T1w and must be excluded
    assert set(by_id) == {"sub-01", "sub-02"}

    sub01 = by_id["sub-01"]
    assert not sub01.is_longitudinal
    assert len(sub01.sessions) == 1
    assert sub01.sessions[0].session_id is None
    assert sub01.output_id(sub01.sessions[0]) == "sub-01"
    assert sub01.sessions[0].t1w.name == "sub-01_T1w.nii.gz"
    assert sub01.sessions[0].t2w is None

    sub02 = by_id["sub-02"]
    assert sub02.is_longitudinal
    assert len(sub02.sessions) == 2
    sessions_by_id = {s.session_id: s for s in sub02.sessions}
    assert set(sessions_by_id) == {"01", "02"}
    assert sub02.output_id(sessions_by_id["01"]) == "sub-02_ses-01"
    assert sessions_by_id["01"].t2w is not None
    assert sessions_by_id["02"].t2w is None


def test_find_subjects_participant_filter(bids_dataset: Path):
    subjects = bids.find_subjects(bids_dataset, participant_labels=["01"], validate=False)
    assert [s.subject_id for s in subjects] == ["sub-01"]

    # accepts labels with or without the sub- prefix
    subjects = bids.find_subjects(bids_dataset, participant_labels=["sub-01"], validate=False)
    assert [s.subject_id for s in subjects] == ["sub-01"]


def test_find_subjects_unknown_participant_raises(bids_dataset: Path):
    with pytest.raises(ValueError, match="99"):
        bids.find_subjects(bids_dataset, participant_labels=["99"], validate=False)


def test_find_subjects_session_filter(bids_dataset: Path):
    subjects = bids.find_subjects(
        bids_dataset, participant_labels=["02"], session_labels=["01"], validate=False,
    )
    assert len(subjects) == 1
    assert not subjects[0].is_longitudinal
    assert subjects[0].sessions[0].session_id == "01"


def test_write_derivatives_dataset_description(tmp_path: Path):
    out_dir = tmp_path / "output"
    bids.write_derivatives_dataset_description(out_dir, "2.6.0-dev0")

    description_file = out_dir / "dataset_description.json"
    assert description_file.exists()
    description = json.loads(description_file.read_text())
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"][0]["Name"] == "FastSurfer"
    assert description["GeneratedBy"][0]["Version"] == "2.6.0-dev0"
