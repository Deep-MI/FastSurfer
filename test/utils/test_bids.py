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
Check what run_fastsurfer_bids.py discovers in a BIDS dataset and what it would run.

Discovery is a glob over the BIDS directory layout, so the interesting cases are the ones where
that layout varies: a dataset with no session level, a session with a T2w next to the T1w, and a
subject with no T1w at all, which has to be dropped rather than handed to FastSurfer. The routing
half is checked through --dry, which prints the subject list and the command without running
anything, so it needs no data and no FreeSurfer.
"""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from FastSurferCNN.utils import bids

FASTSURFER_HOME = Path(__file__).parent.parent.parent
sys.path.insert(0, str(FASTSURFER_HOME))  # run_fastsurfer_bids.py is a script, not part of the package

import run_fastsurfer_bids  # noqa: E402


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


@pytest.fixture
def bids_dataset(tmp_path: Path) -> Path:
    """Build a minimal dataset covering the layout variants discovery has to handle."""
    root = tmp_path / "bids"
    _touch(root / "dataset_description.json")
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "Test Dataset", "BIDSVersion": "1.8.0"})
    )
    # sub-01: no session level at all
    _touch(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    # sub-02: two sessions, only the first with a T2w
    _touch(root / "sub-02" / "ses-01" / "anat" / "sub-02_ses-01_T1w.nii.gz")
    _touch(root / "sub-02" / "ses-01" / "anat" / "sub-02_ses-01_T2w.nii.gz")
    _touch(root / "sub-02" / "ses-02" / "anat" / "sub-02_ses-02_T1w.nii.gz")
    # sub-03: a T2w but no T1w, which FastSurfer cannot process
    _touch(root / "sub-03" / "anat" / "sub-03_T2w.nii.gz")
    return root


def test_every_session_with_a_t1w_is_found(bids_dataset: Path) -> None:
    """Each session is its own case, and the one without a T1w is not."""
    sessions = bids.find_sessions(bids_dataset)
    assert [s.output_id for s in sessions] == ["sub-01", "sub-02_ses-01", "sub-02_ses-02"]

    by_id = {s.output_id: s for s in sessions}
    assert by_id["sub-01"].session_id is None
    assert by_id["sub-01"].t1w.name == "sub-01_T1w.nii.gz"


def test_a_t2w_is_only_picked_up_when_it_is_asked_for(bids_dataset: Path) -> None:
    """Using a T2 changes what HypVINN computes, so a file being there must not decide it."""
    without = {s.output_id: s.t2w for s in bids.find_sessions(bids_dataset)}
    assert without["sub-02_ses-01"] is None

    with_t2 = {s.output_id: s.t2w for s in bids.find_sessions(bids_dataset, with_t2=True)}
    assert with_t2["sub-02_ses-01"].name == "sub-02_ses-01_T2w.nii.gz"
    assert with_t2["sub-02_ses-02"] is None


def test_several_t1w_images_in_one_session_is_an_error(tmp_path: Path) -> None:
    """Which acquisition to process is a statement about the data, not a sort order."""
    anat = tmp_path / "bids" / "sub-01" / "anat"
    _touch(anat / "sub-01_run-1_T1w.nii.gz")
    _touch(anat / "sub-01_run-2_T1w.nii.gz")
    with pytest.raises(ValueError, match="ambiguous"):
        bids.find_sessions(tmp_path / "bids")


def test_a_json_sidecar_is_not_mistaken_for_an_image(tmp_path: Path) -> None:
    """The glob is on the suffix, so the sidecar sitting beside the image must not win it."""
    anat = tmp_path / "bids" / "sub-01" / "anat"
    _touch(anat / "sub-01_T1w.json")
    _touch(anat / "sub-01_T1w.nii.gz")
    sessions = bids.find_sessions(tmp_path / "bids")
    assert [s.t1w.name for s in sessions] == ["sub-01_T1w.nii.gz"]


@pytest.mark.parametrize("label", ["01", "sub-01"])
def test_participant_label_works_with_and_without_the_prefix(bids_dataset: Path, label: str) -> None:
    """Both spellings are in use, and the BIDS-App contract does not say which."""
    sessions = bids.find_sessions(bids_dataset, participant_labels=[label])
    assert [s.output_id for s in sessions] == ["sub-01"]


def test_an_unknown_participant_label_is_an_error(bids_dataset: Path) -> None:
    """A typo has to stop the run, not quietly process a different set of subjects."""
    with pytest.raises(ValueError, match="99"):
        bids.find_sessions(bids_dataset, participant_labels=["99"])


def test_session_label_selects_one_session(bids_dataset: Path) -> None:
    """Selecting a session keeps that session, and drops the subject that has no session level.

    sub-01 has its anat directly under the subject, so no session of it can be the requested one.
    Processing it anyway would ignore the filter the user gave, which is how a run ends up holding
    more than was asked for without saying so.
    """
    sessions = bids.find_sessions(bids_dataset, session_labels=["ses-02"])
    assert [s.output_id for s in sessions] == ["sub-02_ses-02"]


def test_an_unknown_session_label_is_an_error(bids_dataset: Path) -> None:
    """As for participants: a typo has to stop the run rather than quietly process nothing."""
    with pytest.raises(ValueError, match="ses-99"):
        bids.find_sessions(bids_dataset, session_labels=["99"])


def test_a_path_with_a_space_is_quoted_for_the_subject_list(tmp_path: Path) -> None:
    """brun_fastsurfer.sh tokenizes the line shell-style, so an unquoted space splits the path.

    BIDS labels cannot hold a space, but the directory the dataset sits in can, and the result is
    a --t1 pointing at a prefix of the real name plus a stray argument.
    """
    root = tmp_path / "a study" / "bids"
    _touch(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    _touch(root / "sub-01" / "anat" / "sub-01_T2w.nii.gz")
    sessions = bids.find_sessions(root, with_t2=True)
    (line,) = run_fastsurfer_bids.subject_list_lines(sessions)

    subject_id, _, image_parameters = line.partition("=")
    assert subject_id == "sub-01"
    # what brun_fastsurfer.sh does with the rest of the line
    assert shlex.split(image_parameters) == [
        str(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz"),
        "--t2",
        str(root / "sub-01" / "anat" / "sub-01_T2w.nii.gz"),
    ]


def test_a_mistyped_label_is_an_error_line_not_a_traceback(bids_dataset: Path, tmp_path: Path) -> None:
    """The user mistyped something; a stack trace tells them nothing about which flag it was."""
    result = subprocess.run(
        [sys.executable, str(FASTSURFER_HOME / "run_fastsurfer_bids.py"),
         str(bids_dataset), str(tmp_path / "out"), "participant",
         "--skip_bids_validator", "--dry", "--participant_label", "99"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "sub-99" in result.stderr


def test_a_missing_bids_dir_is_an_error(tmp_path: Path) -> None:
    """An empty glob would otherwise read as a dataset that holds no T1w at all."""
    result = subprocess.run(
        [sys.executable, str(FASTSURFER_HOME / "run_fastsurfer_bids.py"),
         str(tmp_path / "nope"), str(tmp_path / "out"), "participant", "--dry"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "not a directory" in result.stderr


def test_derivatives_description_records_the_version_and_the_model(tmp_path: Path) -> None:
    """Downstream tooling reads this to learn what produced the outputs, and by which model."""
    bids.write_derivatives_dataset_description(tmp_path / "out", "2.6.0-dev0")
    description = json.loads((tmp_path / "out" / "dataset_description.json").read_text())
    assert description["DatasetType"] == "derivative"
    assert description["GeneratedBy"] == [
        {"Name": "FastSurfer", "Version": "2.6.0-dev0", "Description": bids.CROSS_SECTIONAL}
    ]
    assert bids.read_processing_mode(tmp_path / "out") == bids.CROSS_SECTIONAL


def _dry_run(bids_dataset: Path, tmp_path: Path, *extra: str) -> str:
    """Run the entrypoint with --dry and return what it printed."""
    result = subprocess.run(
        [sys.executable, str(FASTSURFER_HOME / "run_fastsurfer_bids.py"),
         str(bids_dataset), str(tmp_path / "out"), "participant",
         "--skip_bids_validator", "--dry", *extra],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def test_dry_run_routes_every_session_to_brun(bids_dataset: Path, tmp_path: Path) -> None:
    """One subject list, one brun call, and nothing written."""
    stdout = _dry_run(bids_dataset, tmp_path)
    assert "sub-01=" in stdout and "sub-02_ses-01=" in stdout and "sub-02_ses-02=" in stdout
    assert "--t2 " not in stdout  # not without --use_t2
    assert "--t2 " in _dry_run(bids_dataset, tmp_path, "--use_t2")
    assert "brun_fastsurfer.sh" in stdout
    assert not (tmp_path / "out").exists(), "--dry wrote to the output directory"


def test_dry_run_with_slurm_routes_to_srun(bids_dataset: Path, tmp_path: Path) -> None:
    """--slurm swaps the script, and adds the --data srun rewrites the listed paths against."""
    stdout = _dry_run(bids_dataset, tmp_path, "--slurm")
    assert "srun_fastsurfer.sh" in stdout
    assert "brun_fastsurfer.sh" not in stdout
    assert f"--data {bids_dataset}" in stdout


def test_passthrough_options_reach_the_batch_script(bids_dataset: Path, tmp_path: Path) -> None:
    """Everything after -- is handed on unchanged, which is how every other option is supported."""
    stdout = _dry_run(bids_dataset, tmp_path, "--", "--seg_only", "--3T")
    assert "--seg_only --3T" in stdout


def test_a_passthrough_option_this_script_sets_is_refused(bids_dataset: Path, tmp_path: Path) -> None:
    """A second --sd would decide the output directory, silently overriding the positional one."""
    result = subprocess.run(
        [sys.executable, str(FASTSURFER_HOME / "run_fastsurfer_bids.py"),
         str(bids_dataset), str(tmp_path / "out"), "participant",
         "--skip_bids_validator", "--dry", "--", "--sd", "/somewhere/else"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "--sd" in result.stderr


def test_a_passthrough_t2_is_refused(bids_dataset: Path, tmp_path: Path) -> None:
    """One T2 for every session is some other session's image, whichever way --use_t2 is set.

    brun_fastsurfer.sh puts the passthrough options before each line's own, so a --t2 after the --
    reaches every case: with --use_t2 those that have no T2w of their own, without it all of them.
    """
    def run(*extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(FASTSURFER_HOME / "run_fastsurfer_bids.py"),
             str(bids_dataset), str(tmp_path / "out"), "participant",
             "--skip_bids_validator", "--dry", *extra, "--", "--t2", "/elsewhere/t2.nii.gz"],
            capture_output=True, text=True,
        )
    for result in (run(), run("--use_t2")):
        assert result.returncode == 1
        assert "--t2" in result.stderr


def test_a_directory_holding_the_other_model_is_refused(bids_dataset: Path, tmp_path: Path) -> None:
    """A timepoint and a session claim one directory name, so the two models cannot be mixed."""
    output_dir = tmp_path / "out"
    bids.write_derivatives_dataset_description(output_dir, "2.6.0-dev0", bids.LONGITUDINAL)
    result = subprocess.run(
        [sys.executable, str(FASTSURFER_HOME / "run_fastsurfer_bids.py"),
         str(bids_dataset), str(output_dir), "participant", "--skip_bids_validator", "--dry"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert bids.LONGITUDINAL in result.stderr


def test_an_existing_dataset_description_is_kept(tmp_path: Path) -> None:
    """output_dir may be a dataset this run only adds subjects to, so its description stands."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "dataset_description.json").write_text(json.dumps({"Name": "Mine"}))
    bids.write_derivatives_dataset_description(output_dir, "2.6.0-dev0")
    description = json.loads((output_dir / "dataset_description.json").read_text())
    assert description == {"Name": "Mine"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
