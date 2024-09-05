import pytest

from .common import *

from logging import getLogger

from pathlib import Path

logger = getLogger(__name__)


def get_files_from_folder(folder_path: Path):
    """
    Get the list of files in the directory relative to the folder path.

    Parameters
    ----------
    folder_path : Path
        Path to the folder.

    Returns
    -------
    list
        List of files in the directory.
    """

    # Get a list of all files in the folder recursively
    filenames = []
    for file in Path(folder_path).rglob("*"):
        filenames.append(str(file.relative_to(folder_path)))

    return filenames


@pytest.mark.parametrize("test_subject", load_test_subjects())
def test_file_existence(subjects_dir: Path, test_dir: Path, reference_dir: Path, test_subject: Path):
    """
    Test the existence of files in the folder.

    Parameters
    ----------
    subjects_dir : Path
        Path to the subjects directory.
    test_dir : Path
        Name of the test directory.
    reference_dir : Path
        Name of the reference directory.
    test_subject : Path
        Name of the test subject.

    Raises
    ------
    AssertionError
        If a file in the reference list does not exist in the test list.
    """

    print(test_subject)

    # Get reference files from the reference subject directory
    reference_subject = subjects_dir / reference_dir / test_subject
    reference_files = get_files_from_folder(reference_subject)

    # Get test list of files in the test subject directory
    test_subject = subjects_dir / test_dir / test_subject
    test_files = get_files_from_folder(test_subject)

    # Check if each file in the reference list exists in the test list
    missing_files = [file for file in reference_files if file not in test_files]
    assert not missing_files, f"Files '{missing_files}' do not exist in test subject."

    logger.debug("\nAll files present.")
