from logging import getLogger
from pathlib import Path

import pytest
import yaml

from .common import SubjectDefinition

logger = getLogger(__name__)


@pytest.fixture(scope="session")
def expected_files() -> set[str]:
    with open(Path(__file__).parent / "data/expected-files.yaml") as fp:
        return set(yaml.safe_load(fp)["files"])


@pytest.fixture(scope="session")
def files_for_test_subject(test_subject: SubjectDefinition) -> set[str]:
    """
    Get the list of files in the directory relative to the folder path.

    Parameters
    ----------
    test_subject : SubjectDefinition
        Folder definition of a subject.

    Returns
    -------
    set
        A list of all files in the subject directory.
    """
    # Get a list of all files in the folder recursively
    all_files_in_subject_dir = Path(test_subject.path).rglob("*")
    return set(str(file.relative_to(test_subject.path)) for file in all_files_in_subject_dir if file.is_file())


def test_file_existence(files_for_test_subject: set[str], expected_files: set[str]):
    """
    Test the existence of files for the subject test_subject.

    Parameters
    ----------
    files_for_test_subject : set of str
        The set of files to present in the subject test_subject.
    expected_files : set of str
        The set of files expected to be present in the subject.

    Raises
    ------
    AssertionError
        If a file in the reference list does not exist in the test list.
    """

    # Check if each file in the reference list exists in the test list
    missing_files = expected_files - files_for_test_subject
    assert expected_files <= files_for_test_subject, f"Files {tuple(missing_files)} do not exist in test subject."

    logger.debug("All files present.")
