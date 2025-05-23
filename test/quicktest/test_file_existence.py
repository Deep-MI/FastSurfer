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


def test_file_existence(test_subject: SubjectDefinition, expected_files: set[str]):
    """
    Test the existence of files for the subject test_subject.

    Parameters
    ----------
    test_subject : SubjectDefinition
        Definition of the test subject.
    expected_files : set of str
        The set of files expected to be present in the subject.

    Raises
    ------
    AssertionError
        If a file in the reference list does not exist in the test list.
    """
    def condition(file: Path) -> bool:
        return file.is_file() and file.suffix != ".touch" and file.parent.name != "touch"

    # Get a list of all files in the folder recursively
    all_files = Path(test_subject.path).rglob("*")
    files_for_test_subject = set(str(file.relative_to(test_subject.path)) for file in all_files if condition(file))

    # Check if each file in the reference list exists in the test list
    missing_files = expected_files - files_for_test_subject
    assert files_for_test_subject >= expected_files, f"Files {tuple(missing_files)} do not exist in test subject."

    logger.debug("All files present.")
