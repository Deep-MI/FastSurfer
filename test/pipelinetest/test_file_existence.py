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
def expected_patterns() -> list[str]:
    """Globs, for the outputs whose name depends on the input, see the yaml."""
    with open(Path(__file__).parent / "data/expected-files.yaml") as fp:
        return list(yaml.safe_load(fp).get("file_patterns", []))


def test_file_existence(
        test_subject: SubjectDefinition,
        expected_files: set[str],
        expected_patterns: list[str],
):
    """
    Test the existence of files for the subject test_subject.

    Parameters
    ----------
    test_subject : SubjectDefinition
        Definition of the test subject.
    expected_files : set of str
        The set of files expected to be present in the subject.
    expected_patterns : list of str
        Globs that each have to match at least one file in the subject.

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

    # condition again, so a directory or a touch file cannot satisfy a pattern that asks for output
    unmatched = [
        pattern for pattern in expected_patterns
        if not any(condition(match) for match in Path(test_subject.path).glob(pattern))
    ]
    assert unmatched == [], f"Patterns {tuple(unmatched)} match no file in test subject."

    logger.debug("All files present.")
