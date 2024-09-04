import os
import pytest
import yaml
from pathlib import Path

from .common import *

from logging import getLogger

logger = getLogger(__name__)


def load_errors():
    """
    Load the errors and whitelist strings from ./data/logfile.errors.yaml.

    Returns
    -------
    errors : list[str]
        List of errors.
    whitelist : list[str]
        List of whitelisted errors.
    """

    # Open the error_file_path and read the errors and whitelist into arrays

    error_file_path = os.path.join(Path(__file__).parent, "data/logfile.errors.yaml")

    with open(error_file_path, 'r') as file:
        data = yaml.safe_load(file)
        errors = data.get('errors', [])
        whitelist = data.get('whitelist', [])

    return errors, whitelist


def load_log_files(test_subject: str):
    """
    Retrieve the log files in the given log directory.

    Parameters
    ----------
    test_subject : str
        Subject directory to test.

    Returns
    -------
    log_files : list[Path]
        List of log files in the given log directory.
    """

    # Retrieve the log files in given log directory

    log_directory = os.path.join(test_subject, "scripts")
    log_files = [file for file in Path(log_directory).iterdir() if file.suffix == '.log']

    return log_files


@pytest.mark.parametrize("test_subject", load_test_subjects())
def test_errors(subjects_dir, test_dir, test_subject):
    """
    Test if there are any errors in the log files.

    Parameters
    ----------
    subjects_dir : str
        Subjects directory.
    test_dir : str
        Tests directory.
    test_subject : str
        Subject to test.

    Raises
    ------
    AssertionError
        If any of the keywords are in the log files.
    """

    test_subject = os.path.join(subjects_dir, test_dir, test_subject)
    log_files = load_log_files(test_subject)

    error_flag = False

    errors, whitelist = load_errors()

    files_with_errors = {}

    # Check if any of the keywords are in the log files
    for log_file in log_files:
        rel_path = log_file.relative_to(subjects_dir)
        logger.debug(f"Checking file: {rel_path}")
        try:
            with log_file.open('r') as file:
                lines = file.readlines()
                lines_with_errors = []
                for line_number, line in enumerate(lines, start=1):
                    if any(error in line.lower() for error in errors):
                        if not any(white in line.lower() for white in whitelist):
                            # Get two lines before and after the current line
                            context = lines[max(0, line_number - 2):min(len(lines), line_number + 3)]
                            lines_with_errors.append((line_number, context))
                            # print(lines_with_errors)
                            files_with_errors[rel_path] = lines_with_errors
                            error_flag = True
        except FileNotFoundError:
            raise FileNotFoundError(f"Log file not found at path: {log_file}")

    # Print the lines and context with errors for each file
    for file, lines in files_with_errors.items():
        logger.debug(f"\nFile {file}, in line {files_with_errors[file][0][0]}:")
        for line_number, line in lines:
            logger.debug(*line, sep="")

    # Assert that there are no lines with any of the keywords
    assert not error_flag, f"Found errors in the following files: {files_with_errors}"
    logger.debug("\nNo errors found in any log files.")
