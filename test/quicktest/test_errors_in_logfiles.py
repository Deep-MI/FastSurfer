from logging import getLogger
from pathlib import Path

import pytest
import yaml

from .common import SubjectDefinition

logger = getLogger(__name__)

class LogFile:
    """
    Class to handle log files.

    Attributes
    ----------
    path: Path
        The path to the logfile read.
    contents: list[str]
        The content of the logfile.
    errors : list[str]
        List of errors.
    whitelist : list[str]
        List of whitelisted errors.
    """

    path: Path
    contents: list[str]
    errors: list[str]
    whitelist: list[str]

    def __init__(self, logfile: Path):
        """
        Create a LogFile object.

        Parameters
        ----------
        logfile : Path
            The path to the logfile.
        """
        self.path = logfile
        self.contents = list(logfile.read_text().splitlines())

    @property
    def errors(self):
        if not hasattr(self, '_errors'):
            self.__get_config()
        return self._errors

    @property
    def whitelist(self):
        if not hasattr(self, '_whitelist'):
            self.__get_config()
        return self._whitelist

    @classmethod
    def __get_config(cls):
        # Open the error_file_path and read the errors and whitelist into arrays
        error_config_path = Path(__file__).parent / "data/logfile.errors.yaml"
        # Load the errors and whitelist strings from ./data/logfile.errors.yaml.
        with open(error_config_path) as file:
            data = yaml.safe_load(file)
            cls._errors = data.get("errors", [])
            cls._whitelist = data.get("whitelist", [])

    def find_error_lines(self) -> list[int]:
        def check_line(lline: str) -> bool:
            return any(error in lline for error in self.errors) and not any(white in lline for white in self.whitelist)

        return list(i for i, line in enumerate(self.contents) if check_line(line.lower()))


@pytest.fixture(scope="module")
def logfile(request: pytest.FixtureRequest, test_subject: SubjectDefinition) -> LogFile:
    """
    Reads the logfile defined via test_subject and the param.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Request for a filename.
    test_subject : SubjectDefinition
        Definition of the Subject and its directory.

    Returns
    -------
    LogFile
        The LogFile object.
    """
    logfile = test_subject.path / "scripts" / request.param
    assert logfile.exists(), f"{logfile.relative_to(test_subject.path)} does not exist!"
    return LogFile(logfile)


@pytest.mark.parametrize("logfile", ["deep-seg.log", "recon-surf.log"], indirect=True)
def test_errors_in_logfiles(logfile: LogFile, subjects_dir: Path):
    """
    Test if there are any errors in the log files.

    Parameters
    ----------
    logfile : LogFile
        A LogFile object to get the log contents from.

    Raises
    ------
    AssertionError
        If any of the keywords are in the log files.
    """
    # Check if any of the keywords are in the log files
    lines_with_errors = logfile.find_error_lines()

    rel_path = logfile.path.relative_to(subjects_dir)
    if len(lines_with_errors) > 0:
        import numpy as np

        def fmt_line(index):
            error_line = index in lines_with_errors
            fill0 = "*" if error_line else " "
            fill1 = "**: " if error_line else ":   "
            return f"{index + 1:{fill0}>4d}{fill1}{logfile.contents[index]}"

        # Print the lines and context with errors for each file
        logger.debug(f"Errors found in file {rel_path}:")
        lines_to_report = np.unique((np.asarray(lines_with_errors)[None] + np.arange(-2, 3)[:, None]).flat)
        lines_to_report = lines_to_report[lines_to_report > -1]
        lines_diff_highlighted = list(map(fmt_line, lines_to_report))
        lines_diff_same = list(map(fmt_line, (line for line in lines_to_report if line not in lines_with_errors)))
        logger.debug("\n".join(lines_diff_highlighted))
        # Assert that there are no lines with any of the keywords
        assert lines_diff_same == lines_diff_highlighted, f"Found {len(lines_with_errors)} errors in {rel_path}!"
    else:
        logger.debug(f"No errors found in log file {rel_path}.")
