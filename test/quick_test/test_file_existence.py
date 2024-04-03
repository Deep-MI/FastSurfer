import os
import pytest
import yaml

from .common import *

from logging import getLogger

logger = getLogger(__name__)


def get_files_from_yaml(file_path: str):
    """
    Get the list of files from the YAML file.

    Parameters
    ----------
    file_path : str
        Path to the YAML file.

    Returns
    -------
    list
        List of files specified in the YAML file.
    """

    # Open the file_path and read the files into an array
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        files = data.get('files', [])

    return files


def get_files_from_folder(folder_path: str):
    """
    Get the list of files in the directory relative to the folder path.

    Parameters
    ----------
    folder_path : str
        Path to the folder.

    Returns
    -------
    list
        List of files in the directory.
    """

    # Get a list of all files in the folder recursively
    filenames = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            filenames.append(os.path.relpath(os.path.join(root, file), folder_path))

    return filenames


@pytest.mark.parametrize("test_subject", load_test_subjects())
def test_file_existence(subjects_dir, test_dir, reference_dir, test_subject):
    """
    Test the existence of files in the folder.

    Parameters
    ----------
    subjects_dir : str
        Path to the subjects directory.
    test_dir : str
        Name of the test directory.
    reference_dir : str
        Name of the reference directory.
    test_subject : str
        Name of the test subject.

    Raises
    ------
    AssertionError
        If a file in the reference list does not exist in the test list.
    """

    print(test_subject)

    # Get reference files from the reference subject directory
    reference_subject = os.path.join(subjects_dir, reference_dir, test_subject)
    reference_files = get_files_from_folder(reference_subject)

    # Get test list of files in the test subject directory
    test_subject = os.path.join(subjects_dir, test_dir, test_subject)
    test_files = get_files_from_folder(test_subject)

    # Check if each file in the reference list exists in the test list
    for file in reference_files:
        assert file in test_files, f"File '{file}' does not exist."

    logger.debug("\nAll files present.")
