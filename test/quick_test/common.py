import os

from logging import getLogger

logger = getLogger(__name__)


def load_test_subjects():
    """
    Load the test files from the given file path.

    Returns:
        test_subjects (list): List of subjects to test subjects.
    """

    subjects_dir = os.environ["SUBJECTS_DIR"]
    subjects_list = os.environ["SUBJECTS_LIST"]

    test_subjects = []

    # Load the reference and test files
    with open(os.path.join(subjects_dir, subjects_list), 'r') as file:
        for line in file:
            filename = line.strip()
            logger.debug(filename)
            # test_file = os.path.join(subjects_dir, filename)
            test_subjects.append(filename)

    return test_subjects
