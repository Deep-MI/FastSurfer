import os
from pathlib import Path

import pytest

__all__ = ["subjects_dir", "test_dir", "reference_dir", "subjects_list"]


@pytest.fixture
def subjects_dir():
    return Path(os.environ["SUBJECTS_DIR"])


@pytest.fixture
def test_dir():
    return Path(os.environ["TEST_DIR"])


@pytest.fixture
def reference_dir():
    return Path(os.environ["REFERENCE_DIR"])


@pytest.fixture
def subjects_list():
    return Path(os.environ["SUBJECTS_LIST"])
