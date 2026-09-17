"""
conftest is the configuration file for tests on this level.

It defines test-wide fixtures
"""


import os
from pathlib import Path

import pytest

from .common import SubjectDefinition, chain_position, chain_stage

__all__ = [
    "pytest_addoption",
    "pytest_runtest_makereport",
    "pytest_terminal_summary",
    "ref_subject",
    "ref_subjects",
    "reference_dir",
    "subjects_dir",
]

# the files a comparison is parametrised over, in the order the parameter names appear
_COMPARED_FILE_PARAMS = ("segmentation_image", "intensity_image", "image", "surface")

env: dict[str, Path] = {}
# Checking environment variables
for required_env_variable in ["REF_DIR", "SUBJECTS_DIR"]:
    assert required_env_variable in os.environ, f"Required environment variable {required_env_variable} is not set!"
    env[required_env_variable] = Path(os.environ[required_env_variable])


# Note, reference_dir should not be a fixture, because test_subject is tied to the
# reference dir and  pytest does not allow the generation of marks depending on fixtures
# i.e. parametrization of a fixture depending on a fixture is not possible!
"""
Folder with reference data (defined in environment variable).
"""
reference_dir: Path = env["REF_DIR"]
__subjects = (p for p in reference_dir.iterdir() if p.is_dir() and p.name not in ("logs", "slurm"))
__max_subjects = int(os.environ.get("MAX_SUBJECTS", -1))
"""
Load the test subjects from the reference path (one subject per folder).
"""
ref_subjects: list[Path] = [p for i, p in enumerate(__subjects) if i < __max_subjects or __max_subjects < 0]

assert len(ref_subjects) > 0, "No test subjects found!"


@pytest.fixture(scope="session")
def subjects_dir() -> Path:
    return env["SUBJECTS_DIR"]


@pytest.fixture(scope="session", params=ref_subjects, ids=lambda s: s.name)
def ref_subject(request: pytest.FixtureRequest) -> SubjectDefinition:
    """
    The reference subjects from the reference path.

    Returns
    =======
    SubjectDefinition
        Subject name and path.
    """
    return SubjectDefinition(request.param)


# derived fixtures
@pytest.fixture(scope="session")
def test_subject(ref_subject: SubjectDefinition, subjects_dir: Path) -> SubjectDefinition:
    return ref_subject.with_subjects_dir(subjects_dir)


def _divergences(config: pytest.Config) -> set[tuple[int, str]]:
    """The pipeline outputs whose comparison failed, as (chain position, filename)."""
    store = getattr(config, "_pipelinetest_divergences", None)
    if store is None:
        store = set()
        config._pipelinetest_divergences = store
    return store


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Record which pipeline outputs failed a comparison, for the summary below."""
    outcome = yield
    report = outcome.get_result()
    callspec = getattr(item, "callspec", None)
    if report.when != "call" or not report.failed or callspec is None:
        return
    for name in _COMPARED_FILE_PARAMS:
        filename = callspec.params.get(name)
        if isinstance(filename, str):
            _divergences(item.config).add((chain_position(filename), filename))
            break


def pytest_terminal_summary(terminalreporter, exitstatus, config: pytest.Config):
    """
    Name the first pipeline stage that differs.

    Each stage reads the one before it, so a single upstream change shows up as a wall of failures.
    Only the earliest is a finding; the rest are its consequences.
    """
    divergences = _divergences(config)
    if not divergences:
        return
    position, filename = min(divergences)
    stage = chain_stage(filename)
    where = f"{filename}, stage '{stage}'" if stage else f"{filename}, which data/chain.yaml does not list"
    terminalreporter.write_sep("=", "first divergence", yellow=True)
    terminalreporter.write_line(f"Earliest pipeline output that differs: {where}")
    later = sum(1 for pos, _ in divergences if pos > position)
    if later:
        terminalreporter.write_line(
            f"{later} later output(s) also differ and are likely downstream of it, so explain this one first."
        )


def pytest_addoption(parser):
    # the following options is for are for test_images and test_stats only
    parser.addoption(
        "--collect_csv",
        action="store",
        default=None,
        type=Path,
        help="Directory to store csv files that will collect all differences between reference and test.",
    )
