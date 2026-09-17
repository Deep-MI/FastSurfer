"""
conftest is the configuration file for tests on this level.

It defines test-wide fixtures
"""


import os
from pathlib import Path

import pytest

from .common import SubjectDefinition, chain_order, chain_stage, record_failure, recorded_files

__all__ = [
    "pytest_addoption",
    "pytest_runtest_makereport",
    "pytest_terminal_summary",
    "ref_subject",
    "ref_subjects",
    "reference_dir",
    "subjects_dir",
]

# The argnames a comparison is parametrised over, so a failure can be attributed to a file.
# A new comparison module has to be listed here: the hook has no other way to tell which of a
# test's parameters names the file, and a module that is missing contributes nothing to the
# summary below, silently, which would make it name a later file as the first divergence.
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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """Record which pipeline outputs failed a comparison, for the summary below."""
    outcome = yield
    report = outcome.get_result()
    callspec = getattr(item, "callspec", None)
    if report.when != "call" or not report.failed or callspec is None:
        return
    subject = callspec.params.get("ref_subject")
    subject = getattr(subject, "name", None) or str(subject)
    for name in _COMPARED_FILE_PARAMS:
        filename = callspec.params.get(name)
        if isinstance(filename, str):
            record_failure(item.config, subject, filename)
            break


def _earliest(files: set[str]) -> str:
    """The listed file that the pipeline writes first, described for the summary."""
    filename = min(files, key=chain_order)
    stage = chain_stage(filename)
    return f"{filename}, stage '{stage}'" if stage else f"{filename}, which data/chain.yaml does not list"


def pytest_terminal_summary(terminalreporter, exitstatus, config: pytest.Config):
    """
    Name where the outputs start to diverge, by the two questions that have different answers.

    Each stage reads the one before it, so one upstream change shows up as a wall of failures
    downstream of it. Which stage changed and which comparisons failed are not the same thing: a
    difference small enough to pass every tolerance still marks the stage it entered at, and is
    usually the one worth explaining.

    Reported per subject, because two subjects in one session diverge in different places.
    """
    differing = recorded_files(config, failed=False)
    failed = recorded_files(config, failed=True)
    if not differing and not failed:
        return
    terminalreporter.write_sep("=", "where the outputs diverge", yellow=True)
    for subject in sorted({s for s, _ in differing | failed}):
        differ_here = {f for s, f in differing if s == subject}
        failed_here = {f for s, f in failed if s == subject}
        terminalreporter.write_line(f"{subject}:")
        if differ_here:
            terminalreporter.write_line(
                f"  First output that differs from the reference at all: {_earliest(differ_here)}"
            )
            terminalreporter.write_line(
                f"    {len(differ_here)} output(s) differ, of which {len(differ_here & failed_here)} "
                f"also failed a check."
            )
        if failed_here:
            # "failed a check" rather than "exceeds its tolerance": this is recorded for any failing
            # comparison, which includes a header mismatch or a dtype assert, not only a tolerance
            terminalreporter.write_line(f"  First output that failed a check: {_earliest(failed_here)}")
        if differ_here and failed_here and chain_order(min(differ_here, key=chain_order)) < chain_order(
            min(failed_here, key=chain_order)
        ):
            terminalreporter.write_line(
                "  The change enters earlier than the first failure and passes the checks in "
                "between, so explain the earlier one."
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
