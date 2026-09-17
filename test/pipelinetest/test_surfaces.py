"""
Compare the surfaces of a processed subject against the reference.

Until this module existed the surfaces were only checked for existence, so a change in the
tessellation, the topology fix or the surface placement passed silently unless it happened to move
a volume as well. Their tolerances live in data/surface.geometry.yaml.
"""

from functools import lru_cache
from logging import getLogger
from pathlib import Path

import numpy as np
import pytest
import yaml

from .common import (
    SubjectDefinition,
    chain_order,
    chain_stage,
    record_difference,
    skip_if_missing,
    write_table_file,
)

logger = getLogger(__name__)


@lru_cache
def read_surface_tolerances() -> dict[str, float]:
    with open(Path(__file__).parent / "data/surface.geometry.yaml") as fp:
        return yaml.safe_load(fp)["tolerances"]


def test_surface_geometry(
        test_subject: SubjectDefinition,
        ref_subject: SubjectDefinition,
        surface: str,
        pytestconfig: pytest.Config,
):
    """
    Compare a surface against the reference, by vertex count, face count and coordinates.

    Parameters
    ----------
    test_subject : SubjectDefinition
        Definition of the test subject.
    ref_subject : SubjectDefinition
        Definition of the reference subject.
    surface : str
        Name of the surface file under surf/.
    pytestconfig : pytest.Config
        The sessions config object.

    Raises
    ------
    AssertionError
        If the counts differ, or the coordinates differ by more than the tolerance.
    """
    skip_if_missing(ref_subject, test_subject, surface, surface=True)

    _, test_coords, test_faces = test_subject.load_surface(surface)
    _, reference_coords, reference_faces = ref_subject.load_surface(surface)

    def same(a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape == b.shape and np.array_equal(a, b)

    if not same(test_coords, reference_coords) or not same(test_faces, reference_faces):
        record_difference(pytestconfig, surface)

    stage = chain_stage(surface)
    counts = (
        f"vertices {test_coords.shape[0]} against {reference_coords.shape[0]}, "
        f"faces {test_faces.shape[0]} against {reference_faces.shape[0]}"
    )
    assert test_coords.shape == reference_coords.shape, (
        f"{surface} has a different vertex count, so the surfaces cannot be compared vertex by "
        f"vertex: {counts}. First divergence is at or before '{stage}'."
    )
    assert test_faces.shape == reference_faces.shape, f"{surface} has a different face count: {counts}"
    # faces are vertex indices, so they compare exactly. Equal counts with different connectivity is
    # a retessellation, which the coordinate check below would not see.
    differing_faces = int(np.count_nonzero((test_faces != reference_faces).any(axis=1)))
    assert differing_faces == 0, (
        f"{surface} has the same vertex and face counts but {differing_faces} of "
        f"{test_faces.shape[0]} faces connect different vertices. Stage '{stage}'."
    )

    # euclidean distance per vertex, which is what "the surface moved" means
    distance = np.linalg.norm(test_coords - reference_coords, axis=1)
    tolerance = read_surface_tolerances()[surface]

    delta_dir: Path = pytestconfig.getoption("--collect_csv")
    if delta_dir:
        delta_dir.mkdir(parents=True, exist_ok=True)
        scores = {p: v for p, v in zip(("median", "95th", "99th"), np.percentile(distance, (50, 95, 99)), strict=True)}
        scores.update(mean=float(distance.mean()), max=float(distance.max()))
        write_table_file(delta_dir / "surface.csv", test_subject.name, surface, scores)

    moved = int(np.count_nonzero(distance > tolerance))
    assert moved == 0, (
        f"{surface}: {moved} of {distance.size} vertices moved more than {tolerance} mm, "
        f"worst {distance.max():.4f} mm. Stage '{stage}'."
    )
    logger.debug(f"{surface} matches within {tolerance} mm")


def pytest_generate_tests(metafunc: pytest.Metafunc):
    if "surface" in metafunc.fixturenames:
        # pipeline order, so the first failure is the first divergence
        surfaces = sorted(read_surface_tolerances().keys(), key=chain_order)
        metafunc.parametrize("surface", surfaces, scope="module")
