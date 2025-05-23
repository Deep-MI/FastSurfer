from functools import lru_cache
from logging import getLogger
from pathlib import Path

import numpy as np
import pytest

from FastSurferCNN.segstats import PVStats

from .common import SubjectDefinition, Tolerances, write_table_file
from .helper import Approx

logger = getLogger(__name__)

@lru_cache
def read_stats_files() -> list[str]:
    return [file.name[:-5] for file in (Path(__file__).parent / "data").glob("*.stats.yaml")]


class MeasureTolerances(Tolerances):

    def __check_measure_thresholds(self) -> bool:
        if not (is_dict := isinstance(self.config["measure_thresholds"], dict)):
            logger.warning(f"measure_thresholds for {self.config_file} is not a dictionary!")
        return is_dict

    def measures(self) -> list[str]:
        if self.__check_measure_thresholds():
            return list(self.config["measure_thresholds"].keys())
        else:
            return []

    def threshold(self, measure: str) -> float:
        """Returns the threshold belonging to the measure passed."""
        if self.__check_measure_thresholds():
            try:
                return self.config["measure_thresholds"][measure]
            except KeyError:
                pass
        return self.config["default_threshold"]


@pytest.fixture(scope="module")
def measure_tolerances(stats_file: str) -> MeasureTolerances:
    """
    Read the expected measures for the given stats file.

    Parameters
    ----------
    stats_file : str
        The name of the stats file.

    Returns
    -------
    MeasureTolerances
        The list of measures expected for this file.
    """
    thresholds_file = Path(__file__).parent / f"data/{stats_file}.yaml"
    assert Path(thresholds_file).is_file(), f"The threshold file {thresholds_file} does not exist!"
    return MeasureTolerances(thresholds_file)

@pytest.fixture(scope="module")
def stats_tolerances(stats_file: str) -> Tolerances:
    """
    Load the thresholds from the given file path.

    Parameters
    ----------
    stats_file : str
        The name of the stats file.

    Returns
    -------
    Tolerances
        Per-structure tolerances object
    """
    thresholds_file = Path(__file__).parent / f"data/{stats_file}.yaml"
    assert Path(thresholds_file).is_file(), f"The threshold file {thresholds_file} does not exist!"
    return Tolerances(thresholds_file)


def test_measure_exists(
        test_subject: SubjectDefinition,
        stats_file: str,
        measure_tolerances: MeasureTolerances,
):
    """
    Test if the measure exists in the stats file.

    Parameters
    ----------
    test_subject : SubjectDefinition
        Definition of the test subject.
    stats_file : str
        Name of the test directory.
    measure_tolerances : MeasureTolerances
        The object to provide the measure tolerances for stats_file.

    Raises
    ------
    AssertionError
        If the measure does not exist in the stats file.
    """
    _, annotations, _ = test_subject.load_stats_file(stats_file)
    expected_measures = measure_tolerances.measures()

    if not expected_measures:
        # no expected measures, skip
        pytest.skip(f"No measures expected for {stats_file}.")
        return
    # measures are missing => None => not a tuple
    # measures are not a tuple => not a tuple
    missing_invalid_measures = [m for m in expected_measures if not isinstance(annotations.get(m, None), tuple)]

    # Check if all measures exist in stats file
    assert missing_invalid_measures == [], f"Some Measures are missing in {test_subject}: {stats_file}!"


def test_measure_meta(
        test_subject: SubjectDefinition,
        ref_subject: SubjectDefinition,
        stats_file: str,
        measure_tolerances: MeasureTolerances,
):
    """
    Test if the measure meta-information is correct in stats_file.

    Parameters
    ----------
    test_subject : SubjectDefinition
        Definition of the test subject.
    ref_subject : SubjectDefinition
        Definition of the reference subject.
    stats_file : str
        Name of the test directory.
    measure_tolerances : MeasureTolerances
        The object to provide the measure tolerances for stats_file.

    Raises
    ------
    AssertionError
        If the measure is not within the defined threshold in the stats file.
    """
    _, test_annots, _ = test_subject.load_stats_file(stats_file)
    _, ref_annots, _ = ref_subject.load_stats_file(stats_file)

    expected_measures = measure_tolerances.measures()
    if not expected_measures:
        # no expected measures, skip
        pytest.skip(f"No measures expected for {stats_file}.")
        return
    _expected_meta = {k: ref_annots[k][:2] + (None,) + ref_annots[k][3:] for k in expected_measures if k in ref_annots}
    _actual_meta = {k: test_annots[k][:2] + (None,) + test_annots[k][3:] for k in expected_measures if k in test_annots}

    assert _actual_meta == _expected_meta, f"Some Measure meta-information is wrong for {test_subject} in {stats_file}!"


def test_measure_thresholds(
        test_subject: SubjectDefinition,
        ref_subject: SubjectDefinition,
        stats_file: str,
        measure_tolerances: MeasureTolerances,
        pytestconfig: pytest.Config,
):
    """
    Test if the measure is within thresholds in stats_file.

    Parameters
    ----------
    test_subject : SubjectDefinition
        Definition of the test subject.
    ref_subject : SubjectDefinition
        Definition of the reference subject.
    stats_file : str
        Name of the test directory.
    measure_tolerances : MeasureTolerances
        The object to provide the measure tolerances for stats_file.
    pytestconfig : pytest.Config
        The sessions config object.

    Raises
    ------
    AssertionError
        If the measure is not within the defined threshold in the stats file.
    """
    _, expected_annots, _ = ref_subject.load_stats_file(stats_file)
    _, actual_annots, _ = test_subject.load_stats_file(stats_file)

    expected_measures = measure_tolerances.measures()
    if not expected_measures:
        # no expected measures, skip
        pytest.skip(f"No measures expected for {stats_file}.")
        return
    # the more pytest-way to evaluate the thresholds would be
    # assert expected_measures == pytest.approx(actual_measures), (f"Some Measures are outside of the threshold in"
    #                                                              f"{test_subject}: {stats_file}!")
    # but this does not work because we have different thresholds for different measures.

    def has_measure(m: str) -> bool:
        return m in expected_annots and m in actual_annots

    def check_measure(measure: str) -> bool:
        if not has_measure(measure):
            return False
        expected: int | float = expected_annots[measure][2]
        actual: int | float = actual_annots[measure][2]
        return expected == Approx(actual, rel=measure_tolerances.threshold(measure))

    delta_dir: Path = pytestconfig.getoption("--collect_csv")
    if delta_dir:
        delta_dir.mkdir(parents=True, exist_ok=True)
        values = [(m, expected_annots[m][2], actual_annots[m][2]) for m in expected_measures if has_measure(m)]
        scores: dict[str, float] = {m: abs(a - b) for m, a, b in values}
        scores.update({m + "_rel": abs(a - b)/max((abs(a), abs(b))) for m, a, b in values})
        write_table_file(delta_dir / "stats-measure.csv", test_subject.name, stats_file, scores)

    failed_measures = (m for m in expected_measures if not check_measure(m))
    measures_outside_spec = [f"Measure {m}: {expected_annots[m][2]} <> {actual_annots[m][2]}" for m in failed_measures]
    assert measures_outside_spec == [], f"Some Measures are outside of the threshold in {test_subject}: {stats_file}!"


def test_table_structs(
        test_subject: SubjectDefinition,
        ref_subject: SubjectDefinition,
        stats_file: str,
):
    """
    Test if the structs (and potentially SegId information) are in stats_file.

    Parameters
    ----------
    test_subject : SubjectDefinition
        Definition of the test subject.
    ref_subject : SubjectDefinition
        Definition of the reference subject.
    stats_file : str
        Name of the test directory.

    Raises
    ------
    AssertionError
        If the table values are not within the threshold.
    """
    _, _, expected_table = ref_subject.load_stats_file(stats_file)
    _, _, actual_table = test_subject.load_stats_file(stats_file)
    # make sure there is no duplicates in expected and the same number of rows
    expected_num = np.unique([e["SegId"] for e in expected_table]).shape[0]
    assert expected_num == len(expected_table), f"The number of rows in {stats_file} is not the expected!"
    # check if the same SegIds and StructName pairs are in the table
    expected_segids_structs = {stats["SegId"]: stats.get("StructName", None) for stats in expected_table}
    actual_segids_structs = {stats["SegId"]: stats.get("StructName", None) for stats in actual_table}
    assert actual_segids_structs == expected_segids_structs, f"The segid-struct pairs in {stats_file} are different!"


def test_stats_table(
        test_subject: SubjectDefinition,
        ref_subject: SubjectDefinition,
        stats_file: str,
        stats_tolerances: Tolerances,
        pytestconfig: pytest.Config,
):
    """
    Test if the tables are within the threshold.

    Parameters
    ----------
    test_subject : SubjectDefinition
        Definition of the test subject.
    ref_subject : SubjectDefinition
        Definition of the reference subject.
    stats_file : str
        Name of the test directory.
    stats_tolerances : Tolerances
        The object to provide the tolerances for stats_file.
    pytestconfig : pytest.Config
        The sessions config object.

    Raises
    ------
    AssertionError
        If the table values are not within the threshold.
    """
    _, _, expected_table = ref_subject.load_stats_file(stats_file)
    _, _, actual_table = test_subject.load_stats_file(stats_file)
    actual_segids = [stats["SegId"] for stats in actual_table]
    ignored_columns = ["SegId", "StructName"]

    def filter_keys(stats: PVStats) -> dict[str, int | float]:
        return {k: v for k, v in stats.items() if k not in ignored_columns}

    delta_dir: Path = pytestconfig.getoption("--collect_csv")
    table_data = []
    selected_columns = []

    expected_conflicts = []
    actual_conflicts = []
    for expected in expected_table:
        expected_segid = expected["SegId"]
        if expected_segid in actual_segids:
            expected_selected_cols = filter_keys(expected)
            actual = actual_table[actual_segids.index(expected_segid)]
            actual_selected_cols = filter_keys(actual)
            if expected_selected_cols != Approx(actual_selected_cols, abs=stats_tolerances.threshold(expected_segid)):
                expected_conflicts.append(expected)
                actual_conflicts.append(actual)
            if delta_dir:
                selected_columns.append((list(expected_selected_cols.keys()), list(actual_selected_cols.keys())))
                table_data.append((expected_segid, expected_selected_cols, actual_selected_cols))
        else:
            # did not find expected_segid in actual_segids (also fail for test_table_structs)
            # add the segid to expected_conflicts, but ignore for csv computation
            expected_conflicts.append(expected)

    if delta_dir:
        from collections.abc import Hashable
        from typing import TypeVar
        _T = TypeVar("_T", bound=Hashable)

        def relative(a: dict[_T, int | float], b: dict[_T, int | float], field: _T) -> float:
            return abs(a[field] - b[field]) / max((abs(a[field]), abs(b[field]), 1e-8))

        delta_dir.mkdir(parents=True, exist_ok=True)
        expected_selected_cols, actual_selected_cols = zip(*selected_columns, strict=True)
        assert expected_selected_cols == actual_selected_cols, "To compare tables columns have to be identical!"
        assert len(actual_selected_cols) > 0, "actual must have columns!"
        for field in actual_selected_cols[0]:
            scores: dict[str, float] = {f"{seg_id}:{field}": abs(a[field] - b[field]) for seg_id, a, b in table_data}
            scores.update({f"{seg_id}:rel-{field}": relative(a, b, field) for seg_id, a, b in table_data})
            write_table_file(delta_dir / "stats-table.csv", test_subject.name, stats_file, scores)

    assert actual_conflicts == expected_conflicts, f"The differences for some structures in {stats_file} exceed limits!"


def pytest_generate_tests(metafunc: pytest.Metafunc):
    # populate the stats_file fixture
    if "stats_file" in metafunc.fixturenames:
        metafunc.parametrize("stats_file", read_stats_files(), scope="module")
