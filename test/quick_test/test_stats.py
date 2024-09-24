import os
from logging import getLogger
from pathlib import Path

import pandas as pd
import pytest
import yaml
from torch.nn.functional import threshold

from .common import load_test_subjects

logger = getLogger(__name__)

file_types = ["aseg.stats", "aseg+DKT.stats", "aseg.presurf.hypos.stats", "cerebellum.CerebNet.stats",
              "hypothalamus.HypVINN.stats", "wmparc.DKTatlas.mapped.stats"]

@pytest.fixture
def thresholds(file_type):
    """
    Load the thresholds from the given file path.

    Returns
    -------
    default_threshold : float
        Default threshold value.
    thresholds : dict
        Dictionary containing the thresholds
    """

    # Load the thresholds file
    thresholds_file = Path(__file__).parent / "data/thresholds" / f"{file_type}.yaml"

    # Open the file_path and read the thresholds into a dictionary
    with open(thresholds_file) as file:
        data = yaml.safe_load(file)
        default_threshold = data.get("default_threshold")
        thresholds = data.get("thresholds", {})

    return default_threshold, thresholds


def load_stats_file(test_subject: Path, file_type: Path):
    """
    Load the stats file from the given file path.

    Parameters
    ----------
    test_subject : Path
        Path to the test subject.

    Returns
    -------
    stats_file : Path
    """

    files = os.listdir(test_subject / "stats")

    if "aseg.stats" in files:
        return test_subject / "stats" / "aseg.stats"
    elif "aparc+DKT.stats" in files:
        return test_subject / "stats" / "aparc+DKT.stats"
    else:
        raise ValueError("Unknown stats file")


def read_measure_stats(file_path: Path):
    """
    Read the measure stats from the given file path.

    Parameters
    ----------
    file_path : Path
        Path to the stats file.

    Returns
    -------
    measure : list
        List of measures.
    measurements : dict
        Dictionary containing the measurements.
    """

    measure = []
    measurements = {}

    # Retrieve lines starting with "# Measure" from the stats file
    with open(file_path) as file:
        # Read each line in the file
        for _i, line in enumerate(file, 1):
            # Check if the line starts with "# ColHeaders"
            if line.startswith("# ColHeaders"):
                line.removeprefix("# ColHeaders").strip().split(" ")

            # Check if the line starts with "# Measure"
            if line.startswith("# Measure"):
                # Strip "# Measure" from the line
                line = line.removeprefix("# Measure").strip()
                # Append the measure to the list
                line = line.split(", ")
                measure.append(line[1])
                measurements[line[1]] = float(line[3])

    return measure, measurements


def read_table(file_path: Path, file_type: Path):
    """
    Read the table from the given file path.

    Parameters
    ----------
    file_path : Path
        Path to the stats file.
    file_type : Path
        Type of the file.

    Returns
    -------
    table : pandas.DataFrame
        Table containing the
    """

    table_start = 0
    columns = []

    file_path = file_path / "stats" / file_type

    # Retrieve stats table from the stats file
    with open(file_path) as file:
        # Read each line in the file
        for i, line in enumerate(file, 1):
            # Check if the line starts with "# ColHeaders"
            if line.startswith("# ColHeaders"):
                table_start = i
                columns = line.removeprefix("# ColHeaders").strip().split(" ")

    # Read the reference table into a pandas dataframe
    table = pd.read_table(file_path, skiprows=table_start, sep="\s+", header=None)
    table.columns = columns
    table.set_index(columns[0], inplace=True)

    return table


@pytest.mark.parametrize("file_type", file_types)
@pytest.mark.parametrize("test_subject", load_test_subjects())
def test_measure_exists(subjects_dir: Path, test_dir: Path, reference_dir: Path, test_subject: Path, file_type: Path):
    """
    Test if the measure exists in the stats file.

    Parameters
    ----------
    subjects_dir : Path
        Path to the subjects directory.
    test_dir : Path
        Name of the test directory.
    test_subject : Path
        Name of the test subject.

    Raises
    ------
    AssertionError
        If the measure does not exist in the stats file.
    """

    test_subject = subjects_dir / test_dir / test_subject
    test_file = test_subject / "stats" / file_type

    reference_subject = subjects_dir / reference_dir / test_subject
    reference_file = reference_subject / "stats" / file_type

    test_data = read_measure_stats(test_file)
    ref_data = read_measure_stats(reference_file)
    errors = []

    for struct in ref_data[1]:
        if struct not in test_data[1]:
            print("\nstruct:", struct)
            errors.append(
                f"for struct {struct} the value {test_data[1].get(struct)} is not close to {ref_data[1].get(struct)}"
            )

    # Check if all measures exist in stats file
    assert len(errors) == 0, ", ".join(errors)


@pytest.mark.parametrize("file_type", file_types)
@pytest.mark.parametrize("test_subject", load_test_subjects())
def test_tables(subjects_dir: Path, test_dir: Path, reference_dir: Path, test_subject: Path, thresholds,
                 file_type: Path):
    """
    Test if the tables are within the threshold.

    Parameters
    ----------
    subjects_dir : Path
        Path to the subjects directory.
    test_dir : Path
        Name of the test directory.
    reference_dir : Path
        Name of the reference directory.
    test_subject : Path
        Name of the test subject.
    thresholds : tuple
        Tuple containing the default threshold and the thresholds.

    Raises
    ------
    AssertionError
        If the table values are not within the threshold.
    """

    # Load the test and reference tables
    test_file = subjects_dir / test_dir / test_subject
    test_table = read_table(test_file, file_type)

    reference_subject = subjects_dir / reference_dir / test_subject
    ref_table = read_table(reference_subject, file_type)

    # Load the thresholds
    default_threshold, thresholds = thresholds

    variations = {}

    # Check if table values are within the threshold
    for i in ref_table.index:
        struct = ref_table.loc[i, "StructName"]
        for j in ref_table.columns:
            if j == "StructName":
                continue
            threshold = default_threshold
            if ref_table.loc[i, j] == 0:
                continue
            variation = (test_table.loc[i, j] / ref_table.loc[i, j]) - 1
            if abs(variation) > threshold:
                variations[struct] = {j: abs(variation)}

    if variations:
        logger.debug("\nVariations greater than threshold:")
        for key, value in variations.items():
            logger.debug(key, value)

    return variations


