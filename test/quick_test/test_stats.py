import sys
import yaml
import unittest
import argparse
from pathlib import Path
import pandas as pd

class TestStats(unittest.TestCase):
    """
    A test case class to check the existence of files in a folder based on a YAML file.

    This class defines test methods to verify if each file specified in the YAML file exists in the given folder.
    """

    current_file_path = Path(__file__)
    stats_thresholds: Path = current_file_path.parent / "data/stats.yaml"

    ref_path = None
    ref_measure = None
    ref_measurements = None
    ref_table = None

    stats_path = None
    stats_measure = None
    stats_measurements = None
    stats_table = None

    @classmethod
    def setUpClass(cls):
        """
        Set up the test case by loading the YAML file and extracting the folder path.
        """

        # Open the configuration file containing thresholds and read into a data array
        with cls.stats_thresholds.open('r') as file:
            cls.thresholds = yaml.safe_load(file)

        # Reference stats file
        cls.ref_measure = []
        cls.ref_measurements = {}
        table_start = 0
        columns = []

        # Retrieve lines starting with "# Measure" from the stats file
        with open(cls.ref_path, 'r') as file:
            # Read each line in the file
            for i, line in enumerate(file, 1):

                # Check if the line starts with "# ColHeaders"
                if line.startswith("# ColHeaders"):
                    table_start = i
                    columns = line.strip("# ColHeaders").strip().split(" ")

                # Check if the line starts with "# Measure"
                if line.startswith("# Measure"):
                    # Strip "# Measure" from the line
                    line = line.strip("# Measure").strip()
                    # Append the measure to the list
                    line = line.split(", ")
                    cls.ref_measure.append(line[1])
                    cls.ref_measurements[line[1]] = float(line[3])
                    # print(cls.measurements)

        # Read the reference table into a pandas dataframe
        cls.ref_table = pd.read_table(cls.ref_path, skiprows=table_start, sep="\s+", header=None, nrows=5)
        cls.ref_table_numeric = cls.ref_table.apply(pd.to_numeric, errors='coerce')
        cls.ref_table.columns = columns
        cls.ref_table.set_index(columns[0], inplace=True)
        print(cls.ref_table)

        # Stats file
        cls.stats_measure = []
        cls.stats_measurements = {}
        # Retrieve lines starting with "# Measure" from the stats file
        with open(cls.stats_path, 'r') as file:
            # Read each line in the file
            for i, line in enumerate(file, 1):

                # Check if the line starts with "# ColHeaders"
                if line.startswith("# ColHeaders"):
                    table_start = i
                    columns = line.strip("# ColHeaders").strip().split(" ")

                # Check if the line starts with "# Measure"
                if line.startswith("# Measure"):
                    # Strip "# Measure" from the line
                    line = line.strip("# Measure").strip()
                    # Append the measure to the list
                    line = line.split(", ")
                    cls.stats_measure.append(line[1])
                    cls.stats_measurements[line[1]] = float(line[3])
                    # print(cls.measurements)

        # Read the stats table into a pandas dataframe
        cls.stats_table = pd.read_table(cls.stats_path, skiprows=table_start, sep="\s+", header=None, nrows=5)
        cls.stats_table_numeric = cls.stats_table.apply(pd.to_numeric, errors='coerce')
        cls.stats_table.columns = columns
        cls.stats_table.set_index(columns[0], inplace=True)
        print(cls.stats_table)

    def test_stats(self):

        # Check if all measures exist in stats file
        for ref_measure in self.ref_measure:
            self.assertIn(ref_measure, self.stats_measure, f"Measure {ref_measure} not found in the stats file.")

        # Check if measures in the stats file are within the threshold
        for struct in self.ref_measurements:
            variation = ((self.stats_measurements.get(struct) / self.ref_measurements.get(struct)) - 1)
            # If threshold exists for struct, use it, else use default threshold
            if threshold := self.thresholds.get(struct):
                self.assertLessEqual(variation, threshold,
                                     f"Variation of {struct} is greater than threshold.")
            else:
                self.assertLessEqual(variation, self.thresholds.get("default_threshold"),
                                     f"Variation of {struct} is greater than threshold.")

        # Check if table values are within the threshold
        for i in self.ref_table.index:
            for j in self.ref_table.columns:
                if j == "StructName":
                    continue
                threshold = self.thresholds.get("default_threshold")
                variation = ((self.stats_table.loc[i, j] / self.ref_table.loc[i, j]) - 1)
                self.assertLessEqual(variation, threshold, msg=f"Variation of {j} in the table is greater than "
                                                               f"threshold.")


if __name__ == '__main__':
    """
    The main method to run the test case.
    
    This method parses the command line arguments and runs the test case.
    """

    parser = argparse.ArgumentParser(description="Test for stats based on a YAML file.")
    parser.add_argument('ref_path', type=Path, help="The path to the reference stats file.")
    parser.add_argument('stats_path', type=Path, help="The path to the stats file.")

    args = parser.parse_args()

    TestStats.ref_path = args.ref_path
    TestStats.stats_path = args.stats_path

    unittest.main(argv=[sys.argv[0]])
