import argparse
import sys
import unittest
from pathlib import Path

import yaml


class TestFileExistence(unittest.TestCase):
    """
    A test case class to check the existence of files in a folder based on a YAML file.

    This class defines test methods to verify if each file specified in the YAML file exists in the given folder.
    """

    file_path: Path = Path("./test/quick_test/data/files.yaml")

    @classmethod
    def setUpClass(cls):
        """
        Set up the test case by loading the YAML file and extracting the folder path.

        This method is executed once before any test methods in the class.
        """

        # Open the file_path and read the files into an array
        with cls.file_path.open('r') as file:
            data = yaml.safe_load(file)
            cls.files = data.get('files', [])

        # Get a list of all files in the folder recursively
        cls.filenames = []
        for file in cls.folder_path.glob('**/*'):
            if file.is_file():
                # Get the relative path from the current directory to the file
                rel_path = file.relative_to(cls.folder_path)
                cls.filenames.append(str(rel_path))

    def test_file_existence(self):
        """
        Test method to check the existence of files in the folder.

        This method gets a list of all files in the folder recursively and checks
        if each file specified in the YAML file exists in the folder.
        """

        # Check if each file in the YAML file exists in the folder
        if not self.files:
            self.fail("The 'files' key was not found in the YAML file")

        for file in self.files:
            print(f"Checking for file: {file}")
            self.assertIn(file, self.filenames, f"File '{file}' does not exist in the folder.")

        print("All files present")


if __name__ == '__main__':
    """
    Main entry point of the script.

    This block checks if there are any command line arguments, assigns the first argument
    to the error_file_path class variable, and runs the unittest main function.
    """

    parser = argparse.ArgumentParser(description="Test for file existence based on a YAML file.")
    parser.add_argument('folder_path', type=Path, help="The path to the folder to check.")

    args = parser.parse_args()

    TestFileExistence.folder_path = args.folder_path

    unittest.main(argv=[sys.argv[0]])
