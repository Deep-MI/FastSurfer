import sys
import yaml
import unittest
import argparse
from pathlib import Path


class TestErrors(unittest.TestCase):
    """
    A test case class to check for the word "error" in the given log files.
    """

    error_file_path: Path = Path("./test/quick_test/data/errors.yaml")

    error_flag = False

    @classmethod
    def setUpClass(cls):
        """
        Set up the test class.
        This method retrieves the log directory from the command line argument,
        and assigns it to a class variable.
        """

        # Open the error_file_path and read the errors and whitelist into arrays
        with open(cls.error_file_path, 'r') as file:
            data = yaml.safe_load(file)
            cls.errors = data.get('errors', [])
            cls.whitelist = data.get('whitelist', [])

        # Retrieve the log files in given log directory
        try:
            # cls.log_directory = Path(cls.log_directory)
            print(cls.log_directory)
            cls.log_files = [file for file in cls.log_directory.iterdir() if file.suffix == '.log']
        except FileNotFoundError:
            raise FileNotFoundError(f"Log directory not found at path: {cls.log_directory}")

    def test_find_errors_in_logs(self):
        """
        Test that the words "error", "exception", and "traceback" are not in the log files.

        This method retrieves the log files in the log directory, reads each log file line by line,
        and checks that none of the keywords are in any line.
        """

        files_with_errors = {}

        # Check if any of the keywords are in the log files
        for log_file in self.log_files:
            rel_path = log_file.relative_to(self.log_directory)
            print(f"Checking file: {rel_path}")
            try:
                with log_file.open('r') as file:
                    lines = file.readlines()
                    lines_with_errors = []
                    for line_number, line in enumerate(lines, start=1):
                        if any(error in line.lower() for error in self.errors):
                            if not any(white in line.lower() for white in self.whitelist):
                                # Get two lines before and after the current line
                                context = lines[max(0, line_number-2):min(len(lines), line_number+3)]
                                lines_with_errors.append((line_number, context))
                                print(lines_with_errors)
                                files_with_errors[rel_path] = lines_with_errors
                                self.error_flag = True
            except FileNotFoundError:
                raise FileNotFoundError(f"Log file not found at path: {log_file}")
                continue

        # Print the lines and context with errors for each file
        for file, lines in files_with_errors.items():
            print(f"\nFile {file}, in line {files_with_errors[file][0][0]}:")
            for line_number, line in lines:
                print(*line, sep = "")

        # Assert that there are no lines with any of the keywords
        self.assertEqual(self.error_flag, False, f"Found errors in the following files: {files_with_errors}")
        print("No errors found in any log files.")


if __name__ == '__main__':
    """
    Main entry point of the script.
    
    This block checks if there are any command line arguments, 
    assigns the first argument to the log_directory class variable
    """

    parser = argparse.ArgumentParser(description="Test for errors in log files.")
    parser.add_argument('log_directory', type=Path, help="The directory containing the log files.")

    args = parser.parse_args()

    TestErrors.log_directory = args.log_directory

    unittest.main(argv=[sys.argv[0]])
