import os
import sys
import yaml
import unittest
import argparse


class TestErrors(unittest.TestCase):
    """
    A test case class to check for the word "error" in the given log files.
    """

    error_file_path = "./test/quick_test/data/errors.yaml"

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

        try:
            # Retrieve the log files in given log directory
            cls.log_files = [os.path.join(cls.log_directory, file)
                             for file in os.listdir(cls.log_directory) if file.endswith('.log')]
        except FileNotFoundError:
            raise FileNotFoundError(f"Log directory not found at path: {cls.log_directory}")

    def test_find_errors_in_logs(self):
        """
        Test that the words "error", "exception", and "traceback" are not in the log files.

        This method retrieves the log files in the log directory, reads each log file line by line,
        and checks that none of the keywords are in any line.
        """

        files_with_keywords = {}
        # errors = ["error", "exception", "traceback"]
        # white_list = ["without error", "correcting", "distance error", ]

        # Check if any of the keywords are in the log files
        for log_file in self.log_files:
            lines_with_keywords = []
            try:
                with open(log_file, 'r') as file:
                    for line_number, line in enumerate(file, start=1):
                        if any(keyword in line.lower() for keyword in self.errors):
                            if not any(white in line.lower() for white in self.whitelist):
                                lines_with_keywords.append((line_number, line.strip()))
            except FileNotFoundError:
                raise FileNotFoundError(f"Log file not found at path: {log_file}")
                continue

        if lines_with_keywords:
            files_with_keywords[log_file] = lines_with_keywords

        # Print the lines with keywords for each file
        for file, lines in files_with_keywords.items():
            print(f"\nIn file {file}, found errors in the following lines:")
            for line_number, line in lines:
                print(f"Line {line_number}: {line}")

        # Assert that there are no lines with any of the keywords
        self.assertEqual(files_with_keywords, {}, f"Found errors in the following files: {files_with_keywords}")


if __name__ == '__main__':
    """
    Main entry point of the script.
    
    This block checks if there are any command line arguments, 
    assigns the first argument to the log_directory class variable
    """

    parser = argparse.ArgumentParser(description="Test for errors in log files.")
    parser.add_argument('log_directory', type=str, help="The directory containing the log files.")

    args = parser.parse_args()

    TestErrors.log_directory = args.log_directory

    unittest.main(argv=[sys.argv[0]])