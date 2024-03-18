import os
import sys
import unittest
import yaml

class TestErrorMessages(unittest.TestCase):
    """
    A test case class to check for errors in the given files based on a YAML file.

    This class defines test methods to verify if each file specified in the YAML file exists in the given folder.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up the test class.

        This method loads a YAML file, retrieves the log directory and error messages from the file,
        and assigns them to class variables.
        """
        try:
            # Load the YAML file
            with open(cls.error_file_path, 'r') as file:
                data = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"YAML file not found at path: {cls.error_file_path}")
            return
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return

        # Get the folder path from the YAML file
        # cls.log_directory = data.get('log_path')
        # if not cls.log_directory:
        #     print("The 'log_path' key was not found in the YAML file")
        #     return
        cls.log_directory = ''
        # Get the error messages from the YAML file
        cls.errors = data.get('error_messages')
        if not cls.errors:
            print("The 'error_messages' key was not found in the YAML file")
            return

    def test_find_errors_in_logs(self):
        """
        Test that the error messages are not in the log files.

        This method retrieves the log files in the log directory, reads each log file,
        and checks that each error message is not in the log file.
        """
        if not self.log_directory or not self.errors:
            return

        try:
            # Retrieve the log files in given log directory
            log_files = [os.path.join(self.log_directory, file) 
                         for file in os.listdir(self.log_directory) if file.endswith('.log')]
        except FileNotFoundError:
            print(f"Log directory not found at path: {self.log_directory}")
            return

        # Check if each error message is in the log files
        for log_file in log_files:
            try:
                with open(log_file, 'r') as file:
                    log_data = file.read()
            except FileNotFoundError:
                print(f"Log file not found at path: {log_file}")
                continue

            for error in self.errors:
                self.assertNotIn(error, log_data)
                print(f"Error message '{error}' not found in {log_file}")

if __name__ == '__main__':
    """
    Main entry point of the script.

    This block checks if there are any command line arguments, assigns the first argument to the error_file_path class variable,
    and runs the unittest main function.
    """
    if len(sys.argv) > 1:
        TestErrorMessages.log_directory = sys.argv.pop()
        TestErrorMessages.error_file_path = sys.argv.pop()
        
        print('log directory: ' + TestErrorMessages.log_directory)
        print('error file path: ' + TestErrorMessages.error_file_path)
        
    else:
        print("Please provide error file and log directory")
        test_args = sys.argv.pop()
        print('test_args (else): ' + test_args)
    unittest.main()