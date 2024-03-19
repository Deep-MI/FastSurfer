import os
import unittest
import yaml

class TestErrorMessages(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        error_file_path = './test/error_file.yaml'  # './' refers to the current directory

        try:
            # Load the YAML file
            with open(error_file_path, 'r') as file:
                data = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"YAML file not found at path: {error_file_path}")
            return
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return

        # Get the folder path from the YAML file
        self.log_directory = data.get('log_path')
        if not self.log_directory:
            print("The 'log_path' key was not found in the YAML file")
            return

        # Get the error messages from the YAML file
        self.errors = data.get('error_messages')
        if not self.errors:
            print("The 'error_messages' key was not found in the YAML file")
            return

    def test_find_errors_in_logs(self):
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

if __name__ == '__main__':
    unittest.main()