import os
import yaml
import unittest

class TestFileExistence(unittest.TestCase):
    def test_file_existence(self):
        yaml_file_path = './files.yaml'  # './' refers to the current directory

        # Load the YAML file
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Get the folder path from the YAML file
        folder_path = data['folder_path']

        # Get a list of all files in the folder recursively
        filenames = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Get the relative path from the current directory to the file
                rel_path = os.path.relpath(os.path.join(root, file), folder_path)
                filenames.append(rel_path)

        # Check if each file in the YAML file exists in the folder
        for file_name in data['files']:
            self.assertIn(file_name, filenames, f"File '{file_name}' does not exist in the folder.")

if __name__ == '__main__':
    unittest.main()