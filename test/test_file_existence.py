import os
import sys
import yaml
import unittest
import argparse

class TestFileExistence(unittest.TestCase):
    
    # yaml_file_path = ''
    
    # if len(sys.argv) > 2:
    #     yaml_file_path = sys.argv.pop()
    # else:
    #     print("Please provide the path to the YAML file")
    
    @classmethod
    def setUpClass(cls):
        try:
            # Load the YAML file
            with open(cls.yaml_file_path, 'r') as file:
                cls.data = yaml.safe_load(file)
        except FileNotFoundError:
            raise Exception(f"YAML file not found at path: {cls.yaml_file_path}")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML file: {e}")

        # Get the folder path from the YAML file
        cls.folder_path = cls.data.get('folder_path')
        if not cls.folder_path:
            raise Exception("The 'folder_path' key was not found in the YAML file")

    def test_file_existence(self):
        # Get a list of all files in the folder recursively
        filenames = []
        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                # Get the relative path from the current directory to the file
                rel_path = os.path.relpath(os.path.join(root, file), self.folder_path)
                filenames.append(rel_path)

        # Check if each file in the YAML file exists in the folder
        files = self.data.get('files')
        if not files:
            self.fail("The 'files' key was not found in the YAML file")

        for file in files:
            self.assertIn(file, filenames, f"File '{file}' does not exist in the folder.")

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("yaml_file_path", help="Path to the YAML file")
    # args = parser.parse_args()
    
    # TestFileExistence.yaml_file_path = sys.argv[2]
    
    if len(sys.argv) > 1:
        TestFileExistence.yaml_file_path = sys.argv.pop()
    else:
        print("Please provide the path to the YAML file")
        
    unittest.main()