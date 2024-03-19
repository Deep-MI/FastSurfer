import os
import sys
import yaml
import unittest


class TestFileExistence(unittest.TestCase):
    """
    A test case class to check the existence of files in a folder based on a YAML file.

    This class defines test methods to verify if each file specified in the YAML file exists in the given folder.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test case by loading the YAML file and extracting the folder path.

        This method is executed once before any test methods in the class.
        """
        yaml_file_path = './test/files.yaml'  # './' refers to the current directory

        try:
            # Load the YAML file
            with open(yaml_file_path, 'r') as file:
                cls.data = yaml.safe_load(file)
        except FileNotFoundError:
            raise Exception(f"YAML file not found at path: {yaml_file_path}")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML file: {e}")

        # Get the folder path from the YAML file
        cls.folder_path = cls.data.get('folder_path')
        if not cls.folder_path:
            raise Exception("The 'folder_path' key was not found in the YAML file")

    def testFileExistence(self):
        """
        Test method to check the existence of files in the folder.

        This method gets a list of all files in the folder recursively and checks if each file specified in the YAML file exists in the folder.
        """
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
    
    if len(sys.argv) > 1:
        TestFileExistence.yaml_file_path = sys.argv.pop()
    else:
        print("Please provide the path to the YAML file")
    
    unittest.main()