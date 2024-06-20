
# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# IMPORTS
from logging import *
from logging import getLogger as get_logger, StreamHandler, FileHandler, INFO, DEBUG, getLogger, basicConfig
from os import path, makedirs
from sys import stdout as _stdout
import sys


def setup_logging(log_file_path: str = None):
    """
    Sets up the logging
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
    handlers = [StreamHandler(_stdout)]

    if log_file_path:
        log_dir_path = path.dirname(log_file_path)
        if not path.exists(log_dir_path):
            makedirs(log_dir_path)

        handlers.append(FileHandler(filename=log_file_path, mode='a'))

    

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        get_logger().error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    basicConfig(level=INFO, format=_FORMAT, handlers=handlers)

    get_logger().info(f"Logging to {log_file_path}")






