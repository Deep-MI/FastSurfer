
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
import sys
import logging
from os import makedirs
from os.path import join


def setup_logging(output_dir, expr_num):
    """
    Sets up the logging
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
    log_folder = join(output_dir, "logs")
    makedirs(log_folder, exist_ok=True)
    log_file = join(log_folder, f"expr_{expr_num}.log")

    fh = logging.FileHandler(filename=log_file, mode='a')
    ch = logging.StreamHandler(sys.stdout)

    logging.basicConfig(level=logging.INFO, format=_FORMAT, handlers=[fh, ch])


def get_logger(name):
    """
    Retrieve the logger with the given name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)
