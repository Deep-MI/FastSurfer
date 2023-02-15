
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


def setup_logging(output: str, expr_num: str=None):
    """
    Sets up the logging
    during trainnig output is a directory and expr_num is the experiment number
    during inference output is a full file path and expr_num is None
    TODO: resolve this dual purpose function
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    if expr_num is not None: # training
        assert(path.isdir(output)), f"Output directory {output} does not exist."
        log_folder = path.join(output, "logs")
        makedirs(log_folder, exist_ok=True)
        log_file = path.join(log_folder, f"expr_{expr_num}.log")
    elif expr_num is None and not path.isfile(output):
        raise ValueError("A full full filepath during inference (i.e. when no experiment number is specified).")
    else: # full file path given and no experiment number -> inference
        log_file = output

    fh = FileHandler(filename=log_file, mode='a')
    ch = StreamHandler(_stdout)

    basicConfig(level=INFO, format=_FORMAT, handlers=[fh, ch])
