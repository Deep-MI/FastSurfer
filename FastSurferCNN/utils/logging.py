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
from logging import INFO, FileHandler, StreamHandler, basicConfig
from pathlib import Path as _Path
from sys import stdout as _stdout


def setup_logging(log_file_path: _Path | str):
    """
    Set up the logging.

    Parameters
    ----------
    log_file_path : Path, str
        Path to the logfile.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
    handlers = [StreamHandler(_stdout)]

    if log_file_path:
        if not isinstance(log_file_path, _Path):
            log_file_path = _Path(log_file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        handlers.append(FileHandler(filename=log_file_path, mode="a"))

    basicConfig(level=INFO, format=_FORMAT, handlers=handlers)
