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
import logging as _logging
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, FileHandler, Logger, StreamHandler, basicConfig, getLogger
from logging import getLogger as get_logger
from os import environ as _environ
from pathlib import Path as _Path
from sys import stdout as _stdout

VALID_LOG_LEVEL_STRINGS = ("INFO", "DEBUG", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL")


def setup_logging(log_file_path: _Path | str | None = None, log_level: int | None = None) -> None:
    """
    Set up the logging.

    Parameters
    ----------
    log_file_path : Path, str, optional
        Path to the logfile (only log to file if passed).
    log_level : int, optional
        Logging level defaults to environment variable FASTSURFER_LOG_LEVEL.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
    handlers = [StreamHandler(_stdout)]

    if log_file_path:
        if not isinstance(log_file_path, _Path):
            log_file_path = _Path(log_file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        handlers.append(FileHandler(filename=log_file_path, mode="a"))

    if log_level is None:
        log_level = _environ.get("FASTSURFER_LOG_LEVEL", "INFO").upper()
    if isinstance(log_level, str):
        log_level = log_level.upper()

    if not isinstance(log_level, str) or log_level not in VALID_LOG_LEVEL_STRINGS:
        try:
            log_number = int(log_level)
            if log_number < 0:
                log_level = "ERROR"
            elif log_number == 0:
                log_level = "WARNING"
            elif log_number == 1:
                log_level = "INFO"
            else: # > 1
                log_level = "DEBUG"
        except ValueError:
            raise ValueError(f"Invalid log level: {log_level}") from None

    basicConfig(level=getattr(_logging, log_level), format=_FORMAT, handlers=handlers)
