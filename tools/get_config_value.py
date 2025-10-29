# Copyright 2024 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
import argparse
import sys
from pathlib import Path

import yaml


def make_parser() -> argparse.ArgumentParser:
    """
    Create a command line interface and return command line options.

    Returns
    -------
    options
        Namespace object holding options.
    """
    parser = argparse.ArgumentParser(
        description="Tool for extracting values from configuration file",
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        dest="file",
        help="configuration file",
        required=True,
    )
    parser.add_argument(
        "--key", "-k",
        type=str,
        dest="key",
        help="key to lookup",
        required=True,
    )
    return parser


def extract_value(config_file: Path | str, key: str) -> str:
    """
    Read configuration file and return the value for the given key.

    Parameters
    ----------
    config_file : Path, str
        Path to configuration file.
    key : str
        Key to lookup.

    Returns
    -------
    value
        Value under the given key.
    """
    with open(config_file) as config:
        try:
            conf = yaml.safe_load(config)
            value = conf[key.split('.')[0]]
            for k in key.split('.')[1:]:
                value = value[k]
            return value
        except yaml.YAMLError as e:
            print(e) 
            sys.exit(1)

    
if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    print(extract_value(args.file,args.key))
    sys.exit(0)
