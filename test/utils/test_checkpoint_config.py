# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

"""
Guard the md5 of the default checkpoints in */config/checkpoint_paths.yaml.

The md5 of a checkpoint is what its download is checked against, so a mistyped one fails every
download of that checkpoint, long after the commit that bumped its version.
"""

import re

import pytest

from FastSurferCNN.download_checkpoints import MODULES
from FastSurferCNN.utils.checkpoint import get_config_file, load_checkpoint_config_defaults


@pytest.mark.parametrize("module", MODULES)
def test_every_default_checkpoint_has_a_valid_md5(module):
    config_file = get_config_file(module)
    # loading the config already fails for a checkpoint without md5
    checkpoints = load_checkpoint_config_defaults("checkpoint", config_file)

    malformed = {key: file.md5 for key, file in checkpoints.items() if not re.fullmatch("[0-9a-f]{32}", file.md5)}
    assert not malformed, f"{config_file} has md5 values that are not 32 hex digits: {malformed}"
