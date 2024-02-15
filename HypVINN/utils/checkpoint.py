# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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
import os

from FastSurferCNN.utils import logging
from FastSurferCNN.utils.checkpoint import (
    FASTSURFER_ROOT,
    load_from_checkpoint,
    create_checkpoint_dir,
    get_checkpoint,
    get_checkpoint_path,
    save_checkpoint,
)

logger = logging.get_logger(__name__)

# Defaults
URL = "https://b2share.fz-juelich.de/api/files/7133b542-733b-4cc6-a284-5c333ff25f78"
HYPVINN_AXI = os.path.join(FASTSURFER_ROOT, "checkpoints/HypVINN_axial_v1.0.0.pkl")
HYPVINN_COR = os.path.join(FASTSURFER_ROOT, "checkpoints/HypVINN_coronal_v1.0.0.pkl")
HYPVINN_SAG = os.path.join(FASTSURFER_ROOT, "checkpoints/HypVINN_sagittal_v1.0.0.pkl")

