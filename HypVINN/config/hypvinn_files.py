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
from FastSurferCNN.utils.checkpoint import FASTSURFER_ROOT

HYPVINN_LUT = FASTSURFER_ROOT / "HypVINN/config/HypVINN_ColorLUT.txt"

HYPVINN_STATS_NAME = "hypothalamus.HypVINN.stats"

HYPVINN_MASK_NAME = "hypothalamus_mask.HypVINN.nii.gz"

HYPVINN_SEG_NAME = "hypothalamus.HypVINN.nii.gz"

HYPVINN_QC_IMAGE_NAME = "hypothalamus.HypVINN_qc_screenshoot.png"
