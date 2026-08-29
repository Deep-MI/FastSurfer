# Copyright 2025 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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

import json

import numpy as np

from CorpusCallosum.data import constants
from CorpusCallosum.shape.contour import CCContour


def load_fsaverage_cc_template() -> CCContour:
    """Load the bundled fsaverage corpus callosum contour.

    The contour was precomputed from FreeSurfer 7.4.1's fsaverage
    ``aparc+aseg.mgz`` and is stored alongside the fsaverage centroid target.
    Loading it does not require a FreeSurfer installation.

    Returns
    -------
    CCContour
        The precomputed contour, endpoint indices, and zero slice position.
    """
    with open(constants.FSAVERAGE_CC_CONTOUR_PATH) as file:
        data = json.load(file)

    points = np.asarray(data["points"], dtype=np.float64)
    anterior_endpoint_idx, posterior_endpoint_idx = map(int, data["endpoint_idxs"])
    return CCContour(
        points,
        None,
        endpoint_idxs=(anterior_endpoint_idx, posterior_endpoint_idx),
        z_position=float(data["z_position"]),
    )
