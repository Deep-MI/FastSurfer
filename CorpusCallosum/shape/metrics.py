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

import numpy as np

import FastSurferCNN.utils.logging as logging

logger = logging.get_logger(__name__)

def calculate_cc_index(cc_contour: np.ndarray) -> float:
    """Calculate CC index based on three perpendicular measurements.

    Parameters
    ----------
    cc_contour : np.ndarray
        Array of shape (2, N) containing contour points in ACPC space.

    Returns
    -------
    cc_index : float
        The CC index, which is the sum of thicknesses at three measurement points divided by AP length.
    """
    # Get anterior and posterior points
    anterior_idx = np.argmin(cc_contour[0])  # Leftmost point
    posterior_idx = np.argmax(cc_contour[0])  # Rightmost point

    # Get the longest line (anterior to posterior)
    ap_line = cc_contour[:, posterior_idx] - cc_contour[:, anterior_idx]
    ap_length = np.linalg.norm(ap_line)
    ap_unit = np.array([-ap_line[1], ap_line[0]]) / ap_length

    # Get midpoint of AP line
    midpoint = cc_contour[:, anterior_idx] + (ap_line / 2)

    # Get perpendicular direction

    # Get intersection points with contour for each measurement line
    def get_intersections(start_point: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Find intersection points between a line and the contour.

        Parameters
        ----------
        start_point : np.ndarray
            Starting point of the line, shape (2,).
        direction : np.ndarray
            Direction vector of the line, shape (2,).

        Returns
        -------
        np.ndarray
            Array of shape (N, 2) containing intersection points.
        """
        # Get all points above and below the line
        points = cc_contour.T - start_point[None, :]
        dots = np.dot(points, direction)
        signs = np.sign(dots)
        sign_changes = np.where(np.diff(signs))[0]

        # Linear interpolation between points
        t = -dots[sign_changes] / (dots[sign_changes + 1] - dots[sign_changes])
        return cc_contour[:, sign_changes] + t * (cc_contour[:, sign_changes + 1] - cc_contour[:, sign_changes])

    # Get three measurements
    most_anterior_pt = cc_contour[:, anterior_idx]
    perpendicular_unit = np.array([-ap_unit[1], ap_unit[0]])

    anterior_intersections = get_intersections(most_anterior_pt - 10 * perpendicular_unit, ap_unit)

    # sort by x
    anterior_intersections = anterior_intersections[np.argsort(anterior_intersections[:, 0])]

    middle_ints = get_intersections(midpoint, perpendicular_unit)

    if len(middle_ints) != 2:
        logger.warning(
            f"The perpendicular line should intersect the contour twice, "
            f"but it intersects {len(middle_ints)} times"
        )

    # plt.close()

    # calculate index
    ap_distance = np.linalg.norm(anterior_intersections[0] - anterior_intersections[-1])
    anterior_distance = np.linalg.norm(anterior_intersections[0] - anterior_intersections[1])
    posterior_distance = np.linalg.norm(anterior_intersections[-1] - anterior_intersections[-2])
    top_distance = np.linalg.norm(middle_ints[0] - middle_ints[1])

    cc_index = (anterior_distance + posterior_distance + top_distance) / ap_distance


    return cc_index
