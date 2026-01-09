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

from CorpusCallosum.utils.types import Points2dType


def compute_curvature(path: Points2dType) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Compute curvature by computing edge angles.

    Parameters
    ----------
    path : np.ndarray
        Array of shape (N, 2) containing path coordinates.

    Returns
    -------
    np.ndarray
        Array of angle differences between consecutive edges.
    """
    # compute curvature by computing edge angles
    edges = np.diff(path, axis=0)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    # compute angle differences between consecutive edges
    angle_diffs = np.diff(angles)
    # wrap angles to [-pi, pi]
    angle_diffs = np.mod(angle_diffs + np.pi, 2 * np.pi) - np.pi
    return angle_diffs


def compute_mean_curvature(path: Points2dType) -> float:
    """Compute mean absolute curvature of a path in degrees.

    Parameters
    ----------
    path : np.ndarray
        Array of shape (N, 2) containing path coordinates.

    Returns
    -------
    float
        Mean absolute curvature of the path in degrees.
    """
    curvature = compute_curvature(path)
    if len(curvature) == 0:
        return 0.0
    return np.mean(np.abs(np.degrees(curvature))).item()


def calculate_curvature_metrics(
    midline: Points2dType,
    split_points: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """
    Calculate curvature metrics for the CC midline, including overall mean,
    body (central 65%), and subsegment curvatures.

    Parameters
    ----------
    midline : Points2dType
        Equidistant points along the midline.
    split_points : np.ndarray
        Points on the midline where it was split (for orthogonal subdivision).

    Returns
    -------
    mean_curvature : float
        Overall mean curvature.
    curvature_body : float
        Mean curvature of the central 65% of the midline.
    curvature_subsegments : np.ndarray
        Mean curvature for each subsegment.
    """
    mean_curvature = compute_mean_curvature(midline)

    num_midline_points = len(midline)
    # central 65% means we remove 17.5% from each end
    start_idx_body = int(num_midline_points * 0.175)
    end_idx_body = int(num_midline_points * 0.825)
    curvature_body = compute_mean_curvature(midline[start_idx_body:end_idx_body])

    # Find split indices on the midline for subsegment curvature
    split_indices_midline = [0]
    for sp in split_points:
        idx = np.argmin(np.linalg.norm(midline - sp, axis=1))
        split_indices_midline.append(idx)

    split_indices_midline.append(len(midline) - 1)
    split_indices_midline.sort()

    _curvature_subsegments = []
    for i in range(len(split_indices_midline) - 1):
        s_idx = split_indices_midline[i]
        e_idx = split_indices_midline[i + 1]
        if e_idx - s_idx >= 2:  # need at least 3 points for curvature
            curv = compute_mean_curvature(midline[s_idx : e_idx + 1])
        else:
            curv = 0.0
        _curvature_subsegments.append(curv)
    curvature_subsegments = np.asarray(_curvature_subsegments)

    return mean_curvature, curvature_body, curvature_subsegments

