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
import skimage.measure
from scipy.ndimage import label

from CorpusCallosum.utils.types import Points2dType, Polygon2dType
from FastSurferCNN.utils import Image2d, Mask2d, Vector2d


def smooth_contour(x: np.ndarray, y: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a contour using a moving average filter.

    Parameters
    ----------
    x : np.ndarray
        X-coordinates of the contour points.
    y : np.ndarray
        Y-coordinates of the contour points.
    window_size : int
        Size of the smoothing window. Must be odd and > 2.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Smoothed x and y coordinates of the contour.
    """
    # Ensure window_size is an integer
    window_size = int(window_size)

    if window_size // 2 == 0:
        raise ValueError(f"Smoothing window size of {window_size} is too small")

    # Ensure the window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Create a padded version of the arrays to handle the edges
    x_padded = np.pad(x, (window_size // 2, window_size // 2), mode="wrap")
    y_padded = np.pad(y, (window_size // 2, window_size // 2), mode="wrap")

    # Apply moving average
    x_smoothed = np.zeros_like(x)
    y_smoothed = np.zeros_like(y)

    for i in range(len(x)):
        x_smoothed[i] = np.mean(x_padded[i : i + window_size])
        y_smoothed[i] = np.mean(y_padded[i : i + window_size])

    return x_smoothed, y_smoothed


def connect_diagonally_connected_components(cc_mask: Image2d) -> Image2d:
    """Connect diagonally connected components in the CC mask.

    Parameters
    ----------
    cc_mask : np.ndarray
        Binary mask of the corpus callosum.

    Notes
    -----
    Modifies the input mask in-place to connect diagonally connected components.
    """
    
    # Create padded mask to handle boundary conditions
    padded_mask = np.pad(cc_mask, pad_width=1, mode='constant', constant_values=0)
    
    # Get center pixels and diagonal neighbors
    center = padded_mask[1:-1, 1:-1]
    
    # Direct neighbors (4-connectivity)
    left = padded_mask[1:-1, :-2]      # left
    right = padded_mask[1:-1, 2:]      # right  
    up = padded_mask[:-2, 1:-1]        # up
    down = padded_mask[2:, 1:-1]       # down
    
    # Diagonal neighbors
    up_left = padded_mask[:-2, :-2]     # up-left
    up_right = padded_mask[:-2, 2:]     # up-right
    down_left = padded_mask[2:, :-2]    # down-left
    down_right = padded_mask[2:, 2:]    # down-right
    
    potential_diagonal_gaps = (center == 0) & (
        ((up_left > 0) & ((right > 0) | (down > 0))) |
        ((up_right > 0) & ((left > 0) | (down > 0))) |
        ((down_left > 0) & ((right > 0) | (up > 0))) |
        ((down_right > 0) & ((left > 0) | (up > 0)))
    )

    # Get connected components before filling using 4-connectivity
    # This way, diagonal-only connections are treated as separate components
    structure_4conn = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    _, num_components_before = label(cc_mask, structure=structure_4conn)
    
    # For each potential gap, check if filling it would reduce the number of components
    connects_diagonals = np.zeros_like(potential_diagonal_gaps)
    gap_positions = np.where(potential_diagonal_gaps)

    if len(gap_positions[0]) > 0:
        test_mask = cc_mask.copy()
        # Fill all gap voxels, that by themselves would connect 2 components
        for i, j in zip(gap_positions[0], gap_positions[1], strict=True):
            # Temporarily fill this gap
            test_mask[i, j] = 1
            # Check connected components after filling, this is relatively slow...
            _, num_components_after = label(test_mask, structure=structure_4conn)
            # Only fill if it actually connects previously disconnected components
            if num_components_after < num_components_before:
                connects_diagonals[i, j] = True
            # Revert temporary fill
            test_mask[i, j] = cc_mask[i, j]
    
    # Fill the identified diagonal gaps that actually improve connectivity
    return np.where(connects_diagonals, 1, cc_mask)


def extract_cc_contour(cc_mask: Mask2d, contour_smoothing: int = 5) -> Polygon2dType:
    """Extract the contour of the CC from the mask using a marching squares approach.

    Parameters
    ----------
    cc_mask : np.ndarray
        Binary mask of the corpus callosum.
    contour_smoothing : int, default=5
        Window size for contour smoothing.

    Returns
    -------
    lapy.Polygon
        A lapy Polygon object with a closed polygon contour.
    """
    cc_mask = connect_diagonally_connected_components(cc_mask)

    contour = skimage.measure.find_contours(cc_mask, level=0.5)[0].T
    contour = np.array(smooth_contour(contour[0], contour[1], contour_smoothing))

    return contour


def find_cc_endpoints(
        contour: Points2dType,
        ac_2d: Vector2d,
        pc_2d: Vector2d,
        plot: bool = False,
):
    """Extracts the contour of the CC, rotates to AC-PC alignment, and determines closest points of CC to AC and PC.

    Parameters
    ----------
    contour : np.ndarray of shape (2, N)
        Points of the CC contour in AS (millimeter).
    ac_2d : np.ndarray of shape (2,) and type float
        2D AS coordinates of the anterior commissure in millimeter.
    pc_2d : np.ndarray of shape (2,) and type float
        2D AS coordinates of the posterior commissure in millimeter.

    Returns
    -------
    anterior_posterior_point_indices : pair of ints
        Indices of anterior and posterior points in the contour.
    
    Notes
    -----
    Expects AS orientation of contour, ac_2d, and pc_2d.
    """
    if contour.shape[0] != 2:
        raise ValueError(f"contour must have shape (2, N), got {contour.shape}")
    if any(p2d.shape != (2,) for p2d in (ac_2d, pc_2d)):
        raise ValueError(f"ac_2d and pc_2d must have shape (2,), got {ac_2d.shape} and {pc_2d.shape}")

    # Calculate angle between AC-PC line and horizontal using numpy
    ac_pc_vector = pc_2d - ac_2d
    horizontal_vector = np.array([-20, 0])
    # Calculate angle using dot product formula: cos(theta) = (a·b)/(|a||b|)
    dot_product = np.dot(ac_pc_vector, horizontal_vector)
    norms = np.linalg.norm(ac_pc_vector) * np.linalg.norm(horizontal_vector)
    # The sign of theta is the inverse of ac_pc_vector [ X ]
    theta = np.sign(ac_pc_vector[0]) * np.arccos(dot_product / norms)

    rot_matrix_inv = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # move posterior commisure 5 mm posterior, 10 mm inferior
    as_offset_pc = np.array([-5, -10], dtype=float)
    posterior_anchor_2d = pc_2d.astype(float) + rot_matrix_inv @ as_offset_pc
    # move anterior commisure 5 mm anterior
    as_offset_ac = np.array([5, 0], dtype=float)
    anterior_anchor_2d = ac_2d.astype(float) + rot_matrix_inv @ as_offset_ac

    # Find the endpoints of the CC shape relative to AC and PC coordinates
    # find point in contour closest to AC
    ac_startpoint_idx = np.argmin(np.linalg.norm(contour - anterior_anchor_2d[:, None], axis=0))
    # find point in contour closest to PC
    pc_startpoint_idx = np.argmin(np.linalg.norm(contour - posterior_anchor_2d[:, None], axis=0))

    if plot: # interactive debug plot of contour, ac, pc and endpoints
        import matplotlib.pyplot as plt

        from FastSurferCNN.utils.plotting import backend
        with backend("qtagg"):
            plt.figure(figsize=(10, 8))
            plt.plot(contour[0, :], contour[1, :], 'b-', label='CC Contour', linewidth=2)
            plt.plot(*ac_2d[0:2], 'go', markersize=10, label='AC')
            plt.plot(*pc_2d[0:2], 'ro', markersize=10, label='PC')
            plt.plot(*anterior_anchor_2d[0:2], 'g^', markersize=10, label='Anterior Anchor')
            plt.plot(*posterior_anchor_2d[0:2], 'r^', markersize=10, label='Posterior Anchor')
            plt.plot(*contour[0:2, ac_startpoint_idx], 'g*', markersize=15, label='AC Endpoint')
            plt.plot(*contour[0:2, pc_startpoint_idx], 'r*', markersize=15, label='PC Endpoint')
            plt.xlabel('A-S (mm)')
            plt.ylabel('I-S (mm)')
            plt.title('CC Contour with Endpoints')
            plt.legend()
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.show()

    return ac_startpoint_idx, pc_startpoint_idx
