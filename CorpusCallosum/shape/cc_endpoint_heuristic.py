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

import lapy
import numpy as np
import scipy.ndimage
import skimage.measure
from scipy.ndimage import label


def smooth_contour(x, y, window_size):
    """
    Smooth a contour using a moving average filter.
    """

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

    # remove padding
    x_smoothed = x_smoothed[window_size // 2:-window_size // 2]
    y_smoothed = y_smoothed[window_size // 2:-window_size // 2]

    return x_smoothed, y_smoothed


def connect_diagonally_connected_components(cc_mask):
    """
    Connects diagonally connected components in the CC mask.
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
    
    for i, j in zip(gap_positions[0], gap_positions[1], strict=True):
        # Temporarily fill this gap
        test_mask = cc_mask.copy()
        test_mask[i, j] = 1
        
        # Check connected components after filling
        _, num_components_after = label(test_mask, structure=structure_4conn)
        
        # Only fill if it actually connects previously disconnected components
        if num_components_after < num_components_before:
            connects_diagonals[i, j] = True
    
    # Fill the identified diagonal gaps that actually improve connectivity
    cc_mask[connects_diagonals] = 1


def extract_cc_contour(cc_mask, contour_smoothing=5):
    """
    Extracts the contour of the CC from the mask.
    """
    # cc_mask_orig = cc_mask
    cc_mask = cc_mask.copy()

    connect_diagonally_connected_components(cc_mask)

    contour = skimage.measure.find_contours(cc_mask, level=0.5)[0].T
    contour = np.array(smooth_contour(contour[0], contour[1], contour_smoothing))

    # plot contour
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1,2,figsize=(10, 8))
    # ax[0].imshow(cc_mask_orig)
    # ax[1].imshow(cc_mask)
    # ax[0].plot(contour[1], contour[0], 'r-')
    # ax[1].plot(contour[1], contour[0], 'r-')
    # plt.show()

    return contour

def get_endpoints(cc_mask, AC_2d, PC_2d, resolution, return_coordinates=True, contour_smoothing=5):
    """
    Determines endpoints of CC by finding the point in the contour closest to
    the anterior and posterior commisure (with some offsets)

    NOTE: Expects LIA orientation
    """
    image_size = cc_mask.shape

    # Calculate angle between AC-PC line and horizontal using numpy
    ac_pc_vector = PC_2d - AC_2d
    horizontal_vector = np.array([0, -20])
    # Calculate angle using dot product formula: cos(theta) = (a·b)/(|a||b|)
    dot_product = np.dot(ac_pc_vector, horizontal_vector)
    norms = np.linalg.norm(ac_pc_vector) * np.linalg.norm(horizontal_vector)
    theta = np.arccos(dot_product / norms)

    # Convert symbolic theta to float and convert from radians to degrees
    theta_degrees = theta * 180 / np.pi
    rotated_cc_mask = scipy.ndimage.rotate(cc_mask, -theta_degrees, order=0, reshape=False)

    contour = extract_cc_contour(rotated_cc_mask, contour_smoothing)

    # rotate points around center
    origin_point = np.array([image_size[0] // 2, image_size[1] // 2])

    # Create rotation matrix for -theta
    rot_matrix = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])

    # Translate points to origin, rotate, then translate back
    pc_centered = PC_2d - origin_point
    ac_centered = AC_2d - origin_point

    rotated_PC_2d = (rot_matrix @ pc_centered) + origin_point
    rotated_AC_2d = (rot_matrix @ ac_centered) + origin_point
    

    # Add z=0 coordinate to make 3D, then remove it after resampling
    contour_3d = np.vstack([contour, np.zeros(contour.shape[1])])
    contour_3d = lapy.tria_mesh.TriaMesh._TriaMesh__resample_polygon(contour_3d.T, 701).T
    contour = contour_3d[:2]

    
    contour = contour[:, :-1]

    rotated_AC_2d = np.array(rotated_AC_2d).astype(float)
    rotated_PC_2d = np.array(rotated_PC_2d).astype(float)

    # move posterior commisure 5 mm posterior
    rotated_PC_2d = rotated_PC_2d + np.array([10 * resolution, -5 * resolution])

    # move anterior commisure 1.5 mm anterior
    rotated_AC_2d = rotated_AC_2d + np.array([0, 5 * resolution])

    # find point in contour closest to AC
    AC_startpoint_idx = np.argmin(np.linalg.norm(contour - rotated_AC_2d[:, None], axis=0))

    # find point in contour closest to PC
    PC_startpoint_idx = np.argmin(np.linalg.norm(contour - rotated_PC_2d[:, None], axis=0))

    # rotate startpoints to original orientation
    # Create rotation matrix
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # rotate contour to original orientation
    contour_rotated = np.zeros_like(contour)

    origin_point = np.array(origin_point).astype(float)
    # Create rotation matrix
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Translate points to origin, rotate, then translate back
    contour_centered = contour - origin_point[:, None]
    contour_rotated = (rot_matrix @ contour_centered) + origin_point[:, None]

    if return_coordinates:
        AC_contour_point = contour[:, AC_startpoint_idx]
        PC_contour_point = contour[:, PC_startpoint_idx]

        # Translate points to origin, rotate, then translate back
        ac_centered = AC_contour_point - origin_point
        pc_centered = PC_contour_point - origin_point

        start_point_A = (rot_matrix @ ac_centered) + origin_point
        start_point_P = (rot_matrix @ pc_centered) + origin_point

        return contour_rotated, start_point_A, start_point_P
    else:
        return contour_rotated, AC_startpoint_idx, PC_startpoint_idx


def get_endpoints_from_nib(cc_label_nib, paths_csv, subj_id, return_coordinates=True):
    cc_mask = cc_label_nib.get_fdata() == 192
    cc_mask = cc_mask[cc_mask.shape[0] // 2]

    posterior_commisure_center = paths_csv.loc[subj_id, "PC_center_r":"PC_center_s"].to_numpy().astype(float)
    anterior_commisure_center = paths_csv.loc[subj_id, "AC_center_r":"AC_center_s"].to_numpy().astype(float)

    # adjust LR from label coordinates to orig_up coordinates
    posterior_commisure_center[0] = 128
    anterior_commisure_center[0] = 128

    # orientation I, A
    # rotate image so anterior and posterior commisure are horizontal
    AC_2d = anterior_commisure_center[1:]
    PC_2d = posterior_commisure_center[1:]

    return get_endpoints(
        cc_mask, AC_2d, PC_2d, resolution=cc_label_nib.header.get_zooms()[1], return_coordinates=return_coordinates
    )

