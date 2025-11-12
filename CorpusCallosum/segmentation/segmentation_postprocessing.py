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
from numpy import typing as npt
from scipy import integrate, ndimage
from scipy.spatial.distance import cdist
from skimage.measure import label

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.data.constants import CC_LABEL, FORNIX_LABEL

logger = logging.get_logger(__name__)


def find_component_boundaries(labels_arr: npt.NDArray[int], component_id: int) -> npt.NDArray[int]:
    """Find boundary voxels of a connected component.

    Parameters
    ----------
    labels_arr : np.ndarray
        Labeled array from connected components analysis
    component_id : int
        ID of the component to find boundaries for

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing boundary coordinates

    Notes
    -----
    Uses 6-connectivity (face neighbors only) to determine boundaries.
    Boundary voxels are those that are part of the component but have
    at least one non-component neighbor.
    """
    component_mask = labels_arr == component_id
    
    # Create a structuring element for 6-connectivity (face neighbors only)
    struct = ndimage.generate_binary_structure(3, 1)
    
    # Erode the component to find internal voxels
    eroded = ndimage.binary_erosion(component_mask, structure=struct)
    
    # Boundary is the difference between original and eroded
    boundary = component_mask & ~eroded
    
    return np.array(np.where(boundary)).T


def find_minimal_connection_path(
    boundary_coords1: np.ndarray, 
    boundary_coords2: np.ndarray, 
    max_distance: float = 3.0
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the minimal connection path between two component boundaries.

    Parameters
    ----------
    boundary_coords1 : np.ndarray
        Boundary coordinates of first component, shape (N1, 3)
    boundary_coords2 : np.ndarray
        Boundary coordinates of second component, shape (N2, 3)
    max_distance : float, default=3.0
        Maximum distance to consider for connection, by default 3.0

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or None
        If a valid connection is found:
            - point1 : Coordinates on first boundary
            - point2 : Coordinates on second boundary
        None if no connection within max_distance is found

    Notes
    -----
    Uses Euclidean distance to find the closest pair of points
    between the two boundaries.
    """
    if len(boundary_coords1) == 0 or len(boundary_coords2) == 0:
        return None
    
    # Calculate pairwise distances between all boundary points
    distances = cdist(boundary_coords1, boundary_coords2, metric='euclidean')
    
    # Find the minimum distance and corresponding points
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    min_distance = distances[min_idx]
    
    if min_distance <= max_distance:
        return boundary_coords1[min_idx[0]], boundary_coords2[min_idx[1]]
    
    return None


def create_connection_line(point1: np.ndarray, point2: np.ndarray) -> list[tuple[int, int, int]]:
    """Create a line of voxels connecting two points.

    Uses a simplified 3D line algorithm to create a sequence of voxels
    that form a continuous path between the two points.

    Parameters
    ----------
    point1 : np.ndarray
        Starting point coordinates, shape (3,)
    point2 : np.ndarray
        Ending point coordinates, shape (3,)

    Returns
    -------
    list[tuple[int, int, int]]
        List of (x, y, z) coordinates forming the connection line

    Notes
    -----
    The line is created by interpolating between the points using
    the maximum distance in any dimension as the number of steps.
    """
    x1, y1, z1 = map(int, point1)
    x2, y2, z2 = map(int, point2)
    
    line_points = []
    
    # Calculate the number of steps needed
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    
    steps = max(dx, dy, dz)
    
    if steps == 0:
        return [(x1, y1, z1)]
    
    # Calculate increments for each dimension
    x_inc = (x2 - x1) / steps
    y_inc = (y2 - y1) / steps
    z_inc = (z2 - z1) / steps
    
    # Generate points along the line
    for i in range(steps + 1):
        x = int(round(x1 + i * x_inc))
        y = int(round(y1 + i * y_inc))
        z = int(round(z1 + i * z_inc))
        line_points.append((x, y, z))
    
    return line_points


def connect_nearby_components(seg_arr: np.ndarray, max_connection_distance: float = 3.0) -> np.ndarray:
    """Connect nearby disconnected components that should be connected.

    This function identifies disconnected components in the segmentation and creates
    minimal connections between components that are close to each other.

    Parameters
    ----------
    seg_arr : np.ndarray
        Input binary segmentation array
    max_connection_distance : float, optional
        Maximum distance to connect components, by default 3.0

    Returns
    -------
    np.ndarray
        Segmentation array with minimal connections added between nearby components

    Notes
    -----
    The function:
    1. Identifies connected components in the input segmentation
    2. Finds boundaries between components
    3. Creates minimal connections between nearby components
    4. Returns the modified segmentation with added connections
    """

    # Create a copy to modify
    connected_seg = seg_arr.copy()
    
    # Find connected components without dilation first
    labels_cc = label(seg_arr, connectivity=3, background=0)
    
    # Get component sizes (excluding background)
    bincount = np.bincount(labels_cc.flat)
    component_ids = np.where(bincount > 0)[0][1:]  # Exclude background (0)
    
    if len(component_ids) <= 1:
        return connected_seg  # Only one component, no connections needed
    
    # Sort components by size (largest first)
    component_sizes = [(comp_id, bincount[comp_id]) for comp_id in component_ids]
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Use the largest component as the reference
    main_component_id = component_sizes[0][0]


    
    logger.info(f"Found {len(component_ids)} disconnected components. "
                f"Attempting to connect smaller components to main component (size: {component_sizes[0][1]})")
    
    connections_made = 0
    
    # Try to connect each smaller component to the main component
    for comp_id, comp_size in component_sizes[1:]:
        if comp_size < 5:  # Skip very small components (likely noise)
            logger.debug(f"Skipping tiny component {comp_id} with size {comp_size}")
            continue
            
        # Find boundaries of both components
        main_boundary = find_component_boundaries(labels_cc, main_component_id)
        comp_boundary = find_component_boundaries(labels_cc, comp_id)
        
        # Find minimal connection path
        connection = find_minimal_connection_path(main_boundary, comp_boundary, max_connection_distance)
        
        if connection is not None:
            point1, point2 = connection
            distance = np.linalg.norm(point2 - point1)
            
            logger.debug(f"Connecting component {comp_id} (size: {comp_size}) to main component. "
                        f"Distance: {distance:.2f} voxels")
            
            # Create connection line
            connection_line = create_connection_line(point1, point2)
            
            # Add connection voxels to the segmentation
            # Use the same label as the original segmentation at the connection points
            connection_label = seg_arr[point1[0], point1[1], point1[2]] if \
                seg_arr[point1[0], point1[1], point1[2]] != 0 else \
                    seg_arr[point2[0], point2[1], point2[2]]
            
            for x, y, z in connection_line:
                if (0 <= x < connected_seg.shape[0] and 
                    0 <= y < connected_seg.shape[1] and 
                    0 <= z < connected_seg.shape[2]):
                    if connected_seg[x, y, z] == 0:  # Only fill empty voxels
                        connected_seg[x, y, z] = connection_label
            
            connections_made += 1
        else:
            logger.debug(f"Component {comp_id} (size: {comp_size}) too far from main component")
    
    logger.info(f"Created {connections_made} minimal connections between components")


    # Plot components for visualization
    # import matplotlib.pyplot as plt
    # n_components = len(component_sizes)
    # fig, axes = plt.subplots(1, n_components + 1, figsize=(5*(n_components + 1), 5))
    # if n_components == 1:
    #     axes = [axes]
    # # Plot each component in a different color
    # for i, (comp_id, comp_size) in enumerate(component_sizes):
    #     component_mask = labels_cc == comp_id
    #     axes[i].imshow(component_mask[component_mask.shape[0]//2], cmap='gray')
    #     axes[i].set_title(f'Component {comp_id}\nSize: {comp_size}')
    #     axes[i].axis('off')
    
    # # Plot the connected segmentation
    # axes[-1].imshow(connected_seg[connected_seg.shape[0]//2], cmap='gray')
    # axes[-1].set_title('Connected Segmentation')
    # axes[-1].axis('off')
    # plt.tight_layout()
    # plt.show()
    
    return connected_seg


def get_cc_volume_voxel(
    desired_width_mm: int,
    cc_mask: np.ndarray,
    voxel_size: tuple[float, float, float]
) -> float:
    """Calculate the volume of the corpus callosum in cubic millimeters.

    This function calculates the volume of the corpus callosum (CC) in cubic millimeters.
    If the CC width is larger than desired_width_mm, the voxels on the edges are calculated as
    partial volumes to achieve the desired width.

    Parameters
    ----------
    desired_width_mm : int
        Desired width of the CC in millimeters
    cc_mask : np.ndarray
        Binary mask of the corpus callosum
    voxel_size : tuple[float, float, float]
        Voxel size in millimeters (x, y, z)

    Returns
    -------
    float
        Volume of the CC in cubic millimeters

    Raises
    ------
    ValueError
        If CC width is smaller than desired width
    AssertionError
        If CC mask doesn't have odd number of voxels in x dimension

    Notes
    -----
    The function assumes LIA orientation where:
    - x dimension corresponds to Left/Right
    - y dimension corresponds to Inferior/Superior
    - z dimension corresponds to Anterior/Posterior
    """
    assert cc_mask.shape[0] % 2 == 1, "CC mask must have odd number of voxels in x dimension"


    # Calculate voxel volume
    voxel_volume = np.prod(voxel_size)

    # Get width of CC mask in voxels by finding the extent in x dimension
    width_vox = np.sum(np.any(cc_mask, axis=(1,2)))

    # we are in LIA, so 0 is L/R resolution
    width_mm = width_vox * voxel_size[0]

    if width_mm == desired_width_mm:
        return np.sum(cc_mask) * voxel_volume
    elif width_mm > desired_width_mm:
        # remainder on the left/right side of the CC mask
        desired_width_vox = desired_width_mm / voxel_size[0]
        fraction_of_voxel_at_edge = (desired_width_vox % 1) / 2

        if fraction_of_voxel_at_edge > 0:
            desired_width_vox = int(np.floor(desired_width_vox) + 1)
            desired_width_vox = desired_width_vox + 1 if desired_width_vox % 2 == 0 else desired_width_vox

            assert cc_mask.shape[0] == desired_width_vox, (f"CC mask should have {desired_width_vox} voxels, "
                                                          f"but has {cc_mask.shape[0]}")

        left_partial_volume = np.sum(cc_mask[0]) * voxel_volume * fraction_of_voxel_at_edge
        right_partial_volume = np.sum(cc_mask[-1]) * voxel_volume * fraction_of_voxel_at_edge
        center_volume = np.sum(cc_mask[1:-1]) * voxel_volume
        return left_partial_volume + right_partial_volume + center_volume
    else:
        raise ValueError(f"Width of CC segmentation is smaller than desired width: {width_mm} < {desired_width_mm}")

def get_cc_volume_contour(cc_contours: list[np.ndarray], 
                         voxel_size: tuple[float, float, float]) -> float:
    """Calculate the volume of the corpus callosum using Simpson's rule.

    Parameters
    ----------
    desired_width_mm : int
        Desired width of the CC in millimeters
    cc_contours : list[np.ndarray]
        List of CC contours for each slice in the left-right direction
    voxel_size : tuple[float, float, float]
        Voxel size in millimeters (x, y, z)

    Returns
    -------
    float
        Volume of the CC in cubic millimeters

    Raises
    ------
    ValueError
        If CC width is smaller than desired width or insufficient contours for Simpson's rule

    Notes
    -----
    This function calculates the volume of the corpus callosum (CC) in cubic millimeters 
    using Simpson's rule. If the CC width is larger than desired_width_mm, the voxels on 
    the edges are calculated as partial volumes to achieve the desired width.
    """
    if len(cc_contours) < 3:
        raise ValueError("Need at least 3 contours for Simpson's rule integration")
    
    # Calculate cross-sectional areas for each contour
    areas = []
    
    for contour in cc_contours:
        contour = contour.copy()
        assert voxel_size[1] == voxel_size[2], "volume must be isotropic"
        contour *= voxel_size[1]
        # Calculate area using the shoelace formula for polygon area
        if contour.shape[1] < 3:
            areas.append(0.0)
        else:
            x = contour[0]
            y = contour[1]
            # Shoelace formula: A = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
            area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
            # Convert from voxel^2 to mm^2
            area_mm2 = area * voxel_size[1] * voxel_size[2]  # y * z voxel dimensions
            areas.append(area_mm2)
    
    areas = np.array(areas)
    
    # Calculate spacing between slices (left-right direction)
    lr_spacing = voxel_size[0]  # x-direction voxel size

    measurement_points = np.arange(-voxel_size[0]*(areas.shape[0]//2), 
                                    voxel_size[0]*((areas.shape[0]+1)//2), lr_spacing)
    
    # interpolate areas at 0.25 and 5
    areas_interpolated = np.interp(x=[-2.5, 2.5], 
                                   xp=measurement_points, 
                                   fp=areas)


    # remove measurement points that are outside of the desired range
    # not sure if this can happen, but let's be safe
    outside_range = (measurement_points < -2.5) | (measurement_points > 2.5)
    measurement_points = [-2.5] + measurement_points[~outside_range].tolist() + [2.5]
    areas = [areas_interpolated[0]] + areas[~outside_range].tolist() + [areas_interpolated[1]]
    
    
    # can also use trapezoidal rule
    return integrate.simpson(areas, x=measurement_points)
    

def get_largest_cc(
    seg_arr: np.ndarray,
    max_connection_distance: float = 3.0
) -> np.ndarray:
    """Get largest connected component from a binary segmentation array.

    Parameters
    ----------
    seg_arr : np.ndarray
        Input binary segmentation array
    max_connection_distance : float, optional
        Maximum distance to connect components, by default 3.0

    Returns
    -------
    np.ndarray
        Binary mask of the largest connected component

    Notes
    -----
    The function first attempts to connect nearby disconnected components
    that should be connected, then finds the largest connected component.
    It uses minimal connections between close components before falling
    back to dilation if no connections are made.
    """
    # First attempt: try to connect nearby components with minimal connections
    connected_seg = connect_nearby_components(seg_arr, max_connection_distance)
    
    # Check if connections were successful by comparing connectivity
    original_labels = label(seg_arr, connectivity=3, background=0)
    connected_labels = label(connected_seg, connectivity=3, background=0)
    
    original_components = len(np.unique(original_labels)) - 1  # Exclude background
    connected_components = len(np.unique(connected_labels)) - 1  # Exclude background
    
    if connected_components < original_components:
        logger.info(f"Successfully reduced components from {original_components} to {connected_components} "
                     "using minimal connections")
    mask = connected_seg
    # else:
    #     logger.info("No connections made, falling back to dilation approach")
    #     # Fallback: use the original dilation approach
    #     struct1 = ndimage.generate_binary_structure(3, 3)
    #     mask = ndimage.binary_dilation(seg_arr, structure=struct1, iterations=1).astype(np.uint8)
    
    # Get connected components from the processed mask
    labels_cc = label(mask, connectivity=3, background=0)
    
    # Get component counts
    bincount = np.bincount(labels_cc.flat)
    
    # Get background label (assumed to be the largest component)
    background = np.argmax(bincount)
    bincount[background] = -1
    
    # Get largest connected component
    largest_cc = labels_cc == np.argmax(bincount)

    return largest_cc

def clean_cc_segmentation(
    seg_arr: np.ndarray,
    max_connection_distance: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Clean corpus callosum segmentation by removing non-connected components.

    Parameters
    ----------
    seg_arr : np.ndarray
        Input segmentation array with CC (192) and fornix (250) labels
    max_connection_distance : float, optional
        Maximum distance to connect components, by default 3.0

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - clean_seg : Cleaned segmentation array with only the largest
          connected component of CC and fornix
        - mask : Binary mask of the largest connected component

    Notes
    -----
    The function:
    1. Isolates the CC (label 192)
    2. Attempts to connect nearby disconnected components
    3. Adds the fornix (label 250)
    4. Removes non-connected components from the combined CC and fornix
    """
    # Remove non connected components from the CC alone, with minimal connections
    cc_seg = np.zeros_like(seg_arr)
    cc_seg[seg_arr == CC_LABEL] = CC_LABEL

    cc_label_cleaned = np.zeros_like(cc_seg)
    for i in range(cc_seg.shape[0]):
        cc_label_cleaned[i] = get_largest_cc(cc_seg[None,i], max_connection_distance)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1,3)
        # ax[0].imshow(cc_seg[i])
        # ax[1].imshow(mask[i])
        # ax[2].imshow(cc_seg[i] - mask[i]*CC_LABEL) # difference between pre and post clean
        # plt.show()


    # Add fornix to the CC labels
    clean_seg = np.zeros_like(seg_arr)
    clean_seg[cc_label_cleaned > 0] = CC_LABEL
    clean_seg[seg_arr == FORNIX_LABEL] = FORNIX_LABEL

    return clean_seg, cc_label_cleaned > 0
