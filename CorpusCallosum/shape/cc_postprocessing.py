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

from pathlib import Path

import numpy as np

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.data.constants import FSAVERAGE_MIDDLE, SUBSEGEMNT_LABELS
from CorpusCallosum.data.read_write import run_in_background
from CorpusCallosum.shape.cc_endpoint_heuristic import get_endpoints
from CorpusCallosum.shape.cc_mesh import CC_Mesh
from CorpusCallosum.shape.cc_metrics import calculate_cc_index
from CorpusCallosum.shape.cc_subsegment_contour import (
    get_primary_eigenvector,
    hampel_subdivide_contour,
    subdivide_contour,
    subsegment_midline_orthogonal,
    transform_to_acpc_standard,
)
from CorpusCallosum.shape.cc_thickness import cc_thickness, convert_to_ras
from CorpusCallosum.utils.utils import HiddenPrints
from CorpusCallosum.visualization.visualization import plot_contours

logger = logging.get_logger(__name__)


# assert LIA orientation
LIA_ORIENTATION = np.zeros((3,3))
LIA_ORIENTATION[0,0] = -1
LIA_ORIENTATION[1,2] = 1
LIA_ORIENTATION[2,1] = -1


def create_visualization(subdivision_method, result, midslices_data, output_image_path, 
                         ac_coords, pc_coords, vox_size, title_suffix=""):
    """Helper function to create visualization plots based on subdivision method.
    
    Args:
        subdivision_method: The subdivision method being used
        result: Dictionary containing processing results with split_contours and split_contours_hofer_frahm
        midslices_data: Slice data for visualization
        output_subdir: Directory to save visualization
        ac_coords: AC coordinates
        pc_coords: PC coordinates
        title_suffix: Additional text to append to the title
    
    Returns:
        Process object for background execution
    """
    title = f'CC Subsegmentation by {subdivision_method} {title_suffix}'

    args_dict = {
        'debug': False,
        'transformed': midslices_data,
        'split_contours': None,
        'split_contours_hofer_frahm': None,
        'midline_equidistant': result['midline_equidistant'],
        'levelpaths': result['levelpaths'],
        'output_path': output_image_path,
        'ac_coords': ac_coords,
        'pc_coords': pc_coords,
        'vox_size': vox_size,
        'title': title,
    }

    if subdivision_method == "shape":
        args_dict['split_contours'] = result['split_contours']
    else:
        args_dict['split_contours_hofer_frahm'] = result['split_contours_hofer_frahm']

    return run_in_background(plot_contours, **args_dict)



def create_slice_affine(temp_seg_affine, slice_idx, fsaverage_middle):
    """Create slice-specific affine transformation matrix.
    
    Adjusts the input affine transformation matrix for a specific slice by updating
    the translation component based on the slice index and fsaverage middle reference.
    
    Args:
        temp_seg_affine (np.ndarray): Base 4x4 affine transformation matrix
        slice_idx (int): Index of the slice to transform
        fsaverage_middle (int): Reference middle slice index in fsaverage space
        
    Returns:
        np.ndarray: Modified 4x4 affine transformation matrix for the specific slice
    """
    slice_affine = temp_seg_affine.copy()
    slice_affine[0, 3] = -fsaverage_middle + slice_idx
    return slice_affine


def process_slice(segmentation, slice_idx, ac_coords, pc_coords, affine, num_thickness_points, subdivisions, 
                  subdivision_method, contour_smoothing):
    """Process a single slice for corpus callosum measurements.
    
    Performs detailed analysis of a corpus callosum slice, including:
    - Contour extraction and endpoint detection
    - Thickness profile calculation
    - Area and perimeter measurements
    - Shape-based metrics (circularity, CC index)
    - Subdivision into anatomical regions
    
    Args:
        segmentation (np.ndarray): 
            3D segmentation array
        slice_idx (int): 
            Index of the slice to process
        ac_coords (np.ndarray): 
            Anterior commissure coordinates
        pc_coords (np.ndarray): 
            Posterior commissure coordinates
        affine (np.ndarray): 
            4x4 affine transformation matrix
        num_thickness_points (int): 
            Number of points for thickness estimation
        subdivisions (list[float]): 
            List of fractions for anatomical subdivisions
        subdivision_method (str): 
            Method for contour subdivision ('shape', 'vertical', 
            'angular', or 'eigenvector')
        contour_smoothing (float): 
            Gaussian sigma for contour smoothing
        
    Returns:
        slice_data (dict | None): 
            Dictionary containing measurements if successful, including:
        
            - cc_index: Corpus callosum shape index
            - circularity: Shape circularity measure
            - areas: Areas of subdivided regions
            - midline_length: Length along the midline
            - thickness: Array of thickness measurements
            - curvature: Array of curvature measurements
            - thickness_profile: Thickness measurements along the contour
            - total_area: Total area of the CC
            - total_perimeter: Total perimeter length
            - split_contours: Subdivided contour segments
            - split_contours_hofer_frahm: Alternative subdivision (if applicable)
            - midline_equidistant: Equidistant points along midline
            - levelpaths: Paths for thickness measurements
            - thickness_measurement_points: Points where thickness was measured
            - slice_index: Index of the processed slice
            
            Returns None if no CC is found in the slice.
            
    """

    cc_mask_slice = segmentation[slice_idx] == 192
    if not np.any(cc_mask_slice):
        raise ValueError(f'No CC found in slice {slice_idx}')
        

    contour, anterior_endpoint_idx, posterior_endpoint_idx = get_endpoints(cc_mask_slice, ac_coords, pc_coords, 
                                                                           affine.diagonal()[1], 
                                                                           return_coordinates=False, 
                                                                           contour_smoothing=contour_smoothing)
    contour_1mm = convert_to_ras(contour, affine)

    (midline_length, thickness, curvature, midline_equidistant, levelpaths,
     contour_with_thickness, anterior_endpoint_idx, posterior_endpoint_idx) = cc_thickness(contour_1mm.T, 
                                                                                           anterior_endpoint_idx, 
                                                                                           posterior_endpoint_idx, 
                                                                                           n_points=num_thickness_points)
    
    thickness_profile = [
        np.sum(np.sqrt(np.diff(np.array(levelpath[:,:2]), axis=0)**2), axis=0)
        for levelpath in levelpaths
    ]
    thickness_profile = np.linalg.norm(np.array(thickness_profile),axis=1)

    contour_acpc, ac_pt_acpc, pc_pt_acpc, rotate_back_acpc = transform_to_acpc_standard(contour_1mm, 
                                                                                        contour_1mm[:,anterior_endpoint_idx], 
                                                                                        contour_1mm[:,posterior_endpoint_idx])
    cc_index = calculate_cc_index(contour_acpc)

    # Apply different subdivision methods based on user choice
    if subdivision_method == "shape":
        areas, split_contours = subsegment_midline_orthogonal(midline_equidistant, subdivisions, 
                                                              contour_1mm, plot=False)
        split_contours = [transform_to_acpc_standard(split_contour, 
                                                     contour_1mm[:,anterior_endpoint_idx], 
                                                     contour_1mm[:,posterior_endpoint_idx])[0] 
                                                     for split_contour in split_contours]
        
        split_contours_hofer_frahm = None
    elif subdivision_method == "vertical":
        areas, split_contours = subdivide_contour(contour_acpc, subdivisions, plot=False)
        split_contours_hofer_frahm = split_contours.copy()
    elif subdivision_method == "angular":
        if not np.allclose(np.diff(subdivisions), np.diff(subdivisions)[0]):
            logger.error('Error: Angular subdivision method (Hampel) only supports equidistant subdivision, '
                  f'but got: {subdivisions}')
            return None
        areas, split_contours = hampel_subdivide_contour(contour_acpc, num_rays=len(subdivisions), plot=False)       
        split_contours_hofer_frahm = split_contours.copy()
    elif subdivision_method == "eigenvector":
        pt0, pt1 = get_primary_eigenvector(contour_acpc)
        contour_eigen, _, _, rotate_back_eigen = transform_to_acpc_standard(contour_acpc, pt0, pt1)
        ac_pt_eigen, _, _, _ = transform_to_acpc_standard(ac_pt_acpc[:, None], pt0, pt1)
        ac_pt_eigen = ac_pt_eigen[:, 0]
        areas, split_contours = subdivide_contour(contour_eigen, subdivisions, oriented=True, hline_anchor=ac_pt_eigen)
        split_contours = [rotate_back_eigen(split_contour) for split_contour in split_contours]
        split_contours_hofer_frahm = split_contours.copy()

    total_area = np.sum(areas)
    total_perimeter = np.sum(np.sqrt(np.sum((np.diff(contour_1mm, axis=0))**2, axis=1)))
    circularity = 4 * np.pi * total_area / (total_perimeter**2)

    # Transform split contours back to original space
    split_contours = [rotate_back_acpc(split_contour) for split_contour in split_contours]
    if split_contours_hofer_frahm is not None:
        split_contours_hofer_frahm = [rotate_back_acpc(split_contour) for split_contour in split_contours_hofer_frahm]

    return {
        'cc_index': cc_index,
        'circularity': circularity,
        'areas': areas,
        'midline_length': midline_length,
        'thickness': thickness,
        'curvature': curvature,
        'thickness_profile': thickness_profile,
        'total_area': total_area,
        'total_perimeter': total_perimeter,
        'split_contours': split_contours,
        'split_contours_hofer_frahm': split_contours_hofer_frahm,
        'midline_equidistant': midline_equidistant,
        'levelpaths': levelpaths,
        'slice_index': slice_idx
    }, contour_with_thickness, anterior_endpoint_idx, posterior_endpoint_idx


def process_slices(segmentation, slice_selection, temp_seg_affine, midslices, ac_coords, pc_coords, 
                  num_thickness_points, subdivisions, subdivision_method, contour_smoothing, 
                  debug_image_path=None, one_debug_image=False,
                  thickness_image_path=None, vox_size=None, 
                  save_template=None, surf_file_path=None, overlay_file_path=None, cc_html_path=None, 
                  vtk_file_path=None, verbose=False):
    """Process corpus callosum slices based on selection mode.
    
    Handles the processing of either a single middle slice, all slices, or a specific slice,
    including affine transformations and measurements for each slice.
    
    Args:
        segmentation (np.ndarray): 3D segmentation array
        slice_selection (str): Which slices to process ('middle', 'all', or slice number)
        temp_seg_affine (np.ndarray): Base affine transformation matrix
        midslices (np.ndarray): Array of mid-sagittal slices
        ac_coords (np.ndarray): Anterior commissure coordinates
        pc_coords (np.ndarray): Posterior commissure coordinates
        num_thickness_points (int): Number of points for thickness estimation
        subdivisions (list[float]): List of fractions for anatomical subdivisions
        subdivision_method (str): Method for contour subdivision
        contour_smoothing (float): Gaussian sigma for contour smoothing
        debug_image_path (str, optional): Path for debug visualization image
        verbose (bool): Whether to print progress information
        save_template (str | Path | None): Directory path where to save template files, or None to skip saving
        
        Returns:
            tuple: Contains:
            
            - list: List of slice processing results
            - list: List of background IO processes
    """
    slice_results = []
    IO_processes = []
    
    if slice_selection == "middle":
        cc_mesh = CC_Mesh(num_slices=1)
        cc_mesh.set_acpc_coords(ac_coords, pc_coords)
        cc_mesh.set_resolution(vox_size) # contour is always scaled to 1 mm

        # Process only the middle slice
        slice_idx = segmentation.shape[0] // 2
        slice_affine = create_slice_affine(temp_seg_affine, slice_idx, FSAVERAGE_MIDDLE)
        
        (result, contour_with_thickness, 
         anterior_endpoint_idx, posterior_endpoint_idx) = process_slice(segmentation, 
                                                                        slice_idx, 
                                                                        ac_coords, 
                                                                        pc_coords, 
                                                                        slice_affine, 
                                                                        num_thickness_points, 
                                                                        subdivisions, 
                                                                        subdivision_method, 
                                                                        contour_smoothing)
        
        cc_mesh.add_contour(0, 
                            contour_with_thickness[0], 
                            contour_with_thickness[1], 
                            start_end_idx=(anterior_endpoint_idx, posterior_endpoint_idx))

        if result is not None and debug_image_path is not None:
            slice_results.append(result)
            # Create visualization
            if verbose:
                logger.info(f"Saving segmentation qc image to {debug_image_path}")
            IO_processes.append(create_visualization(subdivision_method, result, midslices, 
                                                   debug_image_path, ac_coords, pc_coords, vox_size))
    else:
        num_slices = segmentation.shape[0]
        cc_mesh = CC_Mesh(num_slices=num_slices)
        cc_mesh.set_acpc_coords(ac_coords, pc_coords)
        cc_mesh.set_resolution(vox_size) # contour is always scaled to 1 mm

        # Process multiple slices or specific slice
        if slice_selection == "all":
            start_slice = 0
            end_slice = segmentation.shape[0]
        else:  # specific slice number
            slice_idx = int(slice_selection)
            start_slice = slice_idx
            end_slice = slice_idx + 1
        
        for slice_idx in range(start_slice, end_slice):
            if verbose:
                logger.info(f"Calculating CC measurements for slice {slice_idx+1} of {end_slice-start_slice}")
            
            # Update affine for this slice
            slice_affine = create_slice_affine(temp_seg_affine, slice_idx, FSAVERAGE_MIDDLE)
            
            # Process this slice
            (result, contour_with_thickness, 
             anterior_endpoint_idx, posterior_endpoint_idx) = process_slice(segmentation, slice_idx, 
                                                                            ac_coords, pc_coords, 
                                                                            slice_affine, num_thickness_points, 
                                                                            subdivisions, subdivision_method, 
                                                                            contour_smoothing)

            # insert 
            cc_mesh.add_contour(slice_idx, 
                                contour_with_thickness[0], 
                                contour_with_thickness[1], 
                                start_end_idx=(anterior_endpoint_idx, posterior_endpoint_idx))

            if result is not None:
                slice_results.append(result)

                if (one_debug_image and slice_idx == num_slices // 2) or not one_debug_image:
                    if not one_debug_image:
                        debug_path_base, debug_path_ext = str(debug_image_path).rsplit('.', 1)
                        debug_path_with_postfix = f"{debug_path_base}_slice_{slice_idx}"
                    
                        debug_output_path_slice = Path(f"{debug_path_with_postfix}.{debug_path_ext}")
                        debug_output_path_slice = debug_output_path_slice.with_suffix('.png')
                    else:
                        debug_output_path_slice = debug_image_path
                    
                    if verbose:
                        logger.info(f"Saving segmentation qc image to {debug_output_path_slice}")

                    current_slice_in_volume = midslices.shape[0] // 2 - num_slices // 2 + slice_idx
                    # Create visualization for this slice
                    IO_processes.append(create_visualization(subdivision_method, result, 
                                                            midslices[current_slice_in_volume:current_slice_in_volume+1], 
                                                            debug_output_path_slice, ac_coords, pc_coords, 
                                                            vox_size, f' (Slice {slice_idx})'))

    if save_template is not None:
        # Convert to Path object and ensure directory exists
        template_dir = Path(save_template)
        template_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info("Saving template files (contours.txt, thickness_values.txt, "
                        f"thickness_measurement_points.txt) to {template_dir}")
        cc_mesh.save_contours(str(template_dir / 'contours.txt'))
        cc_mesh.save_thickness_values(str(template_dir / 'thickness_values.txt'))
        cc_mesh.save_thickness_measurement_points(str(template_dir / 'thickness_measurement_points.txt'))


    if len(cc_mesh.contours) > 1 and thickness_image_path is not None:
        cc_mesh.fill_thickness_values()
        cc_mesh.create_mesh()
        cc_mesh.smooth_(1)
        cc_mesh.plot_mesh(output_path=cc_html_path)

        if vtk_file_path is not None:
            if verbose: 
                logger.info(f"Saving vtk file to {vtk_file_path}")
            cc_mesh.write_vtk(str(vtk_file_path))
        #cc_mesh.write_vtk(str(output_dir / 'cc_mesh.vtk'))
        
        
        cc_mesh.to_fs_coordinates()

        if overlay_file_path is not None:
            if verbose: 
                logger.info(f"Saving overlay file to {overlay_file_path}")
            cc_mesh.write_overlay(str(overlay_file_path))

        if surf_file_path is not None:
            if verbose: 
                logger.info(f"Saving surf file to {surf_file_path}")
            cc_mesh.write_fssurf(str(surf_file_path))

        

        if thickness_image_path is not None:
            if verbose: 
                logger.info(f"Saving thickness image to {thickness_image_path}")
            with HiddenPrints():
                cc_mesh.snap_cc_picture(str(thickness_image_path))
        
    
    if not slice_results:
        logger.error("Error: No valid slices were found for postprocessing")
        exit(1)
        
    return slice_results, IO_processes




def vectorized_line_test(coords_x, coords_y, line_start, line_end):
    """Vectorized version of point_relative_to_line for arrays of points.
    
    Args:
        coords_x (np.ndarray): Array of x coordinates
        coords_y (np.ndarray): Array of y coordinates  
        line_start (array-like): [x, y] coordinates of line start point
        line_end (array-like): [x, y] coordinates of line end point
        
    Returns:
        np.ndarray: Boolean array where True means point is to the left of the line
    """
    # Vector from line_start to line_end
    line_vec = np.array(line_end) - np.array(line_start)
    
    # Vectors from line_start to all points (vectorized)
    point_vec_x = coords_x - line_start[0]
    point_vec_y = coords_y - line_start[1]
    
    # Cross product (vectorized): positive means point is to the left of the line
    cross_products = line_vec[0] * point_vec_y - line_vec[1] * point_vec_x
    
    return cross_products > 0




def get_unique_contour_points(split_contours):
    """Get unique contour points from the split contours.
    This is a workaround to retrospectively add voxel-based sub-division
    in the future we could keep track of the sub-division lines for
    every sub-division scheme.

    Args:
        split_contours (list): List of split contours (subsegmentations)

    Returns:
        list: List of unique contour points
    
    """
    # For each contour point, check if it appears in other contours
    unique_contour_points = []
    
    for i, contour in enumerate(split_contours):
        # Get points for this contour
        contour_points = np.vstack((contour[0], -contour[1])).T  # Shape: (N,2)
        
        # Check each point against all other contours
        unique_points = []
        for point in contour_points:
            is_unique = True
            
            # Compare against other contours
            for j, other_contour in enumerate(split_contours):
                if i == j:
                    continue
                    
                other_points = np.vstack((other_contour[0], -other_contour[1])).T
                
                # Check if point exists in other contour (with small tolerance)
                if np.any(np.all(np.abs(other_points - point) < 1e-6, axis=1)):
                    is_unique = False
                    break
                    
            if is_unique:
                unique_points.append(point)
                
        unique_contour_points.append(np.array(unique_points))

    return unique_contour_points


def make_subdivision_mask(slice_shape, split_contours):
    """Create a mask for subdividing the corpus callosum based on split contours.

    This function creates a mask that assigns different labels to different segments of the corpus callosum
    based on the subdivision lines defined by the split contours. Each segment is labeled with a value from
    SUBSEGEMNT_LABELS.

    Args:
        slice_shape (tuple): 
            Shape of the slice (rows, cols)
        split_contours (list): 
            List of contours defining the subdivisions. 
            Each contour is a tuple of x and y coordinates.

    Returns:
        ndarray: 
            A mask of shape slice_shape where each pixel is labeled with a value from SUBSEGEMNT_LABELS
            indicating which subdivision segment it belongs to.
    """

    # unique contour points are the points where sub-division lines were inserted
    unique_contour_points = get_unique_contour_points(split_contours)
    subdivision_segments = unique_contour_points[1:]

    for s in subdivision_segments:
        if len(s) != 2:
            logger.error(f'Subdivision segment {s} has {len(s)} points, expected 2')
 
    # Create coordinate grids for all points in the slice
    rows, cols = slice_shape
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]
    
    # Initialize with first segment label
    subdivision_mask = np.full(slice_shape, SUBSEGEMNT_LABELS[0], dtype=np.int32)
    
    # Process each subdivision line
    for segment_idx, segment_points in enumerate(subdivision_segments):
        line_start = segment_points[0]
        line_end = segment_points[-1]
        
        # Vectorized test: find all points to the right of this line
        points_right_of_line = vectorized_line_test(x_coords, y_coords, line_start, line_end)
        
        # All points to the right of this line belong to the next segment or beyond
        subdivision_mask[points_right_of_line] = SUBSEGEMNT_LABELS[segment_idx + 1]
        
        # Debug visualization (optional)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(figsize=(10, 8))
        # ax.imshow(subdivision_mask, cmap='tab10')
        # ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'r-', linewidth=2)
        # ax.set_title(f'After subdivision line {segment_idx}')
        # plt.show()

    return subdivision_mask


def check_area_changes(contours: list[np.ndarray], threshold: float = 0.3, verbose: bool = False) -> None:
    """Check for large changes between consecutive CC areas and issue warnings.
    
    This function checks if any two consecutive areas have a change greater than
    the specified threshold (default 30%) and issues a warning if they do.
    
    Args:
        contours (list[np.ndarray]): List of contours
        threshold (float, optional): Threshold for relative change. Defaults to 0.3 (30%).
    """

    areas = [np.sum(np.sqrt(np.sum((np.diff(contour, axis=0))**2, axis=1))) for contour in contours]

    assert len(areas) > 1, "At least two areas are required to check for area changes"
    
    for i in range(len(areas) - 1):
        if areas[i] == 0 and areas[i+1] == 0:
            continue  # Skip if both areas are zero
        
        if areas[i] == 0 or areas[i+1] == 0:
            # One area is zero, the other is not - this is a 100% change
            if verbose:
                logger.warning(f"Large area change detected: area {i+1} = {areas[i]:.2f} mm², "
                               f"area {i+2} = {areas[i+1]:.2f} mm² (one area is zero)")
            return False
        
        # Calculate relative change
        relative_change = abs(areas[i+1] - areas[i]) / areas[i]
        
        if relative_change > threshold:
            percent_change = relative_change * 100
            if verbose:
                logger.warning(f"Large corpus callosum area change between slices detected: "
                               f"area {i+1} = {areas[i]:.2f} mm², "
                               f"area {i+2} = {areas[i+1]:.2f} mm² ({percent_change:.1f}% change)")
            return False
    return True