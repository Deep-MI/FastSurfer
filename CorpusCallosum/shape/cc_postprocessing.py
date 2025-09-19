from pathlib import Path

import numpy as np
from shape.cc_endpoint_heuristic import get_endpoints
from shape.cc_mesh import CC_Mesh
from shape.cc_metrics import calculate_cc_index
from shape.cc_subsegment_contour import (
    get_primary_eigenvector,
    hampel_subdivide_contour,
    subdivide_contour,
    subsegment_midline_orthogonal,
    transform_to_acpc_standard,
)
from shape.cc_thickness import cc_thickness, convert_to_ras

from CorpusCallosum.data.constants import FSAVERAGE_MIDDLE
from CorpusCallosum.data.read_write import run_in_background
from CorpusCallosum.visualization.visualization import plot_contours

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
    title = f'CC Subsegmentation: {subdivision_method}{title_suffix}'
    
    if subdivision_method == "shape":
        return run_in_background(plot_contours, False, midslices_data, 
                                result['split_contours'], None, result['midline_equidistant'], result['levelpaths'], 
                                output_image_path, ac_coords, pc_coords, vox_size, title)
    else:
        return run_in_background(plot_contours, False, midslices_data, 
                                None, result['split_contours_hofer_frahm'], result['midline_equidistant'], 
                                result['levelpaths'], output_image_path, ac_coords, pc_coords, vox_size, title)


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
        segmentation (np.ndarray): 3D segmentation array
        slice_idx (int): Index of the slice to process
        ac_coords (np.ndarray): Anterior commissure coordinates
        pc_coords (np.ndarray): Posterior commissure coordinates
        affine (np.ndarray): 4x4 affine transformation matrix
        num_thickness_points (int): Number of points for thickness estimation
        subdivisions (list[float]): List of fractions for anatomical subdivisions
        subdivision_method (str): Method for contour subdivision ('shape', 'vertical', 
            'angular', or 'eigenvector')
        contour_smoothing (float): Gaussian sigma for contour smoothing
        verbose (bool): Whether to print progress information
        
    Returns:
        dict or None: Dictionary containing measurements if successful, including:
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
            print('Error: Angular subdivision method (Hampel) only supports equidistant subdivision, '
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
                  output_dir, debug_image_path=None, thickness_image_path=None, vox_size=None, verbose=False, 
                  save_template=None):
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
        output_dir (str): Base output directory
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
        cc_mesh.set_resolution(1) # contour is always scaled to 1 mm

        # Process only the middle slice
        slice_idx = segmentation.shape[0] // 2
        slice_affine = create_slice_affine(temp_seg_affine, slice_idx, FSAVERAGE_MIDDLE)
        
        result, contour_with_thickness, anterior_endpoint_idx, posterior_endpoint_idx = process_slice(segmentation, 
                                                                                                      slice_idx, 
                                                                                                      ac_coords, 
                                                                                                      pc_coords, 
                                                                                                      slice_affine, 
                             num_thickness_points, subdivisions, subdivision_method, contour_smoothing)
        
        cc_mesh.add_contour(0, 
                            contour_with_thickness[0], 
                            contour_with_thickness[1], 
                            start_end_idx=(anterior_endpoint_idx, posterior_endpoint_idx))

        if result is not None:
            slice_results.append(result)
            # Create visualization
            IO_processes.append(create_visualization(subdivision_method, result, midslices, 
                                                   debug_image_path, ac_coords, pc_coords, vox_size))
    else:

        cc_mesh = CC_Mesh(num_slices=segmentation.shape[0])
        cc_mesh.set_acpc_coords(ac_coords, pc_coords)
        cc_mesh.set_resolution(1) # contour is always scaled to 1 mm

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
                print(f"Calculating CC measurements for slice {slice_idx+1} of {end_slice-start_slice}")
            
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
                
                # For single slice mode, save to main directory
                if slice_selection != "all":
                    output_subdir = output_dir
                else:
                    # For all slices mode, create per-slice directory
                    output_subdir = output_dir / f'slice_{slice_idx}'
                    output_subdir.mkdir(exist_ok=True)
                
                # Create visualization for this slice
                IO_processes.append(create_visualization(subdivision_method, result, midslices[slice_idx:slice_idx+1], 
                                                         output_subdir, ac_coords, pc_coords, 
                                                         vox_size, f' (Slice {slice_idx})'))

    if save_template is not None:
        # Convert to Path object and ensure directory exists
        template_dir = Path(save_template)
        template_dir.mkdir(parents=True, exist_ok=True)
        cc_mesh.save_contours(str(template_dir / 'contours.txt'))
        cc_mesh.save_thickness_values(str(template_dir / 'thickness_values.txt'))
        cc_mesh.save_thickness_measurement_points(str(template_dir / 'thickness_measurement_points.txt'))


    if len(cc_mesh.contours) > 1:
        cc_mesh.fill_thickness_values()
        cc_mesh.create_mesh()
        cc_mesh.smooth_(1)
        cc_mesh.plot_mesh()
        #cc_mesh.write_vtk(str(output_dir / 'cc_mesh.vtk'))
        cc_mesh.snap_cc_picture(str(output_dir / thickness_image_path))
        
    
    if not slice_results:
        print("Error: No valid slices were found for postprocessing")
        exit(1)
        
    return slice_results, IO_processes