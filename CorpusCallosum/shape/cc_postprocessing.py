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
import concurrent.futures
from pathlib import Path
from typing import Literal, get_args

import numpy as np
from numpy import typing as npt

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.data.constants import CC_LABEL, FSAVERAGE_MIDDLE, SUBSEGMENT_LABELS
from CorpusCallosum.shape.cc_endpoint_heuristic import get_endpoints
from CorpusCallosum.shape.cc_mesh import CCMesh
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
from FastSurferCNN.utils.common import SubjectDirectory, update_docstring
from FastSurferCNN.utils.common import thread_executor as executor

SubdivisionMethod = Literal["shape", "vertical", "angular", "eigenvector"]

logger = logging.get_logger(__name__)

# assert LIA orientation
LIA_ORIENTATION = np.zeros((3,3))
LIA_ORIENTATION[0,0] = -1
LIA_ORIENTATION[1,2] = 1
LIA_ORIENTATION[2,1] = -1


@update_docstring(SubdivisionMethod=str(get_args(SubdivisionMethod))[1:-1])
def async_create_visualization(
        subdivision_method: SubdivisionMethod,
        result: dict,
        midslices_data: np.ndarray,
        output_image_path: str | Path,
        ac_coords: np.ndarray,
        pc_coords: np.ndarray,
        vox_size: float,
        title_suffix: str = "",
) -> concurrent.futures.Future:
    """Create visualization plots based on subdivision method.

    Parameters
    ----------
    subdivision_method : {SubdivisionMethod}
        The subdivision method being used.
    result : dict
        Dictionary containing processing results with split_contours.
    midslices_data : np.ndarray
        Slice data for visualization.
    output_image_path : Path, str
        Path to save visualization.
    ac_coords : np.ndarray
        AC coordinates.
    pc_coords : np.ndarray
        PC coordinates.
    vox_size : float
        Voxel size in mm.
    title_suffix : str, optional
        Additional text to append to the title, by default "".

    Returns
    -------
    multiprocessing.Process
        Process object for background execution.
    """
    title = f"CC Subsegmentation by {subdivision_method} {title_suffix}"

    args_dict = {
        "debug": True,
        "transformed": midslices_data,
        "split_contours": result["split_contours"],
        "midline_equidistant": result["midline_equidistant"],
        "levelpaths": result["levelpaths"],
        "output_path": output_image_path,
        "ac_coords": ac_coords,
        "pc_coords": pc_coords,
        "vox_size": vox_size,
        "title": title,
    }

    return executor().submit(plot_contours, **args_dict)


def create_slice_affine(temp_seg_affine: np.ndarray, slice_idx: int, fsaverage_middle: int) -> np.ndarray:
    """Create slice-specific affine transformation matrix.

    Parameters
    ----------
    temp_seg_affine : np.ndarray
        Base 4x4 affine transformation matrix.
    slice_idx : int
        Index of the slice to transform.
    fsaverage_middle : int
        Reference middle slice index in fsaverage space.

    Returns
    -------
    np.ndarray
        Modified 4x4 affine transformation matrix for the specific slice.
    """
    slice_affine = temp_seg_affine.copy()
    slice_affine[0, 3] = -fsaverage_middle + slice_idx
    return slice_affine


@update_docstring(SubdivisionMethod=str(get_args(SubdivisionMethod))[1:-1])
def recon_cc_surf_measures_multi(
    segmentation: np.ndarray,
    slice_selection: str,
    temp_seg_affine: np.ndarray,
    midslices: np.ndarray,
    ac_coords: np.ndarray,
    pc_coords: np.ndarray,
    num_thickness_points: int,
    subdivisions: list[float],
    subdivision_method: SubdivisionMethod,
    contour_smoothing: float,
    subject_dir: SubjectDirectory,
    qc_image_path: str | None = None,
    vox_size: tuple[float, float, float] | None = None,
    vox2ras_tkr: np.ndarray | None = None,
) -> tuple[list, list[concurrent.futures.Future]]:
    """Surface reconstruction and metrics computation of corpus callosum slices based on selection mode.

    Parameters
    ----------
    segmentation : np.ndarray
        3D segmentation array.
    slice_selection : str
        Which slices to process ('middle', 'all', or slice number).
    temp_seg_affine : np.ndarray
        Base affine transformation matrix.
    midslices : np.ndarray
        Array of mid-sagittal slices.
    ac_coords : np.ndarray
        Anterior commissure coordinates.
    pc_coords : np.ndarray
        Posterior commissure coordinates.
    num_thickness_points : int
        Number of points for thickness estimation.
    subdivisions : list[float]
        List of fractions for anatomical subdivisions.
    subdivision_method : {SubdivisionMethod}
        Method for contour subdivision.
    contour_smoothing : float
        Gaussian sigma for contour smoothing.
    subject_dir : SubjectDirectory
        The SubjectDirectory object managing file names in the subject directory.
    qc_image_path : Path, str, optional
        Path for QC visualization image.
    vox_size : 3-tuple of floats, optional
        Voxel size in millimeters (x, y, z).
    vox2ras_tkr : np.ndarray, optional
        Voxel to RAS tkr-space transformation matrix.

    Returns
    -------
    list
        List of slice processing results.
    list[concurrent.futures.Future]
        List of background IO processes.
    """
    slice_results = []
    io_futures = []

    if slice_selection == "middle":
        cc_mesh = CCMesh(num_slices=1)
        cc_mesh.set_acpc_coords(ac_coords, pc_coords)
        cc_mesh.set_resolution(vox_size[0])

        # Process only the middle slice
        slice_idx = segmentation.shape[0] // 2
        slice_affine = create_slice_affine(temp_seg_affine, slice_idx, FSAVERAGE_MIDDLE)

        result, contour_with_thickness, *endpoint_idxs = recon_cc_surf_measure(
            segmentation,
            slice_idx,
            ac_coords,
            pc_coords,
            slice_affine,
            num_thickness_points,
            subdivisions,
            subdivision_method,
            contour_smoothing,
            vox_size[0],
        )

        cc_mesh.add_contour(0, *contour_with_thickness, start_end_idx=endpoint_idxs)

        if result is not None and qc_image_path is not None:
            slice_results.append(result)
            # Create visualization
            logger.info(f"Saving segmentation qc image to {qc_image_path}")
            io_futures.append(async_create_visualization(
                subdivision_method,
                result,
                midslices,
                qc_image_path,
                ac_coords,
                pc_coords,
                vox_size[0],
            ))
    else:
        num_slices = segmentation.shape[0]
        cc_mesh = CCMesh(num_slices=num_slices)
        cc_mesh.set_acpc_coords(ac_coords, pc_coords)
        cc_mesh.set_resolution(vox_size[0])

        # Process multiple slices or specific slice
        if slice_selection == "all":
            start_slice = 0
            end_slice = segmentation.shape[0]
        else:  # specific slice number
            slice_idx = int(slice_selection)
            start_slice = slice_idx
            end_slice = slice_idx + 1

        for slice_idx in range(start_slice, end_slice):
            logger.info(f"Calculating CC measurements for slice {slice_idx+1} of {end_slice-start_slice}")

            # Update affine for this slice
            slice_affine = create_slice_affine(temp_seg_affine, slice_idx, FSAVERAGE_MIDDLE)

            # Process this slice
            result, contour_with_thickness, *endpoint_idxs = recon_cc_surf_measure(
                segmentation,
                slice_idx,
                ac_coords,
                pc_coords,
                slice_affine,
                num_thickness_points,
                subdivisions,
                subdivision_method,
                contour_smoothing,
                vox_size[0],
            )

            # insert
            cc_mesh.add_contour(slice_idx, *contour_with_thickness, start_end_idx=endpoint_idxs)

            if result is not None:
                slice_results.append(result)

                if logger.getEffectiveLevel() <= logging.INFO and subject_dir.has_attribute("cc_qc_image"):
                    qc_img = subject_dir.filename_by_attribute("cc_qc_image")
                    if logger.getEffectiveLevel() <= logging.DEBUG:
                        qc_img = (qc_img.parent / f"{qc_img.stem}_slice_{slice_idx}{qc_img.suffix}").with_suffix(".png")

                    if logger.getEffectiveLevel() <= logging.DEBUG or slice_idx == num_slices // 2:
                        logger.info(f"Saving segmentation qc image to {qc_img}")

                        current_slice_in_volume = midslices.shape[0] // 2 - num_slices // 2 + slice_idx
                        # Create visualization for this slice
                        io_futures.append(async_create_visualization(
                            subdivision_method,
                            result,
                            midslices[current_slice_in_volume:current_slice_in_volume+1],
                            qc_img,
                            ac_coords,
                            pc_coords,
                            vox_size[0],
                            f" (Slice {slice_idx})",
                        ))

    if subject_dir.has_attribute("save_template_dir"):
        template_dir = subject_dir.filename_by_attribute("save_template_dir")
        # ensure directory exists
        template_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving template files (contours.txt, thickness_values.txt, "
                    f"thickness_measurement_points.txt) to {template_dir}")
        cc_mesh.save_contours(template_dir / "contours.txt")
        cc_mesh.save_thickness_values(template_dir / "thickness_values.txt")
        cc_mesh.save_thickness_measurement_points(template_dir / "thickness_measurement_points.txt")


    if len(cc_mesh.contours) > 1 and subject_dir.has_attribute("cc_html"):
        cc_mesh.fill_thickness_values()
        cc_mesh.create_mesh()
        cc_mesh.smooth_(1)
        logger.info(f"Saving CC 3D visualization to {subject_dir.filename_by_attribute('cc_html')}")
        cc_mesh.plot_mesh(output_path=subject_dir.filename_by_attribute("cc_html"), show_mesh_edges=True)

        if subject_dir.has_attribute("cc_mesh"):
            vtk_file_path = subject_dir.filename_by_attribute("cc_mesh")
            logger.info(f"Saving vtk file to {vtk_file_path}")
            cc_mesh.write_vtk(vtk_file_path)

        cc_mesh.to_fs_coordinates(vox2ras_tkr=vox2ras_tkr, vox_size=vox_size)
        if subject_dir.has_attribute("overlay_file"):
            overlay_file_path = subject_dir.filename_by_attribute("overlay_file")
            logger.info(f"Saving overlay file to {overlay_file_path}")
            cc_mesh.write_overlay(overlay_file_path)

        if subject_dir.has_attribute("cc_surf_file"):
            surf_file_path = subject_dir.filename_by_attribute("cc_surf_file")
            logger.info(f"Saving surf file to {surf_file_path}")
            cc_mesh.write_fssurf(surf_file_path)

        if subject_dir.has_attribute("thickness_image"):
            thickness_image_path = subject_dir.filename_by_attribute("thickness_image")
            logger.info(f"Saving thickness image to {thickness_image_path}")
            with HiddenPrints():
                cc_mesh.snap_cc_picture(thickness_image_path)


    if not slice_results:
        logger.error("Error: No valid slices were found for postprocessing")
        raise ValueError("No valid slices were found for postprocessing")

    return slice_results, io_futures


def recon_cc_surf_measure(
    segmentation: np.ndarray,
    slice_idx: int,
    ac_coords: np.ndarray,
    pc_coords: np.ndarray,
    affine: np.ndarray,
    num_thickness_points: int,
    subdivisions: list[float],
    subdivision_method: SubdivisionMethod,
    contour_smoothing: float,
    vox_size: float
) -> tuple[dict[str, float | int | np.ndarray | list[float]], np.ndarray, int, int]:
    """Reconstruct surfaces and compute measures for a single slice for the corpus callosum.

    Parameters
    ----------
    segmentation : np.ndarray
        3D segmentation array.
    slice_idx : int
        Index of the slice to process.
    ac_coords : np.ndarray
        Anterior commissure coordinates.
    pc_coords : np.ndarray
        Posterior commissure coordinates.
    affine : np.ndarray
        4x4 affine transformation matrix.
    num_thickness_points : int
        Number of points for thickness estimation.
    subdivisions : list[float]
        List of fractions for anatomical subdivisions.
    subdivision_method : SubdivisionMethod
        Method for contour subdivision ('shape', 'vertical', 'angular', or 'eigenvector').
    contour_smoothing : float
        Gaussian sigma for contour smoothing.
    vox_size : float
        Voxel size in millimeters.

    Returns
    -------
    dict of measures
        Dictionary containing measurements if successful, including:
        - cc_index : float - Corpus callosum shape index.
        - circularity : float - Shape circularity measure.
        - areas : np.ndarray - Areas of subdivided regions.
        - midline_length : float - Length along the midline.
        - thickness : np.ndarray - Array of thickness measurements.
        - curvature : np.ndarray - Array of curvature measurements.
        - thickness_profile : list[float] - Thickness measurements along the contour.
        - total_area : float - Total area of the CC.
        - total_perimeter : float - Total perimeter length.
        - split_contours : list[np.ndarray] - Subdivided contour segments.
        - midline_equidistant : np.ndarray - Equidistant points along midline.
        - levelpaths : list[np.ndarray] - Paths for thickness measurements.
        - thickness_measurement_points : np.ndarray - Points where thickness was measured.
        - slice_index : int - Index of the processed slice.
    contour_with_thickness : np.ndarray
    anterior_endpoint_index : int
    posterior_endpoint_index : int

    Raises
    ------
    ValueError
        If no CC is found in the specified slice.

    Notes
    -----
    The function performs the following steps:
    1. Extracts CC contour and identifies endpoints.
    2. Converts coordinates to RAS space.
    3. Calculates thickness profile using Laplace equation.
    4. Computes shape metrics and subdivisions.
    5. Generates visualization data.

    """
    cc_mask_slice: npt.NDArray[bool] = segmentation[slice_idx] == CC_LABEL
    if not np.any(cc_mask_slice):
        raise ValueError(f"No CC found in slice {slice_idx}")

    contour, *endpoint_idxs = get_endpoints(
        cc_mask_slice,
        ac_coords,
        pc_coords,
        vox_size,
        return_coordinates=False,
        contour_smoothing=contour_smoothing,
    )
    contour_1mm = convert_to_ras(contour, affine)

    midline_len, thickness, curvature, midline_equi, levelpaths, contour_with_thickness, *endpoint_idxs = cc_thickness(
        contour_1mm.T,
        *endpoint_idxs,
        n_points=num_thickness_points,
    )

    thickness_profile = [
        np.sum(np.sqrt(np.diff(np.array(levelpath[:,:2]), axis=0)**2), axis=0)
        for levelpath in levelpaths
    ]
    thickness_profile = np.linalg.norm(np.array(thickness_profile),axis=1)

    acpc_contour_coords = contour_1mm[:, list(endpoint_idxs)].T
    contour_acpc, ac_pt_acpc, pc_pt_acpc, rotate_back_acpc = transform_to_acpc_standard(
        contour_1mm,
        *acpc_contour_coords,
    )
    cc_index = calculate_cc_index(contour_acpc)

    # Apply different subdivision methods based on user choice
    if subdivision_method == "shape":
        areas, split_contours = subsegment_midline_orthogonal(midline_equi, subdivisions, contour_1mm, plot=False)
        split_contours = [transform_to_acpc_standard(split_contour, *acpc_contour_coords)[0]
                          for split_contour in split_contours]
    elif subdivision_method == "vertical":
        areas, split_contours = subdivide_contour(contour_acpc, subdivisions, plot=False)
    elif subdivision_method == "angular":
        if not np.allclose(np.diff(subdivisions), np.diff(subdivisions)[0]):
            logger.error("Error: Angular subdivision method (Hampel) only supports equidistant subdivision, "
                         f"but got: {subdivisions}. No measures are computed.")
            return {}, contour_with_thickness, *endpoint_idxs
        areas, split_contours = hampel_subdivide_contour(contour_acpc, num_rays=len(subdivisions), plot=False)
    elif subdivision_method == "eigenvector":
        pt0, pt1 = get_primary_eigenvector(contour_acpc)
        contour_eigen, _, _, rotate_back_eigen = transform_to_acpc_standard(contour_acpc, pt0, pt1)
        ac_pt_eigen, _, _, _ = transform_to_acpc_standard(ac_pt_acpc[:, None], pt0, pt1)
        ac_pt_eigen = ac_pt_eigen[:, 0]
        areas, split_contours = subdivide_contour(contour_eigen, subdivisions, oriented=True, hline_anchor=ac_pt_eigen)
        split_contours = [rotate_back_eigen(split_contour) for split_contour in split_contours]

    total_area = np.sum(areas)
    total_perimeter = np.sum(np.sqrt(np.sum((np.diff(contour_1mm, axis=0))**2, axis=1)))
    circularity = 4 * np.pi * total_area / (total_perimeter**2)

    # Transform split contours back to original space
    split_contours = [rotate_back_acpc(split_contour) for split_contour in split_contours]

    measures = {
        "cc_index": cc_index,
        "circularity": circularity,
        "areas": areas,
        "midline_length": midline_len,
        "thickness": thickness,
        "curvature": curvature,
        "thickness_profile": thickness_profile,
        "total_area": total_area,
        "total_perimeter": total_perimeter,
        "split_contours": split_contours,
        "midline_equidistant": midline_equi,
        "levelpaths": levelpaths,
        "slice_index": slice_idx
    }
    return measures, contour_with_thickness, *endpoint_idxs


def vectorized_line_test(coords_x: np.ndarray, coords_y: np.ndarray, 
                        line_start: np.ndarray, line_end: np.ndarray) -> np.ndarray:
    """Vectorized version of point_relative_to_line for arrays of points.

    Parameters
    ----------
    coords_x : np.ndarray
        Array of x coordinates.
    coords_y : np.ndarray
        Array of y coordinates.
    line_start : array-like
        [x, y] coordinates of line start point.
    line_end : array-like
        [x, y] coordinates of line end point.

    Returns
    -------
    np.ndarray
        Boolean array where True means point is to the left of the line.
    """
    # Vector from line_start to line_end
    line_vec = np.array(line_end) - np.array(line_start)
    
    # Vectors from line_start to all points (vectorized)
    point_vec_x = coords_x - line_start[0]
    point_vec_y = coords_y - line_start[1]
    
    # Cross product (vectorized): positive means point is to the left of the line
    cross_products = line_vec[0] * point_vec_y - line_vec[1] * point_vec_x
    
    return cross_products > 0




def get_unique_contour_points(split_contours: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """Get unique contour points from the split contours.

    Parameters
    ----------
    split_contours : list[tuple[np.ndarray, np.ndarray]]
        List of split contours (subsegmentations), each containing x and y coordinates.

    Returns
    -------
    list[np.ndarray]
        List of unique contour points for each subsegment.

    Notes
    -----
    This is a workaround to retrospectively add voxel-based subdivision.
    In the future, we could keep track of the subdivision lines for
    every subdivision scheme.

    The function:
    1. Processes each contour point.
    2. Checks if it appears in other contours (with small tolerance).
    3. Collects points unique to each subsegment.
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


def make_subdivision_mask(
    slice_shape: tuple[int, int],
    split_contours: list[tuple[np.ndarray, np.ndarray]],
    vox_size: tuple[float, float, float]
) -> np.ndarray:
    """Create a mask for subdividing the corpus callosum based on split contours.

    Parameters
    ----------
    slice_shape : tuple[int, int]
        Shape of the slice (rows, cols).
    split_contours : list[tuple[np.ndarray, np.ndarray]]
        List of contours defining the subdivisions.
        Each contour is a tuple of x and y coordinates.

    Returns
    -------
    np.ndarray
        A mask of shape slice_shape where each pixel is labeled with a value
        from SUBSEGEMNT_LABELS indicating which subdivision segment it belongs to.

    Notes
    -----
    The function:
    1. Extracts unique contour points at subdivision boundaries.
    2. Creates coordinate grids for all points in the slice.
    3. Initializes mask with first segment label.
    4. For each subdivision line:
    - Tests which points lie to the right of the line.
    - Updates labels for those points.

    """

    # unique contour points are the points where sub-division lines were inserted
    unique_contour_points = get_unique_contour_points(split_contours)
    subdivision_segments = unique_contour_points[1:]

    for s in subdivision_segments:
        if len(s) != 2:
            logger.error(f"Subdivision segment {s} has {len(s)} points, expected 2")
 
    # Create coordinate grids for all points in the slice
    rows, cols = slice_shape
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]

    subsegment_labels_anterior_posterior = SUBSEGMENT_LABELS.copy()
    subsegment_labels_anterior_posterior.reverse()
    
    # Initialize with first segment label
    subdivision_mask = np.full(slice_shape, subsegment_labels_anterior_posterior[0], dtype=np.int32)
    
    # Process each subdivision line
    for segment_idx, segment_points in enumerate(subdivision_segments):
        line_start = segment_points[0] / vox_size[0]
        line_end = segment_points[-1] / vox_size[0]
        
        # Vectorized test: find all points to the right of this line
        points_right_of_line = vectorized_line_test(x_coords, y_coords, line_start, line_end)
        
        # All points to the right of this line belong to the next segment or beyond
        subdivision_mask[points_right_of_line] = subsegment_labels_anterior_posterior[segment_idx + 1]
        
        # Debug visualization (optional)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(figsize=(10, 8))
        # ax.imshow(subdivision_mask, cmap="tab10")
        # ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], "r-", linewidth=2)
        # ax.set_title(f"After subdivision line {segment_idx}")
        # plt.show()

    return subdivision_mask


def check_area_changes(contours: list[np.ndarray], threshold: float = 0.3) -> bool:
    """Check for large changes between consecutive CC areas and issue warnings.
    
    Parameters
    ----------
    contours : list[np.ndarray]
        List of contours.
    threshold : float, default=0.3
        Threshold for relative change.

    Returns
    -------
    bool
        True if no large area changes are detected, False otherwise.
    """

    areas = np.asarray([np.sum(np.sqrt(np.sum((np.diff(contour, axis=0))**2, axis=1))) for contour in contours])

    assert len(areas) > 1, "At least two areas are required to check for area changes"

    if np.any(areas == 0):
        # One area is zero, the other is not - this is a 100% change
        logger.warning(f"Areas {np.where(areas == 0)[0].tolist()} are zero mm²")
        return False

    # Calculate relative change
    relative_change = np.abs(np.diff(areas)) / areas[:-1]

    if np.any(where_change := relative_change > threshold):
        indices = np.where(where_change)[0]
        percent_change = relative_change[where_change] * 100
        logger.info(
            f"Large corpus callosum area change after slices {indices.tolist()} detected: " +
            ", ".join(f"areas {(i,i+1)} = ({areas[i]:.2f},{areas[i+1]:.2f}) mm² ({p:.1f}% change)"
                      for i, p in zip(indices, percent_change, strict=True))
        )
        return False
    return True