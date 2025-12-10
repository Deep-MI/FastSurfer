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
from copy import copy
from functools import partial
from pathlib import Path
from typing import Literal, TypedDict, get_args

import numpy as np

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.data.constants import CC_LABEL, FSAVERAGE_MIDDLE, SUBSEGMENT_LABELS
from CorpusCallosum.shape.contour import CCContour
from CorpusCallosum.shape.endpoint_heuristic import get_endpoints
from CorpusCallosum.shape.mesh import create_CC_mesh_from_contours
from CorpusCallosum.shape.metrics import calculate_cc_index
from CorpusCallosum.shape.subsegment_contour import (
    ContourList,
    get_primary_eigenvector,
    hampel_subdivide_contour,
    subdivide_contour,
    subsegment_midline_orthogonal,
    transform_to_acpc_standard,
)
from CorpusCallosum.shape.thickness import cc_thickness, convert_to_ras
from CorpusCallosum.utils.types import ContourThickness, Points2dType
from CorpusCallosum.utils.visualization import plot_contours
from FastSurferCNN.utils import AffineMatrix4x4, Image3d, ScalarType, Shape2d, Shape3d, Vector2d
from FastSurferCNN.utils.common import SubjectDirectory, suppress_stdout, update_docstring
from FastSurferCNN.utils.parallel import process_executor, thread_executor

SubdivisionMethod = Literal["shape", "vertical", "angular", "eigenvector"]
SliceSelection = Literal["middle", "all"] | int

logger = logging.get_logger(__name__)

# assert LIA orientation
LIA_ORIENTATION = np.zeros((3,3))
LIA_ORIENTATION[0,0] = -1
LIA_ORIENTATION[1,2] = 1
LIA_ORIENTATION[2,1] = -1


class CCMeasuresDict(TypedDict):
    """TypedDict for corpus callosum measures.

    Attributes
    ----------
    cc_index : float
        Corpus callosum shape index.
    circularity : float
        Shape circularity measure.
    areas : np.ndarray
        Areas of subdivided regions.
    midline_length : float
        Length along the midline.
    thickness : float
        Array of thickness measurements.
    curvature : float
        Array of curvature measurements.
    thickness_profile : np.ndarray of type float
        Thickness measurements along the contour.
    total_area : float
        Total area of the CC.
    total_perimeter : float
        Total perimeter length.
    split_contours : list of np.ndarray
        Subdivided contour segments in AS-slice coordinates.
    midline_equidistant : np.ndarray
        Equidistant points along midline in AS-slice coordinates.
    levelpaths : list of np.ndarray
        Paths for thickness measurements in AS-slice coordinates.
    slice_index : int
        Index of the processed slice.
    """
    cc_index: float
    circularity: float
    areas: np.ndarray
    midline_length: float
    thickness: float
    curvature: float
    thickness_profile: np.ndarray[tuple[int], np.dtype[float]]
    total_area: float
    total_perimeter: float
    total_area: float
    total_perimeter: float
    split_contours: ContourList
    midline_equidistant: np.ndarray
    levelpaths: list[np.ndarray]
    slice_index: int


def create_sag_slice_vox2vox(slice_idx: int, fsaverage_middle: float) -> AffineMatrix4x4:
    """Create slice-specific slice to full affine transformation matrix.

    Returns a volume to slice in volume affine.

    Parameters
    ----------
    slice_idx : int
        Index of the slice to transform.
    fsaverage_middle : float
        Reference middle slice index in fsaverage space.

    Returns
    -------
    np.ndarray
        Modified 4x4 affine transformation matrix for the specific slice.
    """
    slice2full_vox2vox: AffineMatrix4x4 = np.eye(4, dtype=float)
    slice2full_vox2vox[0, 3] = -fsaverage_middle + slice_idx
    return slice2full_vox2vox


@update_docstring(SubdivisionMethod=str(get_args(SubdivisionMethod))[1:-1])
def recon_cc_surf_measures_multi(
    segmentation: np.ndarray[Shape3d, np.dtype[int]],
    slice_selection: SliceSelection,
    fsavg_vox2ras: AffineMatrix4x4,
    midslices: Image3d,
    ac_coords: Vector2d,
    pc_coords: Vector2d,
    num_thickness_points: int,
    subdivisions: list[float],
    subdivision_method: SubdivisionMethod,
    contour_smoothing: int,
    subject_dir: SubjectDirectory,
    vox_size: tuple[float, float, float],
    vox2ras_tkr: AffineMatrix4x4 | None = None,
) -> tuple[list[CCMeasuresDict], list[concurrent.futures.Future]]:
    """Surface reconstruction and metrics computation of corpus callosum slices based on selection mode.

    Parameters
    ----------
    segmentation : np.ndarray
        3D segmentation array.
    slice_selection : str
        Which slices to process ('middle', 'all', or slice number).
    fsavg_vox2ras : np.ndarray
        Base affine transformation matrix (fsaverage, upright space).
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
    contour_smoothing : int
        Gaussian sigma for contour smoothing.
    subject_dir : SubjectDirectory
        The SubjectDirectory object managing file names in the subject directory.
    vox_size : 3-tuple of floats
        LIA-oriented voxel size in millimeters (x, y, z).
    vox2ras_tkr : np.ndarray, optional
        Voxel to RAS tkr-space transformation matrix.

    Returns
    -------
    list of CCMeasuresDict
        List of slice processing results.
    list of concurrent.futures.Future
        List of background IO processes.
    """
    slice_cc_measures: list[CCMeasuresDict] = []
    io_futures = []

    if subdivision_method == "angular" and not np.allclose(np.diff(subdivisions), np.diff(subdivisions)[0]):
        raise ValueError(
            f"Angular subdivision method (Hampel) only supports equidistant subdivision, "
            f"but got: {subdivisions}. No measures are computed.",
        )

    _each_slice = partial(
        recon_cc_surf_measure,
        segmentation,
        ac_coords=ac_coords,
        pc_coords=pc_coords,
        num_thickness_points=num_thickness_points,
        subdivisions=subdivisions,
        subdivision_method=subdivision_method,
        contour_smoothing=contour_smoothing,
        vox_size=vox_size,
    )

    # Process multiple slices or specific slice
    if slice_selection == "middle":
        num_slices = 1
        # Process only the middle slice
        slices_to_recon = [segmentation.shape[0] // 2]
    elif slice_selection == "all":
        num_slices = segmentation.shape[0]
        start_slice = 0
        end_slice = segmentation.shape[0]
        slices_to_recon = range(start_slice, end_slice)
    else:  # specific slice number
        num_slices = 1
        slices_to_recon = [int(slice_selection)]

    _gen_fsavg2slice_vox2vox = partial(create_sag_slice_vox2vox, fsaverage_middle=FSAVERAGE_MIDDLE)
    per_slice_vox2ras = fsavg_vox2ras @ np.stack(list(map(_gen_fsavg2slice_vox2vox, slices_to_recon)), axis=0)

    per_slice_recon = process_executor().map(_each_slice, slices_to_recon, per_slice_vox2ras, chunksize=1)
    cc_contours = []

    for i, (slice_idx, _results) in enumerate(zip(slices_to_recon, per_slice_recon, strict=True)):
        progress = f" ({i+1} of {num_slices})" if num_slices > 1 else ""
        logger.info(f"Calculating CC measurements for slice {slice_idx+1}{progress}")
        # unpack values from _results
        cc_measures: CCMeasuresDict = _results[0]
        contour_in_as_space_and_thickness: ContourThickness = _results[1]
        endpoint_idxs: tuple[int, int] = _results[2]
        contour_in_as_space: Points2dType = contour_in_as_space_and_thickness[:, :2]
        thickness_values: np.ndarray[tuple[int], np.dtype[float]] = contour_in_as_space_and_thickness[:, 2]

        cc_contours.append(CCContour(contour_in_as_space, thickness_values, endpoint_idxs, resolution=vox_size[0]))
        if cc_measures is None:
            # this should not happen, but just in case
            logger.warning(f"Slice index {slice_idx+1}{progress} returned result `None`")

        slice_cc_measures.append(cc_measures)
        is_debug = logger.getEffectiveLevel() <= logging.DEBUG
        is_midslice = slice_idx == num_slices // 2
        if subject_dir.has_attribute("cc_qc_image") and (is_debug or is_midslice):
            qc_imgs: list[Path] = (subject_dir.filename_by_attribute("cc_qc_image"),)
            if is_debug:
                qc_slice_img = qc_imgs[0].with_suffix(f".slice_{slice_idx}.png")
                qc_imgs = (qc_imgs if is_midslice else []) + [qc_slice_img]

            logger.info(f"Saving segmentation qc image to {', '.join(map(str, qc_imgs))}")
            current_slice_in_volume = midslices.shape[0] // 2 - num_slices // 2 + slice_idx
            # Create visualization for this slice
            io_futures.append(
                thread_executor().submit(
                    plot_contours,
                    transformed=midslices[current_slice_in_volume:current_slice_in_volume+1],
                    split_contours=cc_measures["split_contours"],
                    midline_equidistant=cc_measures["midline_equidistant"],
                    levelpaths=cc_measures["levelpaths"],
                    output_path=qc_imgs,
                    ac_coords=ac_coords,
                    pc_coords=pc_coords,
                    vox_size=vox_size,
                    title=f"CC Subsegmentation by {subdivision_method} (Slice {slice_idx + 1})",
                )
            )


    if subject_dir.has_attribute("save_template_dir"):
        template_dir = subject_dir.filename_by_attribute("save_template_dir")
        # ensure directory exists
        template_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving template files (contours.txt, thickness_values.txt, "
                    f"thickness_measurement_points.txt) to {template_dir}")
        run = thread_executor().submit
        for j in range(len(cc_contours)):
            # FIXME: check, if this is fixed (thickness values not nan == 200)
            #  this does not seem to be thread-safe, do not parallelize!
            io_futures.append(run(cc_contours[j].save_contour, template_dir / f"contour_{j}.txt"))
            io_futures.append(run(cc_contours[j].save_thickness_values, template_dir / f"thickness_values_{j}.txt"))

    mesh_outputs = ("html", "mesh", "thickness_overlay", "surf", "thickness_image")
    if len(cc_contours) > 1 and any(subject_dir.has_attribute(f"cc_{n}") for n in mesh_outputs):
        _cc_contours = thread_executor().map(_resample_thickness, cc_contours)
        cc_mesh = create_CC_mesh_from_contours(list(cc_contours), smooth=1)
        if subject_dir.has_attribute("cc_html"):
            logger.info(f"Saving CC 3D visualization to {subject_dir.filename_by_attribute('cc_html')}")
            io_futures.append(thread_executor().submit(
                cc_mesh.plot_mesh,output_path=subject_dir.filename_by_attribute("cc_html")))

        if subject_dir.has_attribute("cc_mesh"):
            vtk_file_path = subject_dir.filename_by_attribute("cc_mesh")
            logger.info(f"Saving vtk file to {vtk_file_path}")
            io_futures.append(thread_executor().submit(cc_mesh.write_vtk, vtk_file_path))

        cc_mesh.to_fs_coordinates(vox2ras_tkr=vox2ras_tkr)
        if subject_dir.has_attribute("cc_thickness_overlay"):
            overlay_file_path = subject_dir.filename_by_attribute("cc_thickness_overlay")
            logger.info(f"Saving overlay file to {overlay_file_path}")
            io_futures.append(thread_executor().submit(cc_mesh.write_morph_data, overlay_file_path))

        if subject_dir.has_attribute("cc_surf"):
            surf_file_path = subject_dir.filename_by_attribute("cc_surf")
            logger.info(f"Saving surf file to {surf_file_path}")
            io_futures.append(thread_executor().submit(cc_mesh.write_fssurf, surf_file_path))

        if subject_dir.has_attribute("cc_thickness_image"):
            thickness_image_path = subject_dir.filename_by_attribute("cc_thickness_image")
            logger.info(f"Saving thickness image to {thickness_image_path}")
            # note: suppress_stdout is not thread-safe! But it works fine, if only one thread uses it...
            with suppress_stdout():
                cc_mesh.snap_cc_picture(thickness_image_path)

    if not slice_cc_measures:
        logger.error("Error: No valid slices were found for postprocessing")
        raise ValueError("No valid slices were found for postprocessing")

    return slice_cc_measures, io_futures


def _resample_thickness(contour: CCContour) -> CCContour:
    """Resamples the thickness values of contour."""
    _c = copy(contour)
    _c.fill_thickness_values()
    return _c


def recon_cc_surf_measure(
    segmentation: np.ndarray[Shape2d, np.dtype[int]],
    slice_idx: int,
    affine: AffineMatrix4x4,
    ac_coords: Vector2d,
    pc_coords: Vector2d,
    num_thickness_points: int,
    subdivisions: list[float],
    subdivision_method: SubdivisionMethod,
    contour_smoothing: int,
    vox_size: tuple[float, float, float],
) -> tuple[CCMeasuresDict, ContourThickness, tuple[int, int]]:
    """Reconstruct surfaces and compute measures for a single slice for the corpus callosum.

    Parameters
    ----------
    segmentation : np.ndarray
        3D segmentation array.
    slice_idx : int
        Index of the slice to process.
    affine : AffineMatrix4x4
        4x4 affine transformation matrix.
    ac_coords : np.ndarray of shape (2,) and type float
        Anterior commissure coordinates.
    pc_coords : np.ndarray of shape (2,) and type float
        Posterior commissure coordinates.
    num_thickness_points : int
        Number of points for thickness estimation.
    subdivisions : list[float]
        List of fractions for anatomical subdivisions.
    subdivision_method : SubdivisionMethod
        Method for contour subdivision ('shape', 'vertical', 'angular', or 'eigenvector').
    contour_smoothing : int
        Gaussian sigma for contour smoothing.
    vox_size : triplet of floats
        LIA-oriented voxel size in millimeters.

    Returns
    -------
    measures : CCMeasuresDict
        Dictionary containing measurements if successful.
    contour_with_thickness : np.ndarray
        Contour points with thickness information, shape (3, N) for [x, y, thickness].
    endpoint_indices : pair of ints
        Indices of the anterior and posterior endpoints on the contour.

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
    cc_mask_slice: np.ndarray[tuple[int, int], np.dtype[bool]] = np.equal(segmentation[slice_idx], CC_LABEL)
    if not np.any(cc_mask_slice):
        raise ValueError(f"No CC found in slice {slice_idx}")
    contour, endpoint_idxs = get_endpoints(
        cc_mask_slice,
        ac_coords,
        pc_coords,
        (vox_size[1], vox_size[2]),
        return_coordinates=False,
        contour_smoothing=contour_smoothing,
    )
    contour_ras = convert_to_ras(contour, affine)

    endpoint_idxs: tuple[int, int]
    contour_with_thickness: ContourThickness
    midline_len, thickness, curvature, midline_equi, levelpaths, contour_with_thickness, endpoint_idxs = cc_thickness(
        contour_ras[1:].T,
        endpoint_idxs,
        n_points=num_thickness_points,
    )
    # thickness values in contour_with_thickness is not equally sampled, different shape
    # to compute length of paths: diff between consecutive points (N-1, 2) => norm (N-1,) => sum (1,)
    thickness_profile = np.stack([np.sum(np.linalg.norm(np.diff(x[:, :2], axis=0), axis=1)) for x in levelpaths])

    acpc_contour_coords_ras = contour_ras[:, list(endpoint_idxs)].T
    contour_in_acpc_space, ac_pt_acpc, pc_pt_acpc, rotate_back_acpc = transform_to_acpc_standard(
        contour_ras[1:],
        *acpc_contour_coords_ras[:, 1:],
    )
    cc_index = calculate_cc_index(contour_in_acpc_space)

    # Apply different subdivision methods based on user choice
    split_contours: ContourList
    if subdivision_method == "shape":
        _subdivisions = np.asarray(subdivisions)
        areas, split_contours = subsegment_midline_orthogonal(midline_equi, _subdivisions, contour_ras[1:], plot=False)
        split_contours = [transform_to_acpc_standard(split_contour, *acpc_contour_coords_ras[:, 1:])[0]
                          for split_contour in split_contours]
    elif subdivision_method == "vertical":
        areas, split_contours = subdivide_contour(contour_in_acpc_space, subdivisions, plot=False)
    elif subdivision_method == "angular":
        if not np.allclose(np.diff(subdivisions), np.diff(subdivisions)[0]):
            raise ValueError(
                f"Angular subdivision method (Hampel) only supports equidistant subdivision, "
                f"but got: {subdivisions}. No measures are computed.",
            )
        areas, split_contours = hampel_subdivide_contour(contour_in_acpc_space, num_rays=len(subdivisions), plot=False)
    elif subdivision_method == "eigenvector":
        pt0, pt1 = get_primary_eigenvector(contour_in_acpc_space)
        contour_eigen, _, _, rotate_back_eigen = transform_to_acpc_standard(contour_in_acpc_space, pt0, pt1)
        ac_pt_eigen, _, _, _ = transform_to_acpc_standard(ac_pt_acpc[:, None], pt0, pt1)
        ac_pt_eigen = ac_pt_eigen[:, 0]
        areas, split_contours = subdivide_contour(contour_eigen, subdivisions, oriented=True, hline_anchor=ac_pt_eigen)
        split_contours = [rotate_back_eigen(split_contour) for split_contour in split_contours]
    else:
        raise ValueError(f"Invalid subdivision method {subdivision_method}")

    total_area = np.sum(areas)
    total_perimeter = np.sum(np.sqrt(np.sum((np.diff(contour_ras[:, 1:], axis=0))**2, axis=1)))
    circularity = 4 * np.pi * total_area / (total_perimeter**2)

    # Transform split contours back to original space
    split_contours = [rotate_back_acpc(split_contour) for split_contour in split_contours]

    measures: CCMeasuresDict = {
        "cc_index": cc_index,
        "circularity": circularity,
        "areas": np.asarray(areas),
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
    return measures, contour_with_thickness, endpoint_idxs


def vectorized_line_test(
        coords_x: np.ndarray[tuple[int], np.dtype[ScalarType]],
        coords_y: np.ndarray[tuple[int], np.dtype[ScalarType]],
        line_start: Vector2d,
        line_end: Vector2d,
) -> np.ndarray[tuple[int], np.dtype[bool]]:
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
    # FIXME: rename this function to something more indicative
    # Vector from line_start to line_end
    line_vec = np.array(line_end) - np.array(line_start)
    
    # Vectors from line_start to all points (vectorized)
    point_vec_x = coords_x - line_start[0]
    point_vec_y = coords_y - line_start[1]
    
    # Cross product (vectorized): positive means point is to the left of the line
    cross_products = line_vec[0] * point_vec_y - line_vec[1] * point_vec_x
    
    return cross_products > 0


def get_unique_contour_points(split_contours: ContourList) -> list[Points2dType]:
    """Get unique contour points from the split contours.

    Parameters
    ----------
    split_contours : ContourList
        List of split contours (subsegmentations), each containing x and y coordinates, each of shape (2, N).

    Returns
    -------
    list[np.ndarray]
        List of unique contour points for each subsegment, each of shape (N, 2).

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
    unique_contour_points: list[Points2dType] = []
    
    for i, contour in enumerate(split_contours):
        # Get points for this contour
        contour_points: Points2dType = np.vstack((contour[0], -contour[1])).T  # Shape: (N,2)
        
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
    slice_shape: Shape2d,
    split_contours: ContourList,
    vox_size: tuple[float, float, float],
) -> np.ndarray[Shape2d, np.dtype[int]]:
    """Create a mask for subdividing the corpus callosum based on split contours.

    Parameters
    ----------
    slice_shape : pair of ints
        Shape of the slice (rows, cols).
    split_contours : ContourList
        List of contours defining the subdivisions.
        Each contour is a tuple of x and y coordinates.
    vox_size : pair of floats
        The voxel sizes of the image grid in AS orientation.

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
    unique_contour_points: list[Points2dType] = get_unique_contour_points(split_contours)  # shape (N, 2)
    subdivision_segments = unique_contour_points[1:]

    for s in subdivision_segments:
        if len(s) != 2:
            logger.error(f"Subdivision segment {s} has {len(s)} points, expected 2")
 
    # Create coordinate grids for all points in the slice
    rows, cols = slice_shape
    y_coords, x_coords = np.mgrid[0:rows, 0:cols]

    cc_subsegment_lut_anterior_to_posterior = SUBSEGMENT_LABELS.copy()
    cc_subsegment_lut_anterior_to_posterior.reverse()
    
    # Initialize with first segment label
    subdivision_mask = np.full(slice_shape, cc_subsegment_lut_anterior_to_posterior[0], dtype=np.int32)
    
    # Process each subdivision line
    for segment_idx, segment_points in enumerate(subdivision_segments):
        # FIXME: names for line_start and line_end?
        line_start: Vector2d = segment_points[0] / vox_size
        line_end: Vector2d = segment_points[-1] / vox_size
        
        # Vectorized test: find all points to the right of this line
        # FIXME: line defined by what? Is this inside the polygon or the line from line_start to line_end?
        points_right_of_line = vectorized_line_test(x_coords, y_coords, line_start, line_end)
        
        # All points to the right of this line belong to the next segment or beyond
        subdivision_mask[points_right_of_line] = cc_subsegment_lut_anterior_to_posterior[segment_idx + 1]

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
