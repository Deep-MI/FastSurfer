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
from typing import get_args

import numpy as np
from numpy import typing as npt

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.data.constants import CC_LABEL, SUBSEGMENT_LABELS
from CorpusCallosum.shape.contour import CCContour, contours_for_analysis_width
from CorpusCallosum.shape.curvature import calculate_curvature_metrics
from CorpusCallosum.shape.endpoint_heuristic import connect_diagonally_connected_components
from CorpusCallosum.shape.mesh import CCMesh
from CorpusCallosum.shape.metrics import calculate_cc_index
from CorpusCallosum.shape.subsegment_contour import (
    ContourList,
    get_primary_eigenvector,
    subdivide_contour_hampel,
    subdivide_contour_vertical,
    subsegment_midline_orthogonal,
    transform_to_acpc_standard,
)
from CorpusCallosum.utils.types import (
    CCMeasuresDict,
    Points2dType,
    SliceSelection,
    SubdivisionMethod,
)
from CorpusCallosum.utils.visualization import plot_contours
from FastSurferCNN.utils import AffineMatrix4x4, Image3d, Mask2d, Shape2d, Shape3d, Vector2d, nibabelHeader
from FastSurferCNN.utils.common import SubjectDirectory, update_docstring
from FastSurferCNN.utils.parallel import process_executor, thread_executor

logger = logging.get_logger(__name__)

# assert LIA orientation
LIA_ORIENTATION = np.zeros((3,3))
LIA_ORIENTATION[0,0] = -1
LIA_ORIENTATION[1,2] = 1
LIA_ORIENTATION[2,1] = -1


def offset_affine(offset: npt.ArrayLike) -> AffineMatrix4x4:
    """Generate an affine transformation matrix that only constitutes an offset (vector).

    Parameters
    ----------
    offset : array_like
        A 3-dimensional offset vector (shape (3,)) to offset with.

    Returns
    -------
    np.ndarray
        Modified 4x4 affine transformation matrix with the specific offset.

    Raises
    ------
    TypeError
        If offset is not a
    """
    _offset = np.asarray(offset)
    if not isinstance(_offset, np.ndarray) or _offset.shape != (3,):
        raise TypeError("offset must convert to a ndarray of shape (3,)!")
    vox2vox: AffineMatrix4x4 = np.eye(4, dtype=float)
    vox2vox[0:3, 3] = _offset
    return vox2vox


@update_docstring(SubdivisionMethod=str(get_args(SubdivisionMethod))[1:-1])
def recon_cc_surf_measures_multi(
    segmentation: np.ndarray[Shape3d, np.dtype[np.int_]],
    slice_selection: SliceSelection,
    orig_header: nibabelHeader,
    fsavg2midslab_vox2vox: AffineMatrix4x4,
    fsavg_vox2ras: AffineMatrix4x4,
    orig2fsavg_vox2vox: AffineMatrix4x4,
    midslices: Image3d,
    ac_coords_vox: Vector2d,
    pc_coords_vox: Vector2d,
    num_thickness_points: int,
    subdivisions: list[float],
    subdivision_method: SubdivisionMethod,
    contour_smoothing: int,
    subject_dir: SubjectDirectory,
) -> tuple[list[CCMeasuresDict], list[concurrent.futures.Future], list[CCContour]]:
    """Surface reconstruction and metrics computation of corpus callosum slices based on selection mode.

    Parameters
    ----------
    segmentation : np.ndarray
        3D segmentation array in LIA orientation.
    slice_selection : str
        Which slices to process ('middle', 'all', or slice number).
    orig_header : nibabelHeader
        Header of the conformed orig image. Surface files intended for
        inspection with orig.mgz must be written with this geometry metadata.
    fsavg2midslab_vox2vox : AffineMatrix4x4
        The vox2vox transformation matrix from fsaverage (upright) space to the segmentation slab.
    fsavg_vox2ras : np.ndarray
        Base affine transformation matrix (fsaverage, upright space).
    orig2fsavg_vox2vox : AffineMatrix4x4
        The transformation matrix from orig to fsaverage in voxel space.
    midslices : np.ndarray
        Array of mid-sagittal slices.
    ac_coords_vox : np.ndarray
        AC voxel coordinates with shape (2,) containing its [y,x] positions.
    pc_coords_vox : np.ndarray
        PC voxel coordinates with shape (2,) containing its [y,x] positions.
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

    Returns
    -------
    list of CCMeasuresDict
        List of slice processing results.
    list of concurrent.futures.Future
        List of background IO processes.
    list of CCContour
        List of CC contours.
    int
        Number of failed slices.
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
        ac_coords_vox=ac_coords_vox,
        pc_coords_vox=pc_coords_vox,
        num_thickness_points=num_thickness_points,
        subdivisions=subdivisions,
        subdivision_method=subdivision_method,
        contour_smoothing=contour_smoothing,
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

    def _gen_slice2slab_vox2vox(_slice_idx: int) -> AffineMatrix4x4:
        # The slice_idx offset must be negative, because we are going from left to right.
        return offset_affine([_slice_idx, 0, 0])

    fsavg_midslab_vox2ras = fsavg_vox2ras @ np.linalg.inv(fsavg2midslab_vox2vox)
    per_slice_vox2ras = fsavg_midslab_vox2ras @ np.stack(list(map(_gen_slice2slab_vox2vox, slices_to_recon)), axis=0)

    per_slice_recon = process_executor().map(_each_slice, slices_to_recon, per_slice_vox2ras, chunksize=1)
    cc_contours = []

    run = thread_executor().submit
    wants_output = subject_dir.has_attribute
    output_path = subject_dir.filename_by_attribute

    def _zip_failed(it_idx, it_affine, it_result):
        """Zip slice indices, affines, and results, logging errors for failed slices."""
        _sentinel = object()
        for idx, affine in zip(it_idx, it_affine, strict=True):
            try:
                result = next(it_result, _sentinel)
            except Exception as e:
                logger.error(f"Processing slice {idx} failed: {e}")
                logger.exception(e)
                yield idx, affine, (None, None)
                continue
            if result is _sentinel:
                logger.error("Number of items in idx and affine did not match results")
                return
            yield idx, affine, result

    slice_iterator = _zip_failed(slices_to_recon, per_slice_vox2ras, iter(per_slice_recon))
    for i, (slice_idx, this_slice_vox2ras, _results) in enumerate(slice_iterator):
        progress = f" ({i+1} of {num_slices})" if num_slices > 1 else ""
        # unpack values from _results
        cc_measures: CCMeasuresDict | None = _results[0]
        _contour: CCContour | None = _results[1]

        if cc_measures is None or _contour is None:
            logger.warning(f"Calculating CC measurements for slice {slice_idx+1}{progress} failed")
            continue

        logger.info(f"Calculating CC measurements for slice {slice_idx+1}{progress}")
        cc_contours.append(_contour)
        slice_cc_measures.append(cc_measures)
        is_debug = logger.getEffectiveLevel() <= logging.DEBUG
        is_midslice = i == num_slices // 2
        if wants_output("cc_qc_image") and (is_debug or is_midslice):
            qc_imgs: list[Path] = [output_path("cc_qc_image")]
            if is_debug:
                qc_slice_img = qc_imgs[0].with_suffix(f".slice_{slice_idx}.png")
                qc_imgs = (qc_imgs if is_midslice else []) + [qc_slice_img]

            logger.info(f"Saving segmentation qc image to {', '.join(map(str, qc_imgs))}")
            current_slice_in_volume = midslices.shape[0] // 2 - num_slices // 2 + slice_idx
            # Create visualization for this slice
            io_futures.append(
                run(
                    plot_contours,
                    # select the data of the current slice
                    slice_or_slab=midslices[[current_slice_in_volume]],
                    # the following need to be in voxel coordinates...
                    split_contours=cc_measures["split_contours"],
                    midline_equidistant=cc_measures["midline_equidistant"],
                    levelpaths=cc_measures["levelpaths"],
                    output_path=qc_imgs,
                    ac_coords_vox=ac_coords_vox,
                    pc_coords_vox=pc_coords_vox,
                    vox2ras=this_slice_vox2ras,
                    title=f"CC Subsegmentation by {subdivision_method} (Slice {slice_idx + 1})",
                )
            )

    if wants_output("save_template_dir"):
        template_dir = output_path("save_template_dir")
        # ensure directory exists
        template_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving template files (contours.txt, thickness_values.txt, "
                    f"thickness_measurement_points.txt) to {template_dir}")
        for j in range(len(cc_contours)):
            io_futures.append(run(cc_contours[j].save_contour, template_dir / f"contour_{j}.txt"))
            io_futures.append(run(cc_contours[j].save_thickness_values, template_dir / f"thickness_values_{j}.txt"))

    mesh_outputs = ("html", "mesh", "thickness_overlay", "surf", "thickness_image")
    if len(cc_contours) > 1 and any(wants_output(f"cc_{n}") for n in mesh_outputs):
        _cc_contours = thread_executor().map(_resample_thickness, cc_contours)
        # Surface vertices represent the continuous analysis slab. The
        # discrete segmentation mask can be slightly wider/narrower depending on
        # voxel size, so do not reuse slice-center spacing for mesh output.
        cc_mesh = CCMesh.from_contours(contours_for_analysis_width(list(_cc_contours)), smooth=1)
        if wants_output("cc_html"):
            logger.info(f"Saving CC 3D visualization to {output_path('cc_html')}")
            io_futures.append(run(cc_mesh.plot_mesh, output_path=output_path("cc_html")))

        if wants_output("cc_mesh"):
            vtk_file_path = output_path("cc_mesh")
            logger.info(f"Saving vtk file to {vtk_file_path}")
            io_futures.append(run(cc_mesh.write_vtk, vtk_file_path))

        if wants_output("cc_thickness_overlay"):
            overlay_file_path = output_path("cc_thickness_overlay")
            logger.info(f"Saving overlay file to {overlay_file_path}")
            io_futures.append(run(cc_mesh.write_morph_data, overlay_file_path))

        if any(wants_output(f"cc_{n}") for n in ("thickness_image", "surf")):
            # The mesh is generated in upright/fsaverage RAS coordinates.
            # Convert it to orig voxel coordinates, then pass orig_header to
            # write_fssurf/snap_cc_picture. Lapy interprets vertices as voxel
            # coordinates when an image/header is supplied and converts them to
            # FreeSurfer tkRAS while stamping the surface with that header's
            # volume_info. Using an upright-volume header here makes Freeview
            # place the surface relative to that volume instead of orig.mgz.
            cc_mesh_orig = cc_mesh.to_vox_coordinates(mesh_ras2vox=np.linalg.inv(fsavg_vox2ras @ orig2fsavg_vox2vox))
            if wants_output("cc_surf"):
                surf_file_path = output_path("cc_surf")
                logger.info(f"Saving surf file to {surf_file_path}")
                io_futures.append(run(cc_mesh_orig.write_fssurf, surf_file_path, image=orig_header))

            if wants_output("cc_thickness_image"):
                thickness_image_path = output_path("cc_thickness_image")
                logger.info(f"Saving thickness image to {thickness_image_path}")
                try:
                    cc_mesh_orig.snap_cc_picture(thickness_image_path, ref_header=orig_header)
                except Exception as e:
                    logger.error(
                        "Generation of the thickness image failed (see below). Please ensure that whippersnappy and "
                        "(for headless rendering) EGL libraries (libegl1) are available."
                    )
                    logger.exception(e)

        if not slice_cc_measures:
            logger.error("Error: No valid slices were found for postprocessing")
            raise ValueError("No valid slices were found for postprocessing")

    return slice_cc_measures, io_futures, cc_contours, num_slices - len(cc_contours)


def _resample_thickness(contour: CCContour) -> CCContour:
    """Resamples the thickness values of contour."""
    _c = copy(contour)
    _c.fill_thickness_values()
    return _c


def subdivide_contour(
    midline_equi: Points2dType,
    subdivisions: list[float],
    ac_pt_acpc: Vector2d,
    contour_in_acpc_space: Points2dType,
    subdivision_method: SubdivisionMethod,
) -> tuple[list[float], ContourList, np.ndarray, list[Points2dType]]:
    """Subdivide the contour based on the subdivision method.

    Parameters
    ----------
    midline_equi : Points2dType
        The midline equidistant points in ACPC space.
    subdivisions : list[float]
        The subdivisions.
    ac_pt_acpc : Vector2d
        The AC point in ACPC space (needed for eigenvector subdivision).
    contour_in_acpc_space : Points2dType
        The contour in ACPC space.
    subdivision_method : SubdivisionMethod
        The subdivision method. One of "shape", "vertical", "angular", or "eigenvector".

    Returns
    -------
    tuple[list[float], ContourList, np.ndarray, list[Points2dType]]
        The areas, split contours, split points midline, and subdivision lines.
    """
    if subdivision_method == "shape":
        _subdivisions = np.asarray(subdivisions)
        areas, split_contours, split_points_midline, subdivision_lines = subsegment_midline_orthogonal(
            midline_equi, _subdivisions, contour_in_acpc_space, plot=False
        )
    elif subdivision_method == "vertical":
        areas, split_contours, split_points_midline, subdivision_lines = subdivide_contour_vertical(
            contour_in_acpc_space, subdivisions, plot=False
        )
    elif subdivision_method == "angular":
        if not np.allclose(np.diff(subdivisions), np.diff(subdivisions)[0]):
            raise ValueError(
                f"Angular subdivision method (Hampel) only supports equidistant subdivision, "
                f"but got: {subdivisions}. No measures are computed.",
            )
        areas, split_contours, split_points_midline, subdivision_lines = subdivide_contour_hampel(
            contour_in_acpc_space, num_rays=len(subdivisions), plot=False
        )
    elif subdivision_method == "eigenvector":
        pt0, pt1 = get_primary_eigenvector(contour_in_acpc_space)
        contour_eigen, _, _, rotate_back_eigen = transform_to_acpc_standard(contour_in_acpc_space, pt0, pt1)
        ac_pt_eigen, _, _, _ = transform_to_acpc_standard(ac_pt_acpc[:, None], pt0, pt1)
        ac_pt_eigen = ac_pt_eigen[:, 0]
        areas, split_contours, split_points_midline, subdivision_lines = subdivide_contour_vertical(
            contour_eigen, subdivisions, oriented=True, hline_anchor=ac_pt_eigen
        )
        
        # Transform from the outputs back to the input space
        split_contours = [rotate_back_eigen(split_contour) for split_contour in split_contours]
        subdivision_lines = [rotate_back_eigen(line.T).T for line in subdivision_lines]
        split_points_midline = rotate_back_eigen(np.asarray(split_points_midline).T).T
    else:
        raise ValueError(f"Invalid subdivision method {subdivision_method}")

    return areas, split_contours, split_points_midline, subdivision_lines


def recon_cc_surf_measure(
    segmentation: np.ndarray[Shape3d, np.dtype[np.int_]],
    slice_idx: int,
    slice_lia_vox2midslice_ras: AffineMatrix4x4,
    ac_coords_vox: Vector2d,
    pc_coords_vox: Vector2d,
    num_thickness_points: int,
    subdivisions: list[float],
    subdivision_method: SubdivisionMethod,
    contour_smoothing: int,
) -> tuple[CCMeasuresDict, CCContour]:
    """Reconstruct surfaces and compute measures for a single slice for the corpus callosum.

    Parameters
    ----------
    segmentation : np.ndarray
        3D segmentation array.
    slice_idx : int
        Index of the slice to process.
    slice_lia_vox2midslice_ras : AffineMatrix4x4
        4x4 affine transformation matrix.
    ac_coords_vox : np.ndarray
        AC voxel coordinates with shape (2,) containing its [y,x] positions.
    pc_coords_vox : np.ndarray
        PC voxel coordinates with shape (2,) containing its [y,x] positions.
    num_thickness_points : int
        Number of points for thickness estimation.
    subdivisions : list[float]
        List of fractions for anatomical subdivisions.
    subdivision_method : SubdivisionMethod
        Method for contour subdivision ('shape', 'vertical', 'angular', or 'eigenvector').
    contour_smoothing : int
        Gaussian sigma for contour smoothing.

    Returns
    -------
    measures : CCMeasuresDict
        Dictionary containing measurements.
    contour : CCContour
        The contour object containing points, thickness values, and endpoint indices.

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
    cc_mask_slice: Mask2d = np.equal(segmentation[slice_idx], CC_LABEL)
    if not np.any(cc_mask_slice):
        raise ValueError(f"No CC found in slice {slice_idx}")
    # clean up cc mask
    cc_mask = connect_diagonally_connected_components(cc_mask_slice)
    # create a CCContour from the cc_mask and transform to RAS coordinates
    # - R coordinate is stored in _contour.z_position
    # - AS coordinates are stored in _contour.points
    _contour = CCContour.from_mask_and_acpc(
        cc_mask, ac_coords_vox, pc_coords_vox,
        slice_vox2ras=slice_lia_vox2midslice_ras, contour_smoothing=contour_smoothing,
    )

    levelpaths, thickness, midline_len, midline_equi, contour_with_thickness, endpoint_idxs, curvature = \
        _contour.create_levelpaths(num_thickness_points, inplace=True)

    contour_as = _contour.points.T
    # thickness values in contour_with_thickness is not equally sampled, different shape
    # to compute length of paths: diff between consecutive points (N-1, 2) => norm (N-1,) => sum (1,)
    thickness_profile = np.stack([np.sum(np.linalg.norm(np.diff(x[:, :2], axis=0), axis=1)) for x in levelpaths])

    acpc_contour_coords_as = contour_as[:, list(endpoint_idxs)].T
    contour_in_acpc_space, ac_pt_acpc, pc_pt_acpc, rotate_back_acpc = transform_to_acpc_standard(
        contour_as,
        *acpc_contour_coords_as,
    )
    cc_index = calculate_cc_index(contour_in_acpc_space)

    # Apply different subdivision methods based on user choice
    split_contours: ContourList
    subdivision_lines: list[Points2dType]
    split_points_midline: np.ndarray | None = None

    # Transform midline to ACPC space for subdivision
    midline_acpc, _, _, _ = transform_to_acpc_standard(
        midline_equi.T,
        *acpc_contour_coords_as,
    )

    areas, split_contours, split_points_midline, subdivision_lines = subdivide_contour(
        midline_acpc.T, subdivisions, ac_pt_acpc, contour_in_acpc_space, subdivision_method
    )

    total_area = _contour.area
    total_perimeter = np.sum(_contour.get_contour_edge_lengths())
    circularity = 4 * np.pi * total_area / (total_perimeter**2)

    # Transform split contours back to original space (from ACPC to RAS)
    split_contours = [rotate_back_acpc(split_contour) for split_contour in split_contours]
    subdivision_lines = [rotate_back_acpc(line.T).T for line in subdivision_lines]
    split_points_midline = rotate_back_acpc(np.asarray(split_points_midline).T).T

    # Calculate curvature metrics
    curvature, curvature_body, curvature_subsegments = calculate_curvature_metrics(
        midline_equi, split_points=split_points_midline
    )

    measures: CCMeasuresDict = {
        "cc_index": cc_index,
        "circularity": circularity,
        "areas": np.asarray(areas),
        "midline_length": midline_len,
        "thickness": thickness,
        "curvature": curvature,
        "curvature_subsegments": curvature_subsegments,
        "curvature_body": curvature_body,
        "thickness_profile": thickness_profile,
        "total_area": total_area,
        "total_perimeter": total_perimeter,
        "split_contours": split_contours,
        "subdivision_lines": subdivision_lines,
        "midline_equidistant": midline_equi,
        "levelpaths": levelpaths,
        "slice_index": slice_idx
    }
    return measures, _contour


def test_left_of_line(
        coords: Points2dType,
        line_start: Vector2d,
        line_end: Vector2d,
) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    """Test whether points in coords are to the left of the line (line_start->line_end).

    Parameters
    ----------
    coords : np.ndarray
        Array of coordinates of shape (..., N).
    line_start : array-like
        [x, y] coordinates of line start point (N,).
    line_end : array-like
        [x, y] coordinates of line end point (N,).

    Returns
    -------
    np.ndarray
        Boolean array where True means point is to the left of the line of shape coords.shape[:-1].
    """
    # Vector from line_start to line_end
    line_start_arr = np.expand_dims(line_start, axis=np.arange(line_start.ndim, coords.ndim).tolist())
    line_vec = np.expand_dims(line_end, axis=np.arange(line_end.ndim, coords.ndim).tolist()) - line_start_arr
    
    # Vectors from line_start to all points (vectorized)
    point_vec = np.moveaxis(coords, -1, 0) - line_start_arr

    # Cross product (vectorized): positive means point is to the left of the line
    cross_products = line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]
    
    return np.greater(cross_products, 0)


def make_subdivision_mask(
    slice_shape: Shape2d,
    subdivision_lines: list[Points2dType],
    vox2ras: AffineMatrix4x4,
    plot: bool = False,
) -> np.ndarray[Shape2d, np.dtype[np.int_]]:
    """Create a mask for subdividing the corpus callosum based on split contours.

    Parameters
    ----------
    slice_shape : pair of ints
        Shape of the slice (rows, cols).
    subdivision_lines : list[np.ndarray]
        List of pairs of points defining the subdivision lines.
    vox2ras : AffineMatrix4x4
        The vox2ras transformation matrix for the requested shape.
    plot : bool, default=False
        Whether to plot the subdivision mask.

    Returns
    -------
    np.ndarray
        A mask of shape slice_shape where each pixel is labeled with a value
        from SUBSEGEMNT_LABELS indicating which subdivision segment it belongs to.

    Notes
    -----
    The function:
    1. Creates coordinate grids for all points in the slice.
    2. Initializes mask with first segment label.
    3. For each subdivision line:
    - Tests which points lie to the right of the line.
    - Updates labels for those points.
    """
    from nibabel.affines import apply_affine

    for s in subdivision_lines:
        if len(s) != 2:
            logger.error(f"Subdivision segment {s} has {len(s)} points, expected 2")
 
    # Create coordinate grids for all points in the slice
    rows, cols = slice_shape
    coords_vox = np.stack(np.mgrid[0:1, 0:rows, 0:cols], axis=-1)
    coords_ras = apply_affine(vox2ras, coords_vox)

    # Use only as many labels as needed based on the number of subdivisions
    # Number of regions = number of division lines + 1
    num_labels_needed = len(subdivision_lines) + 1
    cc_labels_anterior_to_posterior = SUBSEGMENT_LABELS[:num_labels_needed][::-1]

    # Initialize with first segment label
    subdivision_mask = np.full(slice_shape, cc_labels_anterior_to_posterior[0], dtype=np.int32)
    # Process each subdivision line, subdivision_lines has for each division line the two points that are on the
    # contour and divide the subsegments
    for label, segment_points in zip(cc_labels_anterior_to_posterior[1:], subdivision_lines, strict=True):
        # line_start and line_end are the intersection points of the CC subsegmentation boundary and the contour line
        line_start, line_end = segment_points

        # --> find all voxels posterior to the line in question
        # Vectorized test: find all points to the right of line (line_start->line_end)
        # right_of_line == posterior to line
        points_left_of_line = test_left_of_line(coords_ras[0, ..., 1:], line_start, line_end)
        
        # All points to the right of this line belong to the next segment or beyond
        subdivision_mask[points_left_of_line] = label
        
        if plot: # interactive debug plot
            import matplotlib.pyplot as plt

            from FastSurferCNN.utils.plotting import backend
            with backend("qtagg"):
                plt.figure(figsize=(10, 8))
                plkwargs = {f"v{op}": getattr(np, op)(cc_labels_anterior_to_posterior) for op in ("min", "max")}
                plt.imshow(subdivision_mask, cmap='tab10', **plkwargs)
                plt.colorbar(label='Subdivision')
                plt.title('CC Subdivision Mask')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.tight_layout()
                plt.show()
    return subdivision_mask


def check_area_changes(contours: list[np.ndarray], threshold: float = 0.3) -> bool:
    """Check for large changes between consecutive CC areas and issue warnings.
    
    Parameters
    ----------
    contours : list[np.ndarray]
        List of contours (2, N).
    threshold : float, default=0.3
        Threshold for relative change.

    Returns
    -------
    bool
        True if no large area changes are detected, False otherwise.
    """

    # support numpy <2 and >=2
    if hasattr(np, 'trapezoid'):
        areas = np.asarray([np.abs(np.trapezoid(c[1], c[0])) for c in contours])
    else:
        areas = np.asarray([np.abs(np.trapz(c[1], c[0])) for c in contours])  # noqa: NPY201

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
