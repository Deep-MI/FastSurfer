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
from nibabel.freesurfer.mghformat import MGHHeader
from numpy import typing as npt

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.data.constants import CC_LABEL, SUBSEGMENT_LABELS
from CorpusCallosum.shape.contour import CCContour
from CorpusCallosum.shape.endpoint_heuristic import connect_diagonally_connected_components
from CorpusCallosum.shape.mesh import CCMesh
from CorpusCallosum.shape.metrics import calculate_cc_index
from CorpusCallosum.shape.subsegment_contour import (
    ContourList,
    get_primary_eigenvector,
    hampel_subdivide_contour,
    subdivide_contour,
    subsegment_midline_orthogonal,
    transform_to_acpc_standard,
)
from CorpusCallosum.shape.thickness import cc_thickness
from CorpusCallosum.utils.types import CCMeasuresDict, ContourThickness, Points2dType, SliceSelection, SubdivisionMethod
from CorpusCallosum.utils.visualization import plot_contours
from FastSurferCNN.utils import AffineMatrix4x4, Image3d, Mask2d, Shape2d, Shape3d, Vector2d
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
    upright_header: MGHHeader,
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
) -> tuple[list[CCMeasuresDict], list[concurrent.futures.Future]]:
    """Surface reconstruction and metrics computation of corpus callosum slices based on selection mode.

    Parameters
    ----------
    segmentation : np.ndarray
        3D segmentation array in LIA orientation.
    slice_selection : str
        Which slices to process ('middle', 'all', or slice number).
    upright_header : MGHHeader
        The header of the upright image.
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
    slice_iterator = zip(slices_to_recon, per_slice_vox2ras, per_slice_recon, strict=True)
    for i, (slice_idx, this_slice_vox2ras, _results) in enumerate(slice_iterator):
        progress = f" ({i+1} of {num_slices})" if num_slices > 1 else ""
        logger.info(f"Calculating CC measurements for slice {slice_idx+1}{progress}")
        # unpack values from _results
        cc_measures: CCMeasuresDict = _results[0]
        contour_in_as_space_and_thickness: ContourThickness = _results[1]
        endpoint_idxs: tuple[int, int] = _results[2]
        contour_in_as_space: Points2dType = contour_in_as_space_and_thickness[:, :2]
        thickness_values: np.ndarray[tuple[int], np.dtype[np.float_]] = contour_in_as_space_and_thickness[:, 2]

        z_value = this_slice_vox2ras[0, 3]
        cc_contours.append(CCContour(contour_in_as_space, thickness_values, endpoint_idxs, z_position=z_value))
        if cc_measures is None:
            # this should not happen, but just in case
            logger.warning(f"Slice index {slice_idx+1}{progress} returned result `None`")

        slice_cc_measures.append(cc_measures)
        is_debug = logger.getEffectiveLevel() <= logging.DEBUG
        is_midslice = slice_idx == num_slices // 2
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
        run = run
        for j in range(len(cc_contours)):
            io_futures.append(run(cc_contours[j].save_contour, template_dir / f"contour_{j}.txt"))
            io_futures.append(run(cc_contours[j].save_thickness_values, template_dir / f"thickness_values_{j}.txt"))

    mesh_outputs = ("html", "mesh", "thickness_overlay", "surf", "thickness_image")
    if len(cc_contours) > 1 and any(wants_output(f"cc_{n}") for n in mesh_outputs):
        _cc_contours = thread_executor().map(_resample_thickness, cc_contours)
        cc_mesh = CCMesh.from_contours(list(_cc_contours), smooth=1)
        if wants_output("cc_html"):
            logger.info(f"Saving CC 3D visualization to {output_path('cc_html')}")
            io_futures.append(run(cc_mesh.plot_mesh, output_path=output_path("cc_html")))

        if wants_output("cc_mesh"):
            vtk_file_path = output_path("cc_mesh")
            logger.info(f"Saving vtk file to {vtk_file_path}")
            io_futures.append(run(cc_mesh.write_vtk, vtk_file_path))

        if wants_output("cc_thickness_overlay") and not wants_output("cc_thickness_image"):
            overlay_file_path = output_path("cc_thickness_overlay")
            logger.info(f"Saving overlay file to {overlay_file_path}")
            io_futures.append(run(cc_mesh.write_morph_data, overlay_file_path))

        if any(wants_output(f"cc_{n}") for n in ("thickness_image", "surf")):
            import nibabel as nib
            up_data: Image3d[np.uint8] = np.empty(upright_header["dims"][:3], dtype=upright_header.get_data_dtype())
            upright_img = nib.MGHImage(up_data, fsavg_vox2ras, upright_header)
            # the mesh is generated in upright coordinates, so we need to also transform to orig coordinates
            # Mesh is fsavg_midplane (RAS); we need to transform to voxel coordinates
            # fsavg ras is also on the midslice, so this is fine and we multiply in the IA and SP offsets
            cc_mesh = cc_mesh.to_vox_coordinates(mesh_ras2vox=np.linalg.inv(fsavg_vox2ras @ orig2fsavg_vox2vox))
            if wants_output("cc_thickness_image"):
                # this will also write overlay and surface
                thickness_image_path = output_path("cc_thickness_image")
                logger.info(f"Saving thickness image to {thickness_image_path}")
                kwargs = {
                    "fssurf_file": output_path("cc_surf") if wants_output("cc_surf") else None,
                    "overlay_file": output_path("cc_thickness_overlay")
                                    if wants_output("cc_thickness_overlay") else None,
                    "ref_image": upright_img,
                }
                cc_mesh.snap_cc_picture(thickness_image_path, **kwargs)
            elif wants_output("cc_surf"):
                surf_file_path = output_path("cc_surf")
                logger.info(f"Saving surf file to {surf_file_path}")
                io_futures.append(run(cc_mesh.write_fssurf, str(surf_file_path), image=upright_img))

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
    segmentation: np.ndarray[Shape2d, np.dtype[np.int_]],
    slice_idx: int,
    slice_lia_vox2midslice_ras: AffineMatrix4x4,
    ac_coords_vox: Vector2d,
    pc_coords_vox: Vector2d,
    num_thickness_points: int,
    subdivisions: list[float],
    subdivision_method: SubdivisionMethod,
    contour_smoothing: int,
) -> tuple[CCMeasuresDict, ContourThickness, tuple[int, int]]:
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
        Dictionary containing measurements if successful.
    contour_with_thickness : np.ndarray
        Contour points with thickness information in fsavg_midslice_ras space, shape (3, N) for [x, y, thickness].
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

    contour_as = _contour.points.T
    endpoint_idxs = _contour.endpoint_idxs
    # FIXME: could probably also use _contour.create_levelpaths here, but that does not currently return all values
    # levelpaths, thickness = _contour.create_levelpaths(num_thickness_points)

    # FIXME: If we create CCContour objects here already (as we can), we should probably return that instead of the
    #        contour_with_thickness value (as the CCContour has all that information as well)

    # # find_contour_and_endpoints extracts the contour and finds ac and pc endpoints for shape analysis
    # # contour is in IA voxel coordinates
    # contour, endpoint_idxs = find_contour_and_endpoints(
    #     cc_mask_slice,
    #     ac_coords_vox,
    #     pc_coords_vox,
    #     (vox_size[1], vox_size[2]),
    #     return_coordinates=False,
    #     contour_smoothing=contour_smoothing,
    # )
    # # contour_ras uses coordinates in the fsavg_midslice_ras coordinate system, now re-order/flip slice_ia
    # # coordinates to fsavg_ras coordinates.
    # #FIXME: double-check the sign of the z_offset (lr) here, currently starts positive for first slice
    # offsets = np.asarray([-vox_size[0] * (slice_idx - segmentation.shape[0] // 2), 0, 0, 1])
    # affine = np.concatenate([slice_lia_vox2midslice_ras[:, :3], offsets[:, None]], axis=1)
    # # convert to fsavg_ras coordinates (which are mid-slice-based)
    # contour_as = (slice_lia_vox2midslice_ras @ np.append(contour, 1, axis=0))[1:3]

    contour_with_thickness: ContourThickness
    # cc_thickness wants contour to be in midslice_ras coordinates, i.e. millimeter distances on the respective slice.
    midline_len, thickness, curvature, midline_equi, levelpaths, contour_with_thickness, endpoint_idxs = \
        cc_thickness(
            contour_as.T,
            endpoint_idxs,
            n_points=num_thickness_points,
        )
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
    if subdivision_method == "shape":
        _subdivisions = np.asarray(subdivisions)
        areas, split_contours = subsegment_midline_orthogonal(midline_equi, _subdivisions, contour_as, plot=False)
        split_contours = [transform_to_acpc_standard(split_contour, *acpc_contour_coords_as)[0]
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
    total_perimeter = np.sum(np.sqrt(np.sum((np.diff(contour_as, axis=0))**2, axis=1)))
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


def test_right_of_line(
        coords: Points2dType,
        line_start: Vector2d,
        line_end: Vector2d,
) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    """Test whether points in coords are to the right of the line (line_start->line_end).

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
    # initialize with values for first_contour, which are by definition just "the contour" (empty)
    unique_contour_points: list[Points2dType] = [np.zeros((0, 2))]
    first_contour = split_contours[0]
    # Check each point against all other contours
    for contour in split_contours[1:]:
        # 0: coord-axis, 1: contour-axis, 2: first_contour_axis
        contour_comparison = np.isclose(first_contour[:, None], contour[:, :, None], atol=1e-6)
        # mask of contour points, that are also in first_contour (axis 1 after all)
        contour_points_in_first_contour_mask = np.any(np.all(contour_comparison, axis=0), axis=1)
        unique_contour_points.append(contour[:, ~contour_points_in_first_contour_mask].T)

    return unique_contour_points


def make_subdivision_mask(
    slice_shape: Shape2d,
    split_contours: ContourList,
    vox2ras: AffineMatrix4x4,
) -> np.ndarray[Shape2d, np.dtype[np.int_]]:
    """Create a mask for subdividing the corpus callosum based on split contours.

    Parameters
    ----------
    slice_shape : pair of ints
        Shape of the slice (rows, cols).
    split_contours : ContourList
        List of contours defining the subdivisions.
        Each contour is a tuple of x and y coordinates.
    vox2ras : AffineMatrix4x4
        The vox2ras transformation matrix for the requested shape.

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
    from nibabel.affines import apply_affine

    # unique_contour_points are the points where sub-division lines were inserted
    unique_contour_points: list[Points2dType] = get_unique_contour_points(split_contours)  # shape (N, 2)
    subdivision_segments = unique_contour_points[1:]

    for s in subdivision_segments:
        if len(s) != 2:
            logger.error(f"Subdivision segment {s} has {len(s)} points, expected 2")
 
    # Create coordinate grids for all points in the slice
    rows, cols = slice_shape
    coords_vox = np.stack(np.mgrid[0:1, 0:rows, 0:cols], axis=-1)
    coords_ras = apply_affine(vox2ras, coords_vox)

    cc_labels_posterior_to_anterior = SUBSEGMENT_LABELS

    # Initialize with first segment label
    subdivision_mask = np.full(slice_shape, cc_labels_posterior_to_anterior[0], dtype=np.int32)

    # Process each subdivision line, subdivision_segments has for each division line the two points that are on the
    # contour and divide the subsegments
    for label, segment_points in zip(cc_labels_posterior_to_anterior[1:], reversed(subdivision_segments), strict=True):
        # line_start and line_end are the intersection points of the CC subsegmentation boundary and the contour line
        line_start, line_end = segment_points

        # --> find all voxels posterior to the line in question
        # Vectorized test: find all points to the right of line (line_start->line_end)
        # right_of_line == posterior to line
        points_right_of_line = test_right_of_line(coords_ras[0, ..., 1:], line_start, line_end)
        
        # All points to the right of this line belong to the next segment or beyond
        subdivision_mask[points_right_of_line] = label
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
