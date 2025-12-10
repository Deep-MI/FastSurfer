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
from typing import Literal, overload

import numpy as np
import scipy.interpolate
from lapy import Solver, TriaMesh
from lapy.diffgeo import compute_rotated_f
from meshpy import triangle

from CorpusCallosum.utils.types import ContourThickness, Points2dType
from FastSurferCNN.utils.common import suppress_stdout


def compute_curvature(path: Points2dType) -> np.ndarray[tuple[int], np.dtype[float]]:
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


@overload
def convert_to_ras(contour: np.ndarray, vox2ras_matrix: np.ndarray, get_parameters: Literal[False] = False) \
        -> np.ndarray: ...

@overload
def convert_to_ras(contour: np.ndarray, vox2ras_matrix: np.ndarray, get_parameters: Literal[True]) \
        -> tuple[np.ndarray, bool, bool, bool]: ...


def convert_to_ras(
    contour: np.ndarray,
    vox2ras_matrix: np.ndarray,
    return_parameters: bool = False
):
    """Convert contour coordinates from voxel space to RAS space.

    Parameters
    ----------
    contour : np.ndarray
        Array of shape (2, N) or (3, N) containing contour coordinates.
    vox2ras_matrix : np.ndarray
        4x4 voxel to RAS transformation matrix.
    return_parameters : bool, default=False
        If True, return additional transformation parameters (see below).

    Returns
    -------
    contour : np.ndarray
        Transformed contour coordinates of shape (3, N).
    anterior_reversed : bool
        Only if return_parameters is True, whether anterior axis was reversed.
    superior_reversed : bool
        Only if return_parameters is True, whether superior axis was reversed.
    swap_axes : bool
        Only if return_parameters is True, whether axes were swapped.
    """
    # converting to AS (no left-right dimension), out of plane movement is ignored,
    # so we only do scaling, axes swapping and flipping - no rotation
    # translation is ignored
    if contour.shape[0] == 2:
        # get only axis swaps from the rotation part of the vox2ras matrix
        axis_swaps = np.round(vox2ras_matrix[:3, :3], 0)
        permutation = np.argwhere(axis_swaps != 0)[:, 1]
        assert len(permutation) == 3

        idx_superior = np.argwhere(permutation == 2)
        idx_anterior = np.argwhere(permutation == 1)

        # swap axes if indicated from vox2ras
        if swap_axes := idx_anterior > idx_superior:
            # swap anterior and superior
            contour = contour[[1, 0]]

        # determine if axis were reversed
        superior_reversed = np.any(axis_swaps[2, :] == -1)
        anterior_reversed = np.any(axis_swaps[1, :] == -1)

        # flip axes if necessary
        if superior_reversed:
            contour[1] = -contour[1]
        if anterior_reversed:
            contour[0] = -contour[0]

        # get scaling by getting length of three column vectors
        scaling = np.linalg.norm(vox2ras_matrix[:3, :3], axis=0)

        # voxel * vox_size = mm
        contour = (contour.T * scaling[1:]).T

        # append a 0-R coordinate
        contour = np.concatenate([np.zeros((1, contour.shape[1])), contour], axis=0)

        if return_parameters:
            return contour, anterior_reversed, superior_reversed, swap_axes
        else:
            return contour

    # Add a third dimension (z) with 0 and a fourth dimension (homogeneous coordinate) with 1
    elif contour.shape[0] == 3:
        contour_homogeneous = np.vstack([contour, np.ones(contour.shape[1])])

        # Apply the transformation
        contour = (vox2ras_matrix @ contour_homogeneous)[:3, :]
        return contour
    else:
        raise ValueError("Invalid shape of contour")


def set_contour_zero_idx(contour, idx, anterior_endpoint_idx, posterior_endpoint_idx):
    """Roll contour points to set a new zero index, while keeping track of CC endpoints.

    Parameters
    ----------
    contour : np.ndarray
        Array of contour points.
    idx : int
        New zero index.
    anterior_endpoint_idx : int
        Index of anterior endpoint.
    posterior_endpoint_idx : int
        Index of posterior endpoint.

    Returns
    -------
    contour : np.ndarray
        Rolled contour points.
    anterior_endpoint_idx : int
        Updated anterior endpoint index.
    posterior_endpoint_idx : int
        Updated posterior endpoint index.
"""
    contour = np.roll(contour, -idx, axis=0)
    anterior_endpoint_idx = (anterior_endpoint_idx - idx) % contour.shape[0]
    posterior_endpoint_idx = (posterior_endpoint_idx - idx) % contour.shape[0]
    return contour, anterior_endpoint_idx, posterior_endpoint_idx


def find_closest_edge(point, contour):
    """Find the index of the edge closest to the given point.

    Parameters
    ----------
    point : np.ndarray
        2D point coordinates.
    contour : np.ndarray
        Array of shape (N, 2) containing contour points.

    Returns
    -------
    int
        Index of the closest edge.
    """
    edges_start = contour[:-1, :2]  # N-1 x 2
    edges_end = contour[1:, :2]  # N-1 x 2
    edges_vec = edges_end - edges_start  # N-1 x 2

    # Calculate projection coefficient for all edges at once
    # (p-a)·(b-a) / |b-a|²
    edge_lengths_sq = np.sum(edges_vec * edges_vec, axis=1)
    # Avoid division by zero for degenerate edges
    valid_edges = edge_lengths_sq > 1e-10
    t = np.zeros(len(edges_start))
    t[valid_edges] = (
        np.sum((point - edges_start[valid_edges]) * edges_vec[valid_edges], axis=1)
        / edge_lengths_sq[valid_edges]
    )
    t = np.clip(t, 0, 1)  # Clamp to edge endpoints

    # Get closest points on all edges
    closest_points = edges_start + t[:, None] * edges_vec

    # Calculate distances to all edges
    distances = np.linalg.norm(point - closest_points, axis=1)

    # Return index of closest edge
    return np.argmin(distances)


@overload
def insert_point_with_thickness(
    contour_in_as_space: np.ndarray,
    contour_thickness: np.ndarray,
    point: np.ndarray,
    thickness_value: float,
    return_index: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def insert_point_with_thickness(
    contour_in_as_space: np.ndarray,
    contour_thickness: np.ndarray,
    point: np.ndarray,
    thickness_value: float,
    return_index: Literal[True],
) -> tuple[np.ndarray, np.ndarray, int] | list[np.ndarray, np.ndarray]:
    ...


def insert_point_with_thickness(
    contour_in_as_space: np.ndarray,
    contour_thickness: np.ndarray,
    point: np.ndarray,
    thickness_value: float,
    return_index: bool = False
) -> tuple[np.ndarray, np.ndarray, int] | tuple[np.ndarray, np.ndarray]:
    """Inserts a point and its thickness value into the contour.

    Parameters
    ----------
    contour_in_as_space : np.ndarray
        Array of coordinates of the contour in AS space, shape (N, 2).
    contour_thickness : np.ndarray
        Array of thickness values of the contour, shape (N,).
    point : np.ndarray
        2D point to insert, shape (2,).
    thickness_value : float
        Thickness value corresponding to the point.
    return_index : bool, default=False
        If True, return the index where point was inserted, by default False.

    Returns
    -------
    contour_in_as_space : np.ndarray
        Updated contour of shape (N+1, 2).
    contour_thickness : np.ndarray
        Updated thickness values of shape (N+1,).
    insertion_index : int
        The index, where the point was inserted (only if return_index is True).
    """
    # Find closest edge for the point
    edge_idx = find_closest_edge(point, contour_in_as_space)

    # Insert point between edge endpoints
    contour_in_as_space = np.insert(contour_in_as_space, edge_idx + 1, point, axis=0)
    contour_thickness = np.insert(contour_thickness, edge_idx + 1, thickness_value)

    if return_index:
        return contour_in_as_space, contour_thickness, edge_idx + 1
    else:
        return contour_in_as_space, contour_thickness


def make_mesh_from_contour(
    contour_2d: np.ndarray,
    max_volume: float = 0.5,
    min_angle: float = 25,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Create a triangular mesh from a 2D contour.

    Parameters
    ----------
    contour_2d : np.ndarray
        Array of shape (N, 2) containing contour points.
    max_volume : float, optional
        Maximum triangle area, by default 0.5.
    min_angle : float, optional
        Minimum angle in triangles (degrees), by default 25.
    verbose : bool, optional
        Whether to print mesh generation info, by default False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - mesh_points : Array of shape (M, 2) containing mesh vertices.
        - mesh_trias : Array of shape (K, 3) containing triangle indices.

    Notes
    -----
    Uses meshpy.triangle to create a constrained Delaunay triangulation
    of the contour. The contour must not have duplicate points.
    """

    facets = np.vstack((np.arange(len(contour_2d)), ((np.arange(len(contour_2d)) + 1) % len(contour_2d)))).T

    # use meshpy to create mesh
    info = triangle.MeshInfo()
    info.set_points(contour_2d)
    info.set_facets(facets)
    # NOTE: crashes if contour has duplicate points !!
    mesh = triangle.build(info, max_volume=max_volume, min_angle=min_angle, verbose=verbose)

    mesh_points = np.array(mesh.points)
    mesh_trias = np.array(mesh.elements)

    return mesh_points, mesh_trias


def cc_thickness(
    contour_2d: Points2dType,
    endpoint_idx: tuple[int, int],
    n_points: int = 100,
) -> tuple[float, float, float, Points2dType , list[Points2dType], ContourThickness, tuple[int, int]]:
    """Calculate corpus callosum thickness using Laplace equation.

    Parameters
    ----------
    contour_2d : np.ndarray
        Array of shape (N, 2) containing contour points.
    endpoint_idx : pair of ints
        Indices of anterior and posterior endpoints in contour.
    n_points : int, default=100
        Number of points for thickness measurement.

    Returns
    -------
    midline_length : float
        Total length of the midline.
    thickness : float
        Mean thickness across all level paths.
    curvature : float
        Mean absolute curvature in degrees.
    midline_equidistant : np.ndarray
        Equidistant points along the midline in same space as contour2d of shape (N, 2).
    levelpaths : list[np.ndarray]
        Level paths for thickness measurement in same space as contour2d, each of shape (N, 2).
    contour_with_thickness : np.ndarray
        Contour coordinates with thickness information in same space as contour2d of shape (N+2, 3).
    endpoint_indices : pair of ints
        Pair of updated indices of anterior and posterior endpoint.

    Notes
    -----
    Uses the Laplace equation to compute thickness by:
    1. Creating a triangular mesh from the contour
    2. Setting boundary conditions (0 at endpoints, ±1 on sides)
    3. Solving Laplace equation to get level sets
    4. Computing thickness along level sets
    """
    anterior_endpoint_idx, posterior_endpoint_idx = endpoint_idx

    # standardize contour indices to start at anterior_endpoint_idx, to get consistent levelpath directions
    contour_2d, anterior_endpoint_idx, posterior_endpoint_idx = set_contour_zero_idx(
        contour_2d, anterior_endpoint_idx, anterior_endpoint_idx, posterior_endpoint_idx,
    )

    mesh_points_contour_space, mesh_trias = make_mesh_from_contour(contour_2d)

    # make points 3D by appending z=0, asz space therefore is the contour space (usually AS space) with a zero z-dim
    mesh_points_asz = np.append(mesh_points_contour_space, np.zeros((mesh_points_contour_space.shape[0], 1)), axis=1)

    # compute poisson
    with suppress_stdout():
        tria_asz = TriaMesh(mesh_points_asz, mesh_trias)
        # extract boundary curve
        bdr = np.array(tria_asz.boundary_loops()[0])

        # find index of endpoints in bdr list
        iidx1 = np.where(bdr == anterior_endpoint_idx)[0][0]
        iidx2 = np.where(bdr == posterior_endpoint_idx)[0][0]

        # create boundary condition (0 at endpoints, -1 on one side, 1 on the other):
        if iidx1 > iidx2:
            iidx1, iidx2 = iidx2, iidx1
        dcond = np.ones(bdr.shape)
        dcond[iidx1] = 0
        dcond[iidx2] = 0
        dcond[iidx1 + 1 : iidx2] = -1

        # Extract path
        fem = Solver(tria_asz)
        vfunc = fem.poisson(0, (bdr, dcond))
        midline_length: float
        midline_equidistant_asz, midline_length = tria_asz.level_path(vfunc, level=0., n_points=n_points + 2)
        midline_equidistant_contour_space: np.ndarray = midline_equidistant_asz[:, :2]

        gf = compute_rotated_f(tria_asz, vfunc)

        # interpolate midline to get levels to evaluate
        level_of_rotated_laplace_contour_space = scipy.interpolate.griddata(
            tria_asz.v[:, 0:2], gf, midline_equidistant_asz[:, 0:2], method="cubic",
        )

    # get levels to evaluate
    levelpaths_contour_space: list[Points2dType] = []
    levelpath_lengths = []
    levelpath_tria_idx = []

    # now, on the rotated laplace function, sample equally spaced (on midline: level_of_rotated_laplace) levelpaths
    contour_thickness = np.full(contour_2d.shape[0], np.nan)
    for current_level in level_of_rotated_laplace_contour_space[1:-1]:
        # levelpath starts at index zero
        levelpath_asz, lvlpath_length, tria_idx = tria_asz.level_path(gf, current_level, get_tria_idx=True)

        levelpaths_contour_space.append(levelpath_asz[:, :2])
        levelpath_lengths.append(lvlpath_length)
        levelpath_tria_idx.append(tria_idx)

        levelpath_start = levelpath_asz[0, :2]
        levelpath_end = levelpath_asz[-1, :2]

        contour_2d, contour_thickness, inserted_idx_start = insert_point_with_thickness(
            contour_2d, contour_thickness, levelpath_start, lvlpath_length, return_index=True,
        )
        # keep track of start index
        if inserted_idx_start <= anterior_endpoint_idx:
            anterior_endpoint_idx += 1
        if inserted_idx_start >= posterior_endpoint_idx:
            posterior_endpoint_idx += 1

        contour_2d, contour_thickness, inserted_idx_end = insert_point_with_thickness(
            contour_2d, contour_thickness, levelpath_end, lvlpath_length, return_index=True,
        )
        # keep track of end index
        if inserted_idx_end <= anterior_endpoint_idx:
            anterior_endpoint_idx += 1
        if inserted_idx_end >= posterior_endpoint_idx:
            posterior_endpoint_idx += 1

    contour_2d_with_thickness = np.concatenate([contour_2d, contour_thickness[:, None]], axis=1)

    # get curvature of path3d_resampled
    curvature = compute_curvature(midline_equidistant_contour_space)
    mean_curvature: float = np.abs(np.degrees(np.mean(curvature))).item() / len(curvature)
    mean_thickness: float = np.mean(levelpath_lengths).item()
    endpoints: tuple[int, int] = (anterior_endpoint_idx, posterior_endpoint_idx)

    return (
        midline_length,
        mean_thickness,
        mean_curvature,
        midline_equidistant_contour_space,
        levelpaths_contour_space,
        contour_2d_with_thickness,
        endpoints,
    )
