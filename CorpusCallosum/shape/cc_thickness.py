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

from CorpusCallosum.utils.utils import HiddenPrints


def compute_curvature(path: np.ndarray) -> np.ndarray:
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
    contour : p.ndarray
        Transformed contour coordinates.
    anterior_reversed : bool
        Only if return_parameters is True, whether anterior axis was reversed.
    superior_reversed : bool
        Only if return_parameters is True, whether superior axis was reversed.
    swap_axes : bool
        Only if return_parameters is True, whether axes were swapped.
    """
    # converting to AS (no left-right dimension), out of plane movement is ignores,
    # so we only do scaling, axes swapping and flipping - no rotation
    # translation is ignored
    if contour.shape[0] == 2:
        # get only axis swaps
        axis_swaps = np.round(vox2ras_matrix[:3, :3], 0)
        permutation = np.argwhere(axis_swaps != 0)[:, 1]
        assert len(permutation) == 3

        idx_superior = np.argwhere(permutation == 2)
        idx_anterior = np.argwhere(permutation == 1)

        swap_axes = idx_anterior > idx_superior
        if swap_axes:
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


def insert_point_with_thickness(
    contour_with_thickness: list[np.ndarray],
    point: np.ndarray,
    thickness_value: float,
    get_index: bool = False
) -> tuple[list[np.ndarray], int] | list[np.ndarray]:
    """Insert a point and its thickness value into the contour.

    Parameters
    ----------
    contour_with_thickness : list[np.ndarray]
        List containing [contour_points, thickness_values].
    point : np.ndarray
        2D point to insert, shape (2,).
    thickness_value : float
        Thickness value corresponding to the point.
    get_index : bool, optional
        If True, return the index where point was inserted, by default False.

    Returns
    -------
    tuple[list[np.ndarray], int] or list[np.ndarray]
        If get_index is True:
            - Updated contour_with_thickness.
            - Index where point was inserted.
        If get_index is False:
            - Updated contour_with_thickness.
    """
    # Find closest edge for the point
    edge_idx = find_closest_edge(point, contour_with_thickness[0])

    # Insert point between edge endpoints
    contour_with_thickness[0] = np.insert(
        contour_with_thickness[0], edge_idx + 1, point, axis=0
    )
    contour_with_thickness[1] = np.insert(
        contour_with_thickness[1], edge_idx + 1, thickness_value
    )

    if get_index:
        return contour_with_thickness, edge_idx + 1
    else:
        return contour_with_thickness


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

    facets = np.vstack(
        (
            np.arange(len(contour_2d)),
            ((np.arange(len(contour_2d)) + 1) % len(contour_2d)),
        )
    ).T

    # plot vertices and facets
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax.scatter(contour_2d[:,0], contour_2d[:,1], label='Contour')
    # ax.plot(contour_2d[:,0], contour_2d[:,1], 'k-', label='Contour')
    # ax.plot(contour_2d[facets[:,0],0], contour_2d[facets[:,0],1], 'r-', label='Facets')
    # plt.show()

    # use meshpy to create mesh
    info = triangle.MeshInfo()
    info.set_points(contour_2d)
    info.set_facets(facets)
    # NOTE: crashes if contour has duplicate points !!
    mesh = triangle.build(
        info, max_volume=max_volume, min_angle=min_angle, verbose=verbose
    )

    mesh_points = np.array(mesh.points)
    mesh_trias = np.array(mesh.elements)

    return mesh_points, mesh_trias


def cc_thickness(
    contour_2d: np.ndarray,
    anterior_endpoint_idx: int,
    posterior_endpoint_idx: int,
    n_points: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate corpus callosum thickness using Laplace equation.

    Parameters
    ----------
    contour_2d : np.ndarray
        Array of shape (N, 2) containing contour points.
    anterior_endpoint_idx : int
        Index of anterior endpoint in contour.
    posterior_endpoint_idx : int
        Index of posterior endpoint in contour.
    n_points : int, optional
        Number of points for thickness measurement, by default 100.

    Returns
    -------
    thickness_values : np.ndarray
        Array of thickness measurements.
    measurement_points : np.ndarray
        Array of points where thickness was measured.

    Notes
    -----
    Uses the Laplace equation to compute thickness by:
    1. Creating a triangular mesh from the contour
    2. Setting boundary conditions (0 at endpoints, ±1 on sides)
    3. Solving Laplace equation to get level sets
    4. Computing thickness along level sets
    """

    # standardize contour indices, to get consistent levelpath directions
    contour_2d, anterior_endpoint_idx, posterior_endpoint_idx = set_contour_zero_idx(
        contour_2d, anterior_endpoint_idx, anterior_endpoint_idx, posterior_endpoint_idx
    )

    mesh_points, mesh_trias = make_mesh_from_contour(contour_2d)

    # plot mesh points with index next to point
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax.plot(mesh_points[:,0], mesh_points[:,1], label='Mesh Points')
    # for i in range(len(mesh_points)):
    #     ax.text(mesh_points[i,0], mesh_points[i,1], str(i), fontsize=7)
    # plt.show()

    # make points 3D by appending z=0
    mesh_points3d = np.append(mesh_points, np.zeros((mesh_points.shape[0], 1)), axis=1)

    # compute poisson
    with HiddenPrints():
        tria = TriaMesh(mesh_points3d, mesh_trias)
    # extract boundary curve
    bdr = np.array(tria.boundary_loops()[0])

    # find index of endpoints in bdr list
    iidx1 = np.where(bdr == anterior_endpoint_idx)[0][0]
    iidx2 = np.where(bdr == posterior_endpoint_idx)[0][0]

    # create boundary condition (0 at endpoints, -1 on one side, 1 on the other):
    if iidx1 > iidx2:
        tmp = iidx2
        iidx2 = iidx1
        iidx1 = tmp
    dcond = np.ones(bdr.shape)
    dcond[iidx1] = 0
    dcond[iidx2] = 0
    dcond[iidx1 + 1 : iidx2] = -1

    # Extract path
    with HiddenPrints():
        fem = Solver(tria)
        vfunc = fem.poisson(0, (bdr, dcond))
    level = 0
    midline_equidistant, midline_length = tria.level_path(
        vfunc, level, n_points=n_points + 2
    )
    midline_equidistant = midline_equidistant[:, :2]

    # try:
    with HiddenPrints():
        gf = compute_rotated_f(tria, vfunc)
    # except Exception as e:
    # Lot contour and path
    # import matplotlib.pyplot as plt
    # import matplotlib.tri as tri
    # fig, ax = plt.subplots(figsize=(10, 8))
    # # Plot contours
    # ax.plot(contour_2d[:,0], contour_2d[:,1], 'k-', label='Contour', marker='o', markersize=3)
    # ax.plot(midline_equidistant[:,0], midline_equidistant[:,1], 'g-', label='Level0', marker='o', markersize=2)
    # # plot mesh
    # mtpltlb_tria = tri.Triangulation(tria.v[:,0], tria.v[:,1], triangles=tria.t)
    # ax.triplot(mtpltlb_tria, 'k-', alpha=0.2, linewidth=0.5)
    # # Plot final endpoint estimates
    # ax.plot(contour_2d[:,0][anterior_endpoint_idx], contour_2d[:,1][anterior_endpoint_idx], 'r*',
    #             markersize=15, label='Final estimate')
    # ax.plot(contour_2d[:,0][posterior_endpoint_idx], contour_2d[:,1][posterior_endpoint_idx], 'r*',
    #             markersize=15, label='Final estimate')
    # ax.legend()
    # #ax.set_title(f'Subject: {subj_id}')
    # plt.show()

    # interpolate midline to get levels to evaluate
    gf_interp = scipy.interpolate.griddata(
        tria.v[:, 0:2], gf, midline_equidistant[:, 0:2], method="cubic"
    )

    # get levels to evaluate
    # level_length = tria.level_length(gf, gf_interp)

    levelpaths = []
    levelpath_lengths = []
    levelpath_tria_idx = []

    contour_with_thickness = [contour_2d.copy(), np.full(contour_2d.shape[0], np.nan)]
    for i in range(1, n_points + 1):
        level = gf_interp[i]
        # levelpath starts at index zero
        lvlpath, lvlpath_length, tria_idx = tria.level_path(
            gf, level, get_tria_idx=True
        )

        levelpaths.append(lvlpath)
        levelpath_lengths.append(lvlpath_length)
        levelpath_tria_idx.append(tria_idx)

        levelpath_start = lvlpath[0, :2]
        levelpath_end = lvlpath[-1, :2]

        contour_with_thickness, inserted_idx_start = insert_point_with_thickness(
            contour_with_thickness, levelpath_start, lvlpath_length, get_index=True
        )
        contour_with_thickness, inserted_idx_end = insert_point_with_thickness(
            contour_with_thickness, levelpath_end, lvlpath_length, get_index=True
        )

        # keep track of start and end indices
        if inserted_idx_start <= anterior_endpoint_idx:
            anterior_endpoint_idx += 1
        if inserted_idx_end <= anterior_endpoint_idx:
            anterior_endpoint_idx += 1

        if inserted_idx_start >= posterior_endpoint_idx:
            posterior_endpoint_idx += 1
        if inserted_idx_end >= posterior_endpoint_idx:
            posterior_endpoint_idx += 1

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(figsize=(10, 8))
    # cont = contour_with_thickness[0]
    # ax.plot(cont[:,0], cont[:,1], 'k-', label='Contour', marker='o', markersize=3)
    # ax.scatter(cont[:,0][anterior_endpoint_idx], cont[:,1][anterior_endpoint_idx], c='r', 
    # label='Anterior Endpoint', marker='o')
    # ax.scatter(cont[:,0][posterior_endpoint_idx], cont[:,1][posterior_endpoint_idx], c='b', 
    # label='Posterior Endpoint', marker='o')
    # ax.legend()
    # plt.show()

    # thickness_measurement_points_top = []
    # thickness_measurement_points_bottom = []
    # for i in range(len(levelpaths)):
    #     thickness_measurement_points_top.append(levelpaths[i][0,:2])
    #     thickness_measurement_points_bottom.append(levelpaths[i][-1,:2])

    # thickness_measurement_points_top = np.array(thickness_measurement_points_top)
    # thickness_measurement_points_bottom = np.array(thickness_measurement_points_bottom)
    # thickness_measurement_points = np.concatenate([thickness_measurement_points_top, 
    # thickness_measurement_points_bottom], axis=0).T

    # # Create a figure with subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # # Plot 1: Contour
    # ax1.plot(contour_2d[:,0], -contour_2d[:,1], 'b-', linewidth=2, label='Contour')
    # ax1.set_title('Corpus Callosum Contour')
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # ax1.axis('equal')
    # ax1.invert_yaxis()
    # ax1.legend()

    # # Plot 2: Thickness measurement points
    # print(thickness_measurement_points.shape)
    # ax2.plot(thickness_measurement_points[0, :100], -thickness_measurement_points[1, :100], 'ro', 
    # markersize=3, label='Thickness Points (start)')
    # ax2.plot(thickness_measurement_points[0, 100:], -thickness_measurement_points[1, 100:], 'go', 
    # markersize=3, label='Thickness Points (end)')
    # ax2.set_title('Thickness Measurement Points')
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.axis('equal')
    # ax2.invert_yaxis()
    # ax2.legend()
    # plt.show()

    # get curvature of path3d_resampled
    curvature = compute_curvature(midline_equidistant)
    out_curvature = np.abs(np.degrees(np.mean(curvature))) / len(curvature)
    # print(f'Curvature: {out_curvature:.2f}')
    # print(f'Length of midline: ', f'{midline_length:.2f}')
    # print(f'Thickness: {np.mean(levelpath_lengths):.2f}')

    # import matplotlib.pyplot as plt
    # import matplotlib.tri as tri
    # fig, ax = plt.subplots(figsize=(5, 4))
    # mtpltlb_tria = tri.Triangulation(tria.v[:,0], tria.v[:,1], triangles=tria.t)
    # triang = plt.tricontourf(mtpltlb_tria, gf, cmap='autumn', alpha=0.2)
    # ax.plot(midline_equidistant[:,0], midline_equidistant[:,1], 'r-', label=f'Levelsets')#, marker='o', markersize=2)
    # #ax.plot(contour_2d[:,0], contour_2d[:,1], 'k-', label='Contour', alpha=0.6)

    # for i in range(len(levelpaths)):
    #     if levelpaths[i] is not None:
    #         ax.plot(levelpaths[i][:,0], levelpaths[i][:,1], 'r-', marker='o', markersize=0) # , 
    # label=f'Level {levelpath_lengths[i]:.2f}'
    # ax.plot(midline_equidistant[:,0], midline_equidistant[:,1], '-', label='Midline', alpha=1, 
    # color='darkgoldenrod')#, marker='o', markersize=2)

    # #plt.colorbar(colorscale, label='Level values')
    # # plot mesh
    # ax.triplot(tria.v[:,0], tria.v[:,1], tria.t, 'k-', alpha=0.2, linewidth=0.5)
    # #ax.scatter(path3d_resampled[99,0], path3d_resampled[99,1], c='g', s=20)

    # ax.set_aspect('equal')
    # #plt.title('Levelpath on rotated Poisson')
    # plt.legend()
    # # invert x axis
    # ax.invert_xaxis()
    # plt.tight_layout()
    # plt.axis('off')
    # plt.savefig(f'levelsets.png', dpi=300, bbox_inches='tight')
    # plt.show()

    return (
        midline_length,
        np.mean(levelpath_lengths),
        out_curvature,
        midline_equidistant,
        levelpaths,
        contour_with_thickness,
        anterior_endpoint_idx,
        posterior_endpoint_idx,
    )
