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

from collections.abc import Callable
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt
from scipy.spatial import ConvexHull

_TS = TypeVar("_TS", bound=np.number)


def minimum_bounding_rectangle(points):
    """Find the smallest bounding rectangle for a set of points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing point coordinates.

    Returns
    -------
    np.ndarray
        Array of shape (4, 2) containing coordinates of the bounding box corners.
    """
    pi2 = np.pi / 2.0
    points = points.T

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2), np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def calc_subsegment_area(split_contours: list[npt.NDArray[_TS]]) -> npt.NDArray[_TS]:
    """Calculate area of each subsegment using the shoelace formula.

    Parameters
    ----------
    split_contours : list[np.ndarray]
        List of contour arrays, each of shape (2, N).

    Returns
    -------
    subsegment_areas : np.ndarray
        Array containing the area of each subsegment.
    """
    # calculate area of each split contour using the shoelace formula
    areas = np.abs([np.trapz(split_contour[1], split_contour[0]) for split_contour in split_contours])
    if len(areas) == 1:
        return np.asarray(areas[0])
    return np.ediff1d(np.asarray(areas)[::-1], to_end=areas[-1])


def subsegment_midline_orthogonal(midline, area_weights, contour, plot=True, ax=None, extremes=None):
    """Subsegment contour orthogonally to the midline based on area weights.

    Parameters
    ----------
    midline : np.ndarray
        Array of shape (N, 2) containing midline points.
    area_weights : np.ndarray
        Array of weights for area-based subdivision.
    contour : np.ndarray
        Array of shape (2, M) containing contour points.
    plot : bool, optional
        Whether to plot the results, by default True.
    ax : matplotlib.axes.Axes, optional
        Axes for plotting, by default None.
    extremes : tuple, optional
        Tuple of extreme points, by default None.

    Returns
    -------
    subsegment_area : list[np.ndarray]
    split_contours : list[np.ndarray]
        List of contour arrays for each subsegment.

    """
    # get points after midline length of splits

    # get vertex closest to midline end
    midline_end_idx = np.argmin(np.linalg.norm(contour.T - midline[-1], axis=1))
    # roll contour start to midline end
    contour = np.roll(contour, -midline_end_idx, axis=1)

    edge_idx, edge_frac = np.divmod(len(midline) * np.array(area_weights), 1)
    edge_idx = edge_idx.astype(int)
    split_points = midline[edge_idx] + (midline[edge_idx + 1] - midline[edge_idx]) * edge_frac[:, None]

    # get edge for each split point
    edge_directions = midline[edge_idx] - midline[edge_idx + 1]
    # get vector perpendicular to each midline edge
    edge_ortho_vectors = np.column_stack((-edge_directions[:, 1], edge_directions[:, 0]))
    edge_ortho_vectors = edge_ortho_vectors / np.linalg.norm(edge_ortho_vectors, axis=1)[:, None]

    split_contours = [contour]

    for pt_idx, split_point in enumerate(split_points):
        intersections = []
        for i in range(contour.shape[1] - 1):
            # get contour segment
            segment_start = contour[:, i]
            segment_end = contour[:, i + 1]
            segment_vector = segment_end - segment_start

            # Check for intersection with the perpendicular line
            matrix = np.array([segment_vector, -edge_ortho_vectors[pt_idx]]).T
            if np.linalg.matrix_rank(matrix) < 2:
                continue  # Skip parallel lines

            # Solve for intersection
            t, s = np.linalg.solve(matrix, split_point - segment_start)
            if 0 <= t <= 1:
                intersection_point = segment_start + t * segment_vector
                intersections.append((i, intersection_point))

                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.plot(contour[0], contour[1], 'k-')
                # plt.plot(midline[:,0], midline[:,1], 'k--')
                # plt.plot(split_point[0], split_point[1], 'ro')

                # plt.plot([segment_start[0], segment_end[0]], [segment_start[1], segment_end[1]], 'bo', linewidth=2)
                # plt.plot([split_point[0]-edge_ortho_vectors[pt_idx][0], split_point[0]+edge_ortho_vectors[pt_idx][0]],
                # [split_point[1]-edge_ortho_vectors[pt_idx][1], 
                # split_point[1]+edge_ortho_vectors[pt_idx][1]], 'k-', linewidth=2)
                # plt.show()

        # get the two intersections closest to split_point
        intersections.sort(key=lambda x: np.linalg.norm(x[1] - split_point))

        # Create new contours by splitting at intersections
        if intersections:
            first_index, first_intersection = intersections[1]
            second_index, second_intersection = intersections[0]

            if first_index > second_index:
                first_index, second_index = second_index, first_index
                first_intersection, second_intersection = second_intersection, first_intersection

            first_index += 1
            # second_index += 1

            # connect first and second half
            start_to_cutoff = np.hstack(
                (
                    contour[:, :first_index],
                    first_intersection[:, None],
                    second_intersection[:, None],
                    contour[:, second_index + 1 :],
                )
            )
            split_contours.append(start_to_cutoff)
        else:
            raise ValueError("No intersections found, this should not happen")

        # plot contour to first index, then split point, then contour to second index

        # import matplotlib.pyplot as plt
        # plt.close()
        # fig, ax = plt.subplots(1,1)
        # ax.plot(contour[:, :first_index][0], contour[:, :first_index][1], '-', linewidth=2, color='grey',
        # label='Contour to first index')
        # ax.plot(first_intersection[0], first_intersection[1], 'o', markersize=8, color='red',
        # label='First intersection')
        # ax.plot(second_intersection[0], second_intersection[1], 'o', markersize=8, color='red',
        # label='Second intersection')
        # ax.plot(contour[:, second_index + 1:][0], contour[:, second_index + 1:][1], '-', linewidth=2, color='red',
        # label='Contour to second index')
        # ax.legend()
        # ax.set_title('Split Contours')
        # ax.set_aspect('equal')
        # ax.axis('off')
        # plt.show()

    if plot:
        extremes = [midline[0], midline[-1]]

        plot_transform = None
        if plot_transform is not None:
            split_contours = [plot_transform(split_contour) for split_contour in split_contours]
            contour = plot_transform(contour)
            extremes = [plot_transform(extreme[:, None]) for extreme in extremes]
            split_points = [plot_transform(split_point[:, None]) for split_point in split_points]
            # split_points_vlines_start = plot_transform(split_points_vlines_start)
            # split_points_vlines_end = plot_transform(split_points_vlines_end)

        import matplotlib.pyplot as plt

        if ax is None:
            SHOW = True
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.axis("equal")
        else:
            SHOW = False
        # pretty plot with areas filled in the polygon and overall area annotated
        colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(split_contours)))
        for color, split_contour in zip(colors, split_contours, strict=True):
            ax.fill(split_contour[0], split_contour[1], alpha=0.5, color=color)
            # ax.text(np.mean(split_contour[0]), np.mean(split_contour[1]), f'{area_out[i]:.2f}', 
            # olor='black', fontsize=12)
        # plot contour
        ax.plot(contour[0], contour[1], "-", linewidth=2, color="grey")
        # put text between split points
        # add endpoints to split_points
        split_points = split_points.tolist()
        split_points.insert(0, extremes[0])
        split_points.append(extremes[1])
        # ax.scatter(np.array(split_points)[:,0], np.array(split_points)[:,1], color='black', s=20)
        ax.plot(midline[:, 0], midline[:, 1], "k--", linewidth=2)

        # plot edge orthogonal to each split point
        for i in range(0, len(edge_ortho_vectors)):
            pt = split_points[i + 1]
            length = 0.4
            ax.plot(
                [pt[0] - edge_ortho_vectors[i][0] * length, pt[0] + edge_ortho_vectors[i][0] * length],
                [pt[1] - edge_ortho_vectors[i][1] * length, pt[1] + edge_ortho_vectors[i][1] * length],
                "k-",
                linewidth=2,
            )

        # convert area_weights into fraction of total line length
        # e.g. area_weights=[1/6, 1/2, 2/3, 3/4] to ['1/6', '2/3', ...]
        # cumulative difference
        area_weights_diff = [area_weights[0]]
        for i in range(1, len(area_weights)):
            area_weights_diff.append(area_weights[i] - area_weights[i - 1])
        area_weights_diff.append(1 - area_weights[-1])

        for i in range(len(split_points) - 1):
            # get_index of split_points[i] in midline
            sp1_midline_idx = np.argmin(np.linalg.norm(midline - split_points[i], axis=1))
            sp2_midline_idx = np.argmin(np.linalg.norm(midline - split_points[i + 1], axis=1))

            # get midpoint on midline
            midpoint_idx = (sp1_midline_idx + sp2_midline_idx) // 2
            midpoint = midline[midpoint_idx]

            # get vector perpendicular to line between split points
            vector = np.array(split_points[i + 1]) - np.array(split_points[i])
            vector = vector / np.linalg.norm(vector)
            vector = np.array([-vector[1], vector[0]])

            midpoint = midpoint - vector * 3
            # ax.text(midpoint[0]-5, midpoint[1]-5, f'{area_out[i]:.2f}', color='black', fontsize=12)
            # ax.text(midpoint[0], midpoint[1], f'{area_weights_txt[i]}', color='black', fontsize=12,
            # horizontalalignment='center', verticalalignment='center')

        # start point & end point
        ax.plot(extremes[0][0], extremes[0][1], marker="o", markersize=8, color="black")
        ax.plot(extremes[1][0], extremes[1][1], marker="o", markersize=8, color="black")

        # plot contour point 0
        # ax.scatter(contour[0,0], contour[1,0], color='red', s=120)
        ax.set_title("Split Contours")

        if SHOW:
            ax.axis("off")
            ax.invert_xaxis()
            ax.axis("equal")
            plt.show()

    return calc_subsegment_area(split_contours), split_contours


def hampel_subdivide_contour(contour, num_rays, plot=False, ax=None):
    # Find the extreme points in the x-direction
    min_x_index = np.argmin(contour[0])
    contour = np.roll(contour, -min_x_index, axis=1)

    # get minimal bounding box around contour
    min_bounding_rectangle = minimum_bounding_rectangle(contour)

    # get long edges of rectangle
    rectangle_duplicate_last = np.vstack((min_bounding_rectangle, min_bounding_rectangle[0]))
    long_edges = np.diff(rectangle_duplicate_last, axis=0)
    long_edges = np.linalg.norm(long_edges, axis=1)
    long_edges_idx = np.argpartition(long_edges, -2)[-2:]

    # select lower long edge
    min_val = np.inf
    min_idx = None
    for i in long_edges_idx:
        if rectangle_duplicate_last[i][1] < min_val:
            min_val = rectangle_duplicate_last[i][1]
            min_idx = i

        if rectangle_duplicate_last[i + 1][1] < min_val:
            min_val = rectangle_duplicate_last[i + 1][1]
            min_idx = i

    lowest_points = rectangle_duplicate_last[[min_idx, min_idx + 1]]

    # sort lowest points by x coordinate
    if lowest_points[0, 0] < lowest_points[1, 0]:
        lowest_points = lowest_points[::-1]

    # get midpoint of lower edge of rectangle
    midpoint_lower_edge = np.mean(lowest_points, axis=0)

    # get angle of lower edge of rectangle to x-axis
    angle_lower_edge = np.arctan2(
        lowest_points[1, 1] - lowest_points[0, 1], lowest_points[1, 0] - lowest_points[0, 0]
    )  # % (np.pi)

    # steps = np.pi / num_rays

    # print(np.degrees(angle_lower_edge))
    # get angles for equally spaced rays
    angles = np.linspace(-angle_lower_edge, -angle_lower_edge + np.pi, num_rays + 2, endpoint=True)  # + np.pi *3
    angles = angles[1:-1]

    # create ray vectors
    ray_vectors = np.vstack((np.cos(angles), np.sin(angles)))
    # make ray vectors unit length
    ray_vectors = ray_vectors / np.linalg.norm(ray_vectors, axis=0)

    # invert x of ray vectors
    ray_vectors[0] = -ray_vectors[0]

    # Subdivision logic
    split_contours = []
    for ray_vector in ray_vectors.T:
        intersections = []
        for i in range(contour.shape[1] - 1):
            segment_start = contour[:, i]
            segment_end = contour[:, i + 1]
            segment_vector = segment_end - segment_start

            # Check for intersection with the ray
            matrix = np.array([segment_vector, -ray_vector]).T
            if np.linalg.matrix_rank(matrix) < 2:
                continue  # Skip parallel lines

            # Solve for intersection
            t, s = np.linalg.solve(matrix, midpoint_lower_edge - segment_start)
            if 0 <= t <= 1:
                intersection_point = segment_start + t * segment_vector
                intersections.append((i, intersection_point))

        # Sort intersections by their position along the contour
        intersections.sort()

        # Create new contours by splitting at intersections
        if intersections:
            first_index, first_intersection = intersections[0]
            second_index, second_intersection = intersections[-1]

            start_to_cutoff = np.hstack(
                (
                    contour[:, :first_index],
                    first_intersection[:, None],
                    second_intersection[:, None],
                    contour[:, second_index + 1 :],
                )
            )

            # connect first and second half
            split_contours.append(start_to_cutoff)
        else:
            raise ValueError("No intersections found, this should not happen")

    split_contours.append(contour)
    split_contours = split_contours[::-1]

    # split_contours = split_contours[::-1]

    # Plotting logic
    if plot:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.axis("equal")
            SHOW = True
        else:
            SHOW = False
        min_bounding_rectangle_plot = np.vstack((min_bounding_rectangle, min_bounding_rectangle[0]))
        # ax.plot(contour[0], contour[1], 'b-', label='Original Contour')
        ax.plot(min_bounding_rectangle_plot[:, 0], min_bounding_rectangle_plot[:, 1], "k--")
        ax.plot(midpoint_lower_edge[0], midpoint_lower_edge[1], "ko", markersize=8)
        for ray_vector in ray_vectors.T:
            ray_length = 15
            ray_vector *= -ray_length
            ax.plot(
                [midpoint_lower_edge[0], midpoint_lower_edge[0] + ray_vector[0]],
                [midpoint_lower_edge[1], midpoint_lower_edge[1] + ray_vector[1]],
                "k--",
            )
        # pretty plot with areas files in the polygon and overall area annotated
        colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(split_contours)))
        for color, split_contour in zip(colors, split_contours, strict=True):
            ax.fill(split_contour[0], split_contour[1], alpha=0.5, color=color)
        ax.plot(contour[0], contour[1], "-", linewidth=2, color="grey")

        ax.set_title("Split Contours")
        ax.axis("off")
        if SHOW:
            ax.axis("equal")
            plt.show()

    return calc_subsegment_area(split_contours), split_contours


def subdivide_contour(
    contour: np.ndarray,
    area_weights: list[float],
    plot: bool = False,
    ax: plt.Axes | None = None,
    plot_transform: Callable | None = None,
    oriented: bool = False,
    hline_anchor: np.ndarray | None = None
):
    """Subdivide contour based on area weights using vertical lines.

    Divides the contour into segments by drawing vertical lines at positions
    determined by the area weights. The lines are drawn perpendicular to a
    reference line connecting the extreme points of the contour.

    Parameters
    ----------
    contour : np.ndarray
        Array of shape (2, N) containing contour points.
    area_weights : list[float]
        List of weights for area-based subdivision.
    plot : bool, optional
        Whether to plot the results, by default False.
    ax : matplotlib.axes.Axes, optional
        Axes for plotting, by default None.
    plot_transform : callable, optional
        Function to transform points before plotting, by default None.
    oriented : bool, optional
        If True, use fixed horizontal reference line, by default False.
    hline_anchor : np.ndarray, optional
        Point to anchor horizontal reference line, by default None.

    Returns
    -------
    areas : np.ndarray
        Array of areas for each subsegment.
    split_contours : list[np.ndarray]
        List of contour arrays for each subsegment.

    Notes
    -----
    The subdivision process:
    1. Finds extreme points in x-direction.
    2. Creates reference line between extremes.
    3. Calculates split points based on area weights.
    4. Divides contour using perpendicular lines at split points.
    
    """
    # Find the extreme points in the x-direction
    min_x_index = np.argmax(contour[0])
    contour = np.roll(contour, -min_x_index, axis=1)

    min_x_index = 0
    max_x_index = np.argmin(contour[0])

    if oriented:
        contour_x_sorted = np.sort(contour[0])
        min_x = contour_x_sorted[0]
        max_x = contour_x_sorted[-1]
        extremes = (np.array([max_x, 0]), np.array([min_x, 0]))

        if hline_anchor is not None:
            extremes = (np.array([max_x, hline_anchor[1]]), np.array([min_x, hline_anchor[1]]))

        # only keep x values of extremes and set y 5 mm below most inferior point of contour
        # if hline_anchor is None:
        #     most_inferior_point = np.min(contour[1])
        #     extremes = (np.array([extremes[0][0], most_inferior_point - 5]), 
        #                   np.array([extremes[1][0], most_inferior_point - 5]))
        # else:
        #     # get y difference between extremes and hline_anchor
        #     y_diff = extremes[1][1] - hline_anchor[1]
        #     extremes = (np.array([extremes[0][0], extremes[0][1] - y_diff]), 
        #                       np.array([extremes[1][0], extremes[1][1] - y_diff]))
    else:
        extremes = (contour[:, min_x_index].copy(), contour[:, max_x_index].copy())
        # Calculate the line between the extreme points
        start_point, end_point = extremes
        line_vector = end_point - start_point
        line_length = np.linalg.norm(line_vector)

        # Normalize the line vector
        line_unit_vector = line_vector / line_length

        # Calculate the perpendicular vector
        perp_vector = np.array([-line_unit_vector[1], line_unit_vector[0]])
        perp_vector = perp_vector / np.linalg.norm(perp_vector)

        if hline_anchor is None:
            most_inferior_point = np.min(contour[1])
            # move extreme 1 down 5 mm below inferior point and extreme 2 the 
            # same distance (so the angle stays the same)
            down_distance = (extremes[1][1] - most_inferior_point) * 1.3
            start_point = extremes[0] + down_distance * perp_vector
            end_point = extremes[1] + down_distance * perp_vector

        else:
            # get closest point on line to hline_anchor
            intersection = start_point + line_unit_vector * np.dot(hline_anchor - start_point, line_unit_vector)
            # get distance closest point on line to hline_anchor
            distance = np.linalg.norm(intersection - hline_anchor)

            # import matplotlib.pyplot as plt
            # plt.close()
            # fig, ax = plt.subplots(1,1,figsize=(8, 6))
            # ax.plot(contour[0], contour[1], 'b-', label='Original Contour')
            # ax.plot(hline_anchor[0], hline_anchor[1], 'ro', label='Hline Anchor')
            # ax.plot(intersection[0], intersection[1], 'go', label='Intersection')
            # ax.plot(start_point[0], start_point[1], 'bo', label='Start Point')
            # ax.plot(end_point[0], end_point[1], 'go', label='End Point')
            # ax.legend()
            # plt.show()

            # move start and end point the same distance
            start_point = extremes[0] + distance * perp_vector
            end_point = extremes[1] + distance * perp_vector

        extremes = (start_point, end_point)

    # Calculate the line between the extreme points
    start_point, end_point = extremes
    line_vector = end_point - start_point
    line_length = np.linalg.norm(line_vector)

    # Normalize the line vector
    line_unit_vector = line_vector / line_length

    # Calculate the perpendicular vector
    perp_vector = np.array([-line_unit_vector[1], line_unit_vector[0]])

    # Calculate split points based on area weights
    split_points = []
    for weight in area_weights:
        # current_weight = np.sum(area_weights[:i])
        split_distance = weight * line_length
        split_point = start_point + split_distance * line_unit_vector
        split_points.append(split_point)

    # Split the contour at the calculated split points
    split_contours = []
    split_contours.append(contour)
    for split_point in split_points:
        intersections = []
        for i in range(contour.shape[1] - 1):
            segment_start = contour[:, i]
            segment_end = contour[:, i + 1]
            segment_vector = segment_end - segment_start

            # Check for intersection with the perpendicular line
            matrix = np.array([segment_vector, -perp_vector]).T
            if np.linalg.matrix_rank(matrix) < 2:
                continue  # Skip parallel lines

            # Solve for intersection
            t, s = np.linalg.solve(matrix, split_point - segment_start)
            if 0 <= t <= 1:
                intersection_point = segment_start + t * segment_vector
                intersections.append((i, intersection_point))

        # Sort intersections by their position along the contour
        # intersections.sort()

        # get the two intersections that have the highest y coordinate
        intersections.sort(key=lambda x: x[1][1], reverse=True)

        # Create new contours by splitting at intersections
        if intersections:
            first_index, first_intersection = intersections[1]
            second_index, second_intersection = intersections[0]

            if first_index > second_index:
                first_index, second_index = second_index, first_index
                first_intersection, second_intersection = second_intersection, first_intersection

            first_index += 1
            # second_index += 1

            # start_to_cutoff = np.hstack((contour[:, :first_index], first_intersection[:, None], 
            # second_intersection[:, None], contour[:, second_index + 1:]))
            start_to_cutoff = np.hstack(
                (first_intersection[:, None], contour[:, first_index:second_index], second_intersection[:, None])
            )

            # import matplotlib.pyplot as plt
            # plt.close()
            # fig, ax = plt.subplots(1,1,figsize=(8, 6))
            # ax.plot(contour[0], contour[1], 'b-', label='Original Contour')
            # ax.plot(contour[0][0], contour[1][0], 'bo', label='First contour point')
            # ax.plot(first_intersection[0], first_intersection[1], 'ro', label='First Intersection')
            # ax.plot(second_intersection[0], second_intersection[1], 'go', label='Second Intersection')
            # # ax.plot(contour[:, :first_index][0], contour[:, :first_index][1]+0.5, 'r-', label='First Segment')
            # # ax.plot(contour[:, second_index+1:][0], contour[:, second_index+1:][1]+1, 'g-', label='Second Segment')
            # ax.plot(contour[:, first_index:second_index][0], 
            #         contour[:, first_index:second_index][1]+0.5, 'r-', label='Segment')
            # ax.plot(start_to_cutoff[0], start_to_cutoff[1], 'g-', label='Start to Cutoff')
            # ax.legend()
            # plt.show()

            # connect first and second half
            split_contours.append(start_to_cutoff)
        else:
            raise ValueError("No intersections found, this should not happen")

    # if plot:
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(contour[0], contour[1], 'b-', label='Original Contour')
    #     plt.plot(extremes[0][0], extremes[0][1], 'rx', markersize=8, label='Start Point')
    #     plt.plot(extremes[1][0], extremes[1][1], 'gx', markersize=8, label='End Point')
    #     for i, split_contour in enumerate(split_contours):
    #         plt.plot(split_contour[0], split_contour[1], label=f'Split {i+1}')
    #         plt.scatter(split_contour[0], split_contour[1], s=10)  # Plot vertices
    #     plt.title('Split Contours')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.legend()
    #     plt.axis('equal')
    #     plt.show()

    #     # same plot but segment are moved apart by 5 mm
    #     plt.figure(figsize=(8, 6))
    #     for i, split_contour in enumerate(split_contours):
    #         plt.plot(split_contour[0], split_contour[1]+i*5, label=f'Split {i+1}')
    #         plt.scatter(split_contour[0], split_contour[1]+i*5, s=10)  # Plot vertices
    #     plt.title('Split Contours')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.legend()
    #     plt.axis('equal')
    #     plt.show()

    if plot:
        # make vline at every split point
        split_points_vlines_start = (np.array(split_points) - perp_vector * 1).T
        split_points_vlines_end = (np.array(split_points) + perp_vector * 1).T

        if oriented:
            # make another vline at start point and end point, this time not 
            # perpendicular to line but perpendicular to x-axis
            start_point_vline = np.array([start_point, np.array([start_point[0], start_point[1] + 8])])
            end_point_vline = np.array([end_point, np.array([end_point[0], end_point[1] + 8])])
        else:
            start_point_vline = np.array([start_point, start_point - perp_vector * 8])
            end_point_vline = np.array([end_point, end_point - perp_vector * 8])

        if plot_transform is not None:
            split_contours = [plot_transform(split_contour) for split_contour in split_contours]
            contour = plot_transform(contour)
            extremes = [plot_transform(extreme[:, None]) for extreme in extremes]
            split_points = [plot_transform(split_point[:, None]) for split_point in split_points]
            split_points_vlines_start = plot_transform(split_points_vlines_start)
            split_points_vlines_end = plot_transform(split_points_vlines_end)
            start_point_vline = plot_transform(start_point_vline.T).T
            end_point_vline = plot_transform(end_point_vline.T).T

        import matplotlib.pyplot as plt

        if ax is None:
            SHOW = True
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.axis("equal")
        else:
            SHOW = False
        # pretty plot with areas filled in the polygon and overall area annotated
        colors = plt.cm.Spectral(np.linspace(0.2, 0.8, len(split_contours)))
        for color, split_contour in zip(colors, split_contours, strict=True):
            ax.fill(split_contour[0], split_contour[1], alpha=0.5, color=color)
            # ax.text(np.mean(split_contour[0]), np.mean(split_contour[1]), 
            # f'{area_out[i]:.2f}', color='black', fontsize=12)
        # plot contour
        ax.plot(contour[0], contour[1], "-", linewidth=2, color="grey")
        # dashed line between start point & end point
        ax.plot(
            np.vstack((extremes[0][0], extremes[1][0])),
            np.vstack((extremes[0][1], extremes[1][1])),
            "--",
            linewidth=2,
            color="grey",
        )
        # markers at every split point
        for i in range(split_points_vlines_start.shape[1]):
            ax.plot(
                np.vstack((split_points_vlines_start[:, i][0], split_points_vlines_end[:, i][0])),
                np.vstack((split_points_vlines_start[:, i][1], split_points_vlines_end[:, i][1])),
                "k-",
                linewidth=2,
            )

        ax.plot(start_point_vline[:, 0], start_point_vline[:, 1], "--", linewidth=2, color="grey")
        ax.plot(end_point_vline[:, 0], end_point_vline[:, 1], "--", linewidth=2, color="grey")
        # put text between split points
        # add endpoints to split_points
        split_points.insert(0, extremes[0])
        split_points.append(extremes[1])
        # convert area_weights into fraction of total line length
        # e.g. area_weights=[1/6, 1/2, 2/3, 3/4] to ['1/6', '2/3', ...]
        # cumulative difference
        area_weights_diff = []
        area_weights_diff.append(area_weights[0])
        for i in range(1, len(area_weights)):
            area_weights_diff.append(area_weights[i] - area_weights[i - 1])
        area_weights_diff.append(1 - area_weights[-1])

        # area_weights_txt = area_weights_txt / area_weights_txt[-1]
        from fractions import Fraction

        area_weights_txt = [
            Fraction(area_weights_diff[i]).limit_denominator(1000) for i in range(len(area_weights_diff))
        ]

        for i in range(len(split_points) - 1):
            midpoint = np.mean([split_points[i], split_points[i + 1]], axis=0)
            # ax.text(midpoint[0]-5, midpoint[1]-5, f'{area_out[i]:.2f}', color='black', fontsize=12)
            ax.text(
                midpoint[0],
                midpoint[1] - 5,
                f"{area_weights_txt[i]}",
                color="black",
                fontsize=11,
                horizontalalignment="center",
            )

        # start point & end point
        ax.plot(extremes[0][0], extremes[0][1], marker="o", markersize=8, color="black")
        ax.plot(extremes[1][0], extremes[1][1], marker="o", markersize=8, color="black")

        # plot contour 0 point
        # ax.scatter(contour[0,0], contour[1,0], color='red', s=100)

        ax.set_title("Split Contours")
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')

        # axis off
        ax.axis("off")
        if SHOW:
            ax.axis("equal")
            plt.show()

    return calc_subsegment_area(split_contours), split_contours


def transform_to_acpc_standard(contour_ras, ac_pt_ras, pc_pt_ras):
    """Transform contour coordinates to AC-PC standard space.

    Transforms the contour coordinates by:
        1. Translating AC point to origin.
        2. Rotating to align PC point with posterior direction.
        3. Scaling to maintain AC-PC distance.

    Parameters
    ----------
    contour_ras : np.ndarray
        Array of shape (2, N) or (3, N) containing contour points in RAS space.
    ac_pt_ras : np.ndarray
        Anterior commissure point coordinates in RAS space.
    pc_pt_ras : np.ndarray
        Posterior commissure point coordinates in RAS space.

    Returns
    -------
    contour_acpc : np.ndarray
        Transformed contour points in AC-PC space.
    ac_pt_acpc : np.ndarray
        AC point in AC-PC space (origin).
    pc_pt_acpc : np.ndarray
        PC point in AC-PC space.
    rotate_back : callable
        Function to transform points back to RAS space.
    
    """
    # translate AC to the origin and PC to (0, ac_pc_dist)
    translation_matrix = np.array([[1, 0, -ac_pt_ras[0]], [0, 1, -ac_pt_ras[1]], [0, 0, 1]])

    ac_pc_vec = pc_pt_ras - ac_pt_ras
    ac_pc_dist = np.linalg.norm(ac_pc_vec)

    posterior_vector = np.array([-ac_pc_dist, 0])

    # get angle between ac_pc_vec and posterior_vector
    dot_product = np.dot(ac_pc_vec, posterior_vector)
    norms_product = np.linalg.norm(ac_pc_vec) * np.linalg.norm(posterior_vector)
    theta = np.arccos(dot_product / norms_product)

    # Determine the sign of the angle using cross product
    cross_product = np.cross(ac_pc_vec, posterior_vector)
    if cross_product < 0:
        theta = -theta

    # create rotation matrix for theta
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    # apply translation and rotation
    if contour_ras.shape[0] == 2:
        contour_ras_homogeneous = np.vstack([contour_ras, np.ones(contour_ras.shape[1])])
    else:
        contour_ras_homogeneous = contour_ras

    contour_acpc = (rotation_matrix @ translation_matrix) @ contour_ras_homogeneous
    contour_acpc = contour_acpc[:2, :]

    def rotate_back(x):
        return (np.linalg.inv(rotation_matrix @ translation_matrix) @ np.vstack([x, np.ones(x.shape[1])]))[:2, :]

    return contour_acpc, np.array([0, 0]), np.array([-ac_pc_dist, 0]), rotate_back


def preprocess_cc(cc_label_nib, paths_csv, subj_id):
    """Preprocess corpus callosum mask and extract AC/PC coordinates.

    Parameters
    ----------
    cc_label_nib : nibabel.Nifti1Image
        NIfTI image containing corpus callosum segmentation.
    paths_csv : pd.DataFrame
        DataFrame containing AC and PC coordinates.
    subj_id : str
        Subject ID to look up in paths_csv.

    Returns
    -------
    cc_mask : np.ndarray
        Binary mask of corpus callosum.
    AC_2d : np.ndarray
        2D coordinates of anterior commissure.
    PC_2d : np.ndarray
        2D coordinates of posterior commissure.
    
    """
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

    return cc_mask, AC_2d, PC_2d


def get_primary_eigenvector(contour_ras):
    """Calculate primary eigenvector of contour points using PCA.

    Computes the principal direction of the contour by:
    1. Centering the points
    2. Computing covariance matrix
    3. Finding eigenvectors
    4. Selecting primary direction

    Parameters
    ----------
    contour_ras : np.ndarray
        Array of shape (2, N) containing contour points in RAS space.

    Returns
    -------
    pt0 : np.ndarray
        Starting point for eigenvector line.
    pt1 : np.ndarray
        End point for eigenvector line.
    
    """
    # Center the data by subtracting mean
    contour_mean = np.mean(contour_ras, axis=1, keepdims=True)
    contour_centered = contour_ras - contour_mean

    # Calculate covariance matrix
    cov_matrix = np.cov(contour_centered)

    # Get eigenvalues and eigenvectors using PCA
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # make first eigenvector unit length
    primary_eigenvector = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])
    pt0 = np.mean(contour_ras, axis=1)
    pt0 -= np.array([0, 5])
    pt1 = pt0 + primary_eigenvector * 100
    # plot mask with eigentvector
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1,2,figsize=(10, 8))
    # ax[0].imshow(cc_mask, cmap='gray')
    # # plot line between pt0 and pt1
    # ax[0].plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 'r-', linewidth=2)
    # plt.show()

    return pt0, pt1

