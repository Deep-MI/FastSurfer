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

import tempfile
from pathlib import Path

import lapy
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import scipy.interpolate
from plotly.io import write_html as plotly_write_html
from scipy.ndimage import gaussian_filter1d

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.data.constants import FSAVERAGE_MIDDLE
from CorpusCallosum.shape.contour import CCContour
from CorpusCallosum.shape.thickness import make_mesh_from_contour

try:
    from pyrr import Matrix44
    HAS_PYRR = True
except ImportError:
    HAS_PYRR = False
    class Matrix44(np.ndarray):
        pass

logger = logging.get_logger(__name__)



def _create_cap(
    points: np.ndarray,
    trias: np.ndarray,
    contour: CCContour,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a cap mesh for one end of the corpus callosum.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) containing mesh points
    trias : np.ndarray
        Array of shape (M, 3) containing triangle indices
    contour : CCContour
        CCContour object to create cap for

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - level_vertices : Array of vertices for the cap mesh
        - level_faces : Array of face indices for the cap mesh
        - level_colors : Array of thickness values for each vertex

    Notes
    -----
    The function:
    1. Creates level paths using _create_levelpaths
    2. Resamples level paths to fixed number of points
    3. Creates triangles between consecutive level paths
    4. Smooths thickness values for visualization
    """
    levelpaths, thickness_values = contour._create_levelpaths(points, trias)

    # Create mesh from level paths
    level_vertices = []
    level_faces = []
    level_colors = []
    vertex_counter = 0
    sorted_thickness_values = np.array(thickness_values)

    # smooth thickness values
    for _ in range(3):
        sorted_thickness_values = gaussian_filter1d(sorted_thickness_values, sigma=5)

    NUM_LEVELPOINTS = 50

    assert len(sorted_thickness_values) == len(levelpaths)

    # TODO: handle gap between first/last levelpath and contour
    for idx, levelpath1 in enumerate(levelpaths):
        levelpath1 = lapy.TriaMesh._TriaMesh__iterative_resample_polygon(levelpath1, NUM_LEVELPOINTS)
        level_vertices.append(levelpath1)
        level_colors.append(np.full((len(levelpath1)), sorted_thickness_values[idx]))
        if idx + 1 < len(levelpaths):
            levelpath2 = lapy.TriaMesh._TriaMesh__iterative_resample_polygon(levelpaths[idx + 1], NUM_LEVELPOINTS)

            # Create faces between the two paths by connecting vertices
            faces_between = []
            i, j = 0, 0

            while i < len(levelpath1) - 1 and j < len(levelpath2) - 1:
                faces_between.append([i, i + 1, len(levelpath1) + j])
                faces_between.append([i + 1, len(levelpath1) + j + 1, len(levelpath1) + j])

                i += 1
                j += 1

            while i < len(levelpath1) - 1:
                faces_between.append([i, i + 1, len(levelpath1) + j])
                i += 1

            while j < len(levelpath2) - 1:
                faces_between.append([i, len(levelpath1) + j + 1, len(levelpath1) + j])
                j += 1

            if faces_between:
                faces_between = np.array(faces_between)
                level_faces.append(faces_between + vertex_counter)

        vertex_counter += len(levelpath1)

    # Convert to numpy arrays
    level_vertices = np.vstack(level_vertices)
    level_faces = np.vstack(level_faces)
    level_colors = np.concatenate(level_colors)

    return level_vertices, level_faces, level_colors


def make_triangles_between_contours(contour1: np.ndarray, contour2: np.ndarray) -> np.ndarray:
    """Create a triangular mesh between two contours using a robust method.

    Parameters
    ----------
    contour1 : np.ndarray
        First contour points of shape (N, 2).
    contour2 : np.ndarray
        Second contour points of shape (M, 2).

    Returns
    -------
    np.ndarray
        Array of triangle indices of shape (K, 3) where K is the number of triangles.

    Notes
    -----
    The function:
    1. Finds closest point on contour2 to first point of contour1
    2. Creates triangles by connecting corresponding points
    3. Handles contours with different numbers of points
    4. Creates two triangles to form a quad between each pair of points
    """
    start_idx_c1 = 0
    # get closest point on contour2 to contour1[0]
    start_idx_c2 = np.argmin(np.linalg.norm(contour2 - contour1[0], axis=1))

    triangles = []
    n1 = len(contour1)
    n2 = len(contour2)

    for i in range(n1):
        # Current and next indices for contour1
        c1_curr = (start_idx_c1 + i) % n1
        c1_next = (start_idx_c1 + i + 1) % n1

        # Current and next indices for contour2, offset by n1 to account for vertex stacking
        c2_curr = ((start_idx_c2 + i) % n2) + n1
        c2_next = ((start_idx_c2 + i + 1) % n2) + n1

        # Create two triangles to form a quad between the contours
        triangles.append([c1_curr, c2_curr, c1_next])
        triangles.append([c2_curr, c2_next, c1_next])

    return np.array(triangles)



def create_CC_mesh_from_contours(contours: list[CCContour], 
                lr_center: float = 0, 
                closed: bool = False, 
                smooth: int = 0) -> None:
    """Create a surface mesh by triangulating between consecutive contours.

    Parameters
    ----------
    contours : list[CCContour]
        List of CCContour objects to create mesh from.
    lr_center : float, optional
        Center position in the left-right axis, by default 0.
    closed : bool, optional
        Whether to create a closed mesh by adding caps, by default False.
    smooth : int, optional
        Number of smoothing iterations to apply, by default 0.

    Raises
    ------
    Warning
        If no valid contours are found.

    Notes
    -----
    The function:
    1. Filters out None contours.
    2. Calculates z-coordinates for each slice.
    3. Creates triangles between adjacent contours.
    4. Optionally:
    - Creates caps at both ends.
    - Applies smoothing.
    - Colors caps based on thickness values.
    
    """

    # Check that all contours have the same resolution
    resolution = contours[0].resolution
    for idx, contour in enumerate(contours[1:], start=1):
        if not np.isclose(contour.resolution, resolution):
            raise ValueError(
                f"All contours must have the same resolution. "
                f"Expected {resolution}, but contour at index {idx} has {contour.resolution}."
            )


    # Calculate z coordinates for each slice
    z_coordinates = (np.arange(len(contours)) - len(contours) // 2) * contours[0].resolution + lr_center

    # Build vertices list with z-coordinates
    vertices = []
    faces = []
    vertex_start_indices = []  # Track starting index for each contour
    current_index = 0

    for i, contour in enumerate(contours):
        vertex_start_indices.append(current_index)
        vertices.append(np.hstack([contour.contour, np.full((len(contour.contour), 1), z_coordinates[i])]))

        # Check if there's a next valid contour to connect to
        if i + 1 < len(contours):
            contour2 = contours[i + 1]
            faces_between = make_triangles_between_contours(contour.contour, contour2.contour)
            faces.append(faces_between + current_index)

        current_index += len(contour.contour)

    vertex_values = np.concatenate([contour.thickness_values for contour in contours])

    

    if smooth > 0:
        tmp_mesh = CCMesh(vertices, faces, vertex_values=vertex_values)
        tmp_mesh.smooth_(smooth)
        vertices = tmp_mesh.v
        faces = tmp_mesh.t
        vertex_values = tmp_mesh.mesh_vertex_colors

    if closed:
        # Close the mesh by creating caps on both ends
        # Left cap (first slice) - use counterclockwise orientation
        left_side_points, left_side_trias = make_mesh_from_contour(vertices[: vertex_start_indices[1]][..., :2])
        left_side_points = np.hstack([left_side_points, np.full((len(left_side_points), 1), z_coordinates[0])])

        # Right cap (last slice) - reverse points for proper orientation
        right_side_points, right_side_trias = make_mesh_from_contour(vertices[vertex_start_indices[-1] :][..., :2])
        right_side_points = np.hstack([right_side_points, np.full((len(right_side_points), 1), z_coordinates[-1])])

        color_sides = True
        if color_sides:
            left_side_points, left_side_trias, left_side_colors = _create_cap(
                left_side_points, left_side_trias, 0
            )
            right_side_points, right_side_trias, right_side_colors = _create_cap(
                right_side_points, right_side_trias, len(contours) - 1
            )

            # reverse right side trias
            right_side_trias = right_side_trias[:, ::-1]

        left_side_trias = left_side_trias + current_index
        current_index += len(left_side_points)

        right_side_trias = right_side_trias + current_index
        current_index += len(right_side_points)

        vertices = [vertices, left_side_points, right_side_points]
        faces = [faces, left_side_trias, right_side_trias]
        vertex_values = [vertex_values, left_side_colors, right_side_colors]

    return CCMesh(vertices, faces, vertex_values=vertex_values, resolution=resolution)


class CCMesh(lapy.TriaMesh):
    """A class for representing and manipulating corpus callosum (CC) meshes.

    This class extends lapy.TriaMesh to provide specialized functionality for working with
    corpus callosum meshes, including contour management, thickness measurements, and
    visualization capabilities.

    The mesh can be constructed from a series of 2D contours representing slices of the
    corpus callosum, with optional thickness measurements at various points along these
    contours.

    Attributes
    ----------
    contours : list[np.ndarray]
        List of numpy arrays containing 2D contour points for each slice.
    thickness_values : list[np.ndarray]
        List of thickness measurements for each contour point.
    start_end_idx : list[tuple[int, int]]
        List of tuples containing start and end indices for each contour.
    ac_coords : np.ndarray
        Coordinates of the anterior commissure.
    pc_coords : np.ndarray
        Coordinates of the posterior commissure.
    resolution : float
        Spatial resolution of the mesh.
    v : np.ndarray
        Vertex coordinates of the mesh.
    t : np.ndarray
        Triangle indices of the mesh.
    original_thickness_vertices : list[np.ndarray]
        List of vertex indices where thickness was originally measured.
    """

    def __init__(self, 
                 vertices: list | np.ndarray, 
                 faces: list | np.ndarray, 
                 vertex_values: list | np.ndarray | None = None,
                 resolution: float = 1.0):
        """Initialize a CC_Mesh object.

        Parameters
        ----------
        vertices : list or numpy.ndarray
            List of vertex coordinates or array of shape (N, 3).
        faces : list or numpy.ndarray
            List of face indices or array of shape (M, 3).
        vertex_values : list or numpy.ndarray, optional
            Vertex values for each vertex (CC thickness values)
        """
        super().__init__(np.vstack(vertices), np.vstack(faces))
        self.mesh_vertex_colors = vertex_values
        self.resolution = resolution

    def plot_mesh(
        self,
        output_path: Path | str | None = None,
        colormap: str = "red_to_yellow",
        thickness_overlay: bool = True,
        show_grid: bool = False,
        color_range: tuple[float, float] | None = None,
        legend: str = "",
        threshold: tuple[float, float] | None = None,
    ):
        """Plot the mesh using Plotly for better performance and interactivity.

        Creates an interactive 3D visualization of the mesh with optional features like
        thickness overlay, contour display, and grid visualization.

        Parameters
        ----------
        output_path : Path, str, optional
            Path to save the plot. If None, displays the plot interactively.
        colormap : str, optional
            Which colormap to use, by default "red_to_yellow".
            Options:
            - "red_to_blue": Red -> Orange -> Grey -> Light Blue -> Blue
            - "red_to_yellow": Red -> Yellow -> Light Blue -> Blue
            - "yellow_to_red": Yellow -> Light Blue -> Blue -> Red
            - "blue_to_red": Blue -> Light Blue -> Grey -> Orange -> Red
        thickness_overlay : bool, optional
            Whether to overlay thickness values on the mesh, by default True.
        show_contours : bool, optional
            Whether to show the contours, by default False.
        show_grid : bool, optional
            Whether to show the grid, by default False.
        color_range : tuple[float, float], optional
            Fixed range (min, max) for the colorbar, by default None.
        show_mesh_edges : bool, optional
            Whether to show the mesh edges, by default False.
        legend : str, optional
            Legend text for the colorbar, by default "".
        threshold : tuple[float, float], optional
            Values between these thresholds will be shown in grey, by default None.

        Notes
        -----
        The plot can be saved to an HTML file or displayed in a web browser.
        """
        assert self.v is not None and self.t is not None, "Mesh has not been created yet"

        if len(self.v) == 0:
            logger.warning("Warning: No vertices in mesh to plot")
            return

        if len(self.t) == 0:
            logger.warning("Warning: No faces in mesh to plot")
            return

        # Define available colormaps
        colormaps = {
            "red_to_blue": [
                [0.0, "rgb(255,0,0)"],  # Bright red
                [0.25, "rgb(255,165,0)"],  # Light orange
                [0.5, "rgb(150,150,150)"],  # Dark grey for middle
                [0.75, "rgb(173,216,230)"],  # Light blue
                [1.0, "rgb(0,0,255)"],  # Bright blue
            ],
            "blue_to_red": [
                [0.0, "rgb(0,0,255)"],  # Bright blue
                [0.25, "rgb(173,216,230)"],  # Light blue
                [0.5, "rgb(150,150,150)"],  # Dark grey for middle
                [0.75, "rgb(255,165,0)"],  # Light orange
                [1.0, "rgb(255,0,0)"],  # Bright red
            ],
            "red_to_yellow": [
                [0.0, "rgb(255,0,0)"],  # Bright red
                [0.33, "rgb(255,85,0)"],  # Red-orange
                [0.66, "rgb(255,170,0)"],  # Orange
                [1.0, "rgb(255,255,0)"],  # Yellow
            ],
            "yellow_to_red": [
                [0.0, "rgb(255,255,0)"],  # Yellow
                [0.33, "rgb(255,170,0)"],  # Orange
                [0.66, "rgb(255,85,0)"],  # Red-orange
                [1.0, "rgb(255,0,0)"],  # Bright red
            ],
        }

        # Select the colormap
        if colormap not in colormaps:
            logger.warning(f"Warning: Unknown colormap '{colormap}'. Using 'red_to_blue' instead.")
            colormap = "red_to_blue"

        selected_colormap = colormaps[colormap]

        # If threshold is provided, modify the colormap to include grey region
        if threshold is not None and thickness_overlay and hasattr(self, "mesh_vertex_colors"):
            data_min = np.min(self.mesh_vertex_colors) if color_range is None else color_range[0]
            data_max = np.max(self.mesh_vertex_colors) if color_range is None else color_range[1]
            data_range = data_max - data_min

            # Calculate normalized threshold positions
            thresh_low = (threshold[0] - data_min) / data_range
            thresh_high = (threshold[1] - data_min) / data_range

            # Ensure thresholds are within [0,1]
            thresh_low = max(0, min(1, thresh_low))
            thresh_high = max(0, min(1, thresh_high))

            # Create new colormap with grey threshold region
            grey_color = "rgb(150,150,150)"  # Medium grey
            new_colormap = []

            # Add colors before threshold with adjusted positions
            if thresh_low > 0:
                for pos, color in selected_colormap:
                    if pos < 1:  # Only use positions less than 1
                        new_pos = pos * thresh_low
                        new_colormap.append([new_pos, color])

            # Add threshold boundaries with grey
            new_colormap.extend([[thresh_low, grey_color], [thresh_high, grey_color]])

            # Add colors after threshold with adjusted positions
            if thresh_high < 1:
                remaining_range = 1 - thresh_high
                for pos, color in selected_colormap:
                    if pos > 0:  # Only use positions greater than 0
                        new_pos = thresh_high + pos * remaining_range
                        if new_pos <= 1:  # Ensure we don't exceed 1
                            new_colormap.append([new_pos, color])

            selected_colormap = new_colormap

        # Calculate data ranges and center
        xyz_min = self.v.min(axis=0)
        xyz_max = self.v.max(axis=0)
        xyz_range = xyz_max - xyz_min
        max_range = xyz_range.max()
        center = (xyz_max + xyz_min) / 2

        # Create mesh plot
        fig = go.Figure()

        # Add the mesh as a surface
        mesh_args = {
            "x": self.v[:, 0],
            "y": self.v[:, 1],
            "z": self.v[:, 2],
            "i": self.t[:, 0],  # First vertex of each triangle
            "j": self.t[:, 1],  # Second vertex
            "k": self.t[:, 2],  # Third vertex
            "hoverinfo": "skip",
            "lighting": dict(ambient=0.9, diffuse=0.1, roughness=0.3),
        }

        if thickness_overlay and hasattr(self, "mesh_vertex_colors"):
            mesh_args.update(
                {
                    "intensity": self.mesh_vertex_colors,  # Add intensity values for colorbar
                    "showscale": True,
                    "colorbar": dict(
                        title=dict(
                            text=legend,
                            font=dict(size=35, color="white"),  # Increase title font size and make white
                            side="right",  # Place title on right side
                        ),
                        len=0.55,  # Make colorbar shorter
                        thickness=35,  # Make colorbar wider
                        tickfont=dict(size=30, color="white"),  # Increase tick font size and make white
                        tickformat=".1f",  # Show one decimal place
                    ),
                    "opacity": 1,
                    "colorscale": selected_colormap,
                }
            )

            # Set the colorbar range
            if color_range is not None:
                mesh_args["cmin"] = color_range[0]
                mesh_args["cmax"] = color_range[1]
            else:
                # Use data range if no explicit range is provided
                mesh_args["cmin"] = np.min(self.mesh_vertex_colors)
                mesh_args["cmax"] = np.max(self.mesh_vertex_colors)
        else:
            mesh_args["color"] = "lightsteelblue"

        fig.add_trace(go.Mesh3d(**mesh_args))

        # Calculate axis ranges to maintain equal aspect ratio
        ranges = []
        for i in range(3):
            axis_range = [center[i] - max_range / 2, center[i] + max_range / 2]
            ranges.append(axis_range)

        # Configure axes and grid visibility
        axis_config = dict(
            showgrid=show_grid,
            showline=show_grid,
            zeroline=show_grid,
            showbackground=show_grid,
            showticklabels=show_grid,
            gridcolor="white",
            tickfont=dict(color="white"),
            title=dict(font=dict(color="white")),
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(range=ranges[0], **{**axis_config, "title": "AP" if show_grid else ""}),
                yaxis=dict(range=ranges[1], **{**axis_config, "title": "SI" if show_grid else ""}),
                zaxis=dict(range=ranges[2], **{**axis_config, "title": "LR" if show_grid else ""}),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1), up=dict(x=0, y=0, z=1)),
                aspectmode="cube",  # Force equal aspect ratio
                aspectratio=dict(x=1, y=1, z=1),
                bgcolor="black",
                dragmode="orbit",  # Enable orbital rotation by default
            ),
            showlegend=False,
            margin=dict(l=0, r=100, t=0, b=0),  # Increased right margin for colorbar
            paper_bgcolor="black",
            plot_bgcolor="black",
        )

        if output_path is not None:
            self.__make_parent_folder(output_path)
            plotly_write_html(fig, output_path, include_plotlyjs="cdn")  # Save as interactive HTML
        else:
            # For non-interactive display, save to a temporary HTML and open in browser
            import tempfile
            import webbrowser

            temp_path = Path(tempfile.gettempdir()) / "cc_mesh_plot.html"
            plotly_write_html(fig, temp_path, include_plotlyjs="cdn")
            webbrowser.open(f"file://{temp_path}")




    def plot_contour(self, slice_idx: int, output_path: str) -> None:
        """Plot a single contour with thickness values.

        Parameters
        ----------
        slice_idx : int
            Index of the slice to plot.
        output_path : str
            Path where to save the plot.

        Raises
        ------
        ValueError
            If the contour for the specified slice is not set.

        Notes
        -----
        Creates a 2D visualization with:
        - Points colored by thickness values.
        - Gray points for missing thickness values.
        - Connected contour line.
        - Grid, labels, and legend.
        """
        self.__make_parent_folder(output_path)

        if self.contours[slice_idx] is None:
            raise ValueError(f"Contour for slice {slice_idx} is not set")

        contour = self.contours[slice_idx]

        plt.figure(figsize=(15, 10))
        # Get thickness values for this slice
        thickness = self.thickness_values[slice_idx]

        # Plot points with colors based on thickness
        for i in range(len(contour)):
            if np.isnan(thickness[i]):
                plt.plot(contour[i, 0], contour[i, 1], "o", color="gray", markersize=1)
            else:
                # Map thickness to color from red to yellow
                plt.plot(
                    contour[i, 0],
                    contour[i, 1],
                    "o",
                    color=plt.cm.YlOrRd(thickness[i] / np.nanmax(thickness)),
                    markersize=1,
                )

        # Connect points with lines
        plt.plot(contour[:, 0], contour[:, 1], "-", color="black", alpha=0.3, label="Contour")
        plt.axis("equal")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"CC contour for slice {slice_idx}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)



    def plot_cc_contour_with_levelsets(
        self,
        contour_idx: int = 0,
        #FIXME: levelpaths is not used
        levelpaths: list | None = None,
        title: str | None = None,
        save_path: str | None = None,
        colorbar: bool = True,
        mode: str = "p-value",
    ) -> matplotlib.figure.Figure:
        """Plot a contour with levelset visualization.

        Creates a visualization of a contour with interpolated levelsets, useful for
        analyzing the thickness distribution across the corpus callosum.

        Parameters
        ----------
        contour_idx : int, default=0
            Index of the contour to plot, by default 0.
        levelpaths : list, optional
            List of levelset paths. If None, uses stored levelpaths.
        title : str, optional
            Title for the plot.
        save_path : str, optional
            Path to save the plot. If None, displays interactively.
        colorbar : bool, default=True
            Whether to show the colorbar.
        mode : {"p-value", "icc"}, default="p-value"
            Mode of the plot.
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object.
        """

        plot_values = np.array(self.thickness_values[contour_idx][~np.isnan(self.thickness_values[contour_idx])])[::-1]
        points, trias = make_mesh_from_contour(self.contours[contour_idx], max_volume=0.5, min_angle=25, verbose=False)

        # make points 3D by adding zero
        points = np.column_stack([points, np.zeros(len(points))])

        levelpaths, _ = self._create_levelpaths(contour_idx, points, trias, num_points=len(plot_values)-2)

        outside_contour = self.contours[contour_idx].T

        # Create a grid of points covering the contour area with higher resolution
        x_min, x_max = np.min(outside_contour[0]), np.max(outside_contour[0])
        y_min, y_max = np.min(outside_contour[1]), np.max(outside_contour[1])
        margin = 1
        resolution = 0.05  # Higher resolution for smoother interpolation
        x_grid, y_grid = np.meshgrid(
            np.arange(x_min - margin, x_max + margin, resolution), np.arange(y_min - margin, y_max + margin, resolution)
        )

        # Create a path from the outside contour
        contour_path = matplotlib.path.Path(np.column_stack([outside_contour[0], outside_contour[1]]))

        # Check which points are inside the contour
        points = np.column_stack([x_grid.flatten(), y_grid.flatten()])
        mask = contour_path.contains_points(points).reshape(x_grid.shape)

        # Collect all levelpath points and their corresponding values
        # Extend each levelpath at both ends to improve extrapolation
        all_level_points_x = []
        all_level_points_y = []
        all_level_values = []

        for i, path in enumerate(levelpaths):
            if len(path) == 1:
                all_level_points_x.append(path[0][0])
                all_level_points_y.append(path[0][1])
                all_level_values.append(plot_values[i])
                continue

            # make levelpath
            path = lapy.TriaMesh._TriaMesh__resample_polygon(path, 1000)

            # Extend at the beginning: add point in direction opposite to first segment
            first_segment = path[1] - path[0]
            # standardize length of first segment
            first_segment = first_segment / np.linalg.norm(first_segment) * 10
            extension_start = path[0] - first_segment
            all_level_points_x.append(extension_start[0])
            all_level_points_y.append(extension_start[1])
            all_level_values.append(plot_values[i])

            # Add original path points
            for point in path:
                all_level_points_x.append(point[0])
                all_level_points_y.append(point[1])
                all_level_values.append(plot_values[i])

            # Extend at the end: add point in direction of last segment
            last_segment = path[-1] - path[-2]
            # standardize length of last segment
            last_segment = last_segment / np.linalg.norm(last_segment) * 10
            extension_end = path[-1] + last_segment
            all_level_points_x.append(extension_end[0])
            all_level_points_y.append(extension_end[1])
            all_level_values.append(plot_values[i])

        # Convert to numpy arrays
        all_level_points_x = np.array(all_level_points_x)
        all_level_points_y = np.array(all_level_points_y)
        all_level_values = np.array(all_level_values)

        # Use griddata to perform smooth interpolation - using 'linear' instead of 'cubic'
        # and properly formatting the input points
        grid_values = scipy.interpolate.griddata(
            (all_level_points_x, all_level_points_y), all_level_values, (x_grid, y_grid), method="linear", fill_value=0,
        )

        # smooth the grid_values
        grid_values = scipy.ndimage.gaussian_filter(grid_values, sigma=5, radius=5)

        # Apply the mask to only show values inside the contour
        masked_values = np.where(mask, grid_values, np.nan)

        if mode == "p-value":
            # Sample colormaps
            colors1 = plt.cm.binary([0.4] * 128)
            colors2 = plt.cm.hot(np.linspace(0.8, 0.1, 128))
        elif mode == "icc":
            colors1 = plt.cm.Blues(np.linspace(0, 1, 128))
            colors2 = plt.cm.binary([0.4] * 128)
        else:
            raise ValueError(f"Invalid mode '{mode}'")

        # Combine the color samples
        colors = np.vstack((colors2, colors1))

        # Create a new colormap
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_colormap", colors)

        # Plot CC contour with levelsets
        fig = plt.figure(figsize=(10, 3))
        # Apply a 10-degree rotation to the entire plot
        base = plt.gca().transData
        transform = matplotlib.transforms.Affine2D().rotate_deg(10)
        transform = transform + base

        # Plot the filled contour with interpolated colors
        plt.imshow(
            masked_values,
            extent=(x_min - margin, x_max + margin, y_min - margin, y_max + margin),
            origin="lower",
            cmap=cmap,
            alpha=1,
            interpolation="bilinear",
            vmin=0,
            vmax=0.10 if mode == "p-value" else 1,
            transform=transform,
        )

        plt.imshow(
            masked_values,
            extent=(x_min - margin, x_max + margin, y_min - margin, y_max + margin),
            origin="lower",
            cmap=cmap,
            alpha=1,
            interpolation="bilinear",
            vmin=0,
            vmax=0.10 if mode == "p-value" else 1,
            # norm=LogNorm(vmin=1e-3, vmax=0.1),  # Set minimum to avoid log(0)
            transform=transform,
        )

        if colorbar:
            # Add a colorbar
            cbar = plt.colorbar(aspect=10)
            if mode == "p-value":
                cbar.ax.set_ylim(0.001, 0.054)
                cbar.ax.set_yticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
                cbar.set_label("p-value (log scale)")
            elif mode == "icc":
                cbar.ax.set_ylim(0, 1)
                cbar.ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                cbar.ax.set_label("Intraclass correlation coefficient")

        # Plot the outside contour on top for clear boundary
        plt.plot(outside_contour[0], outside_contour[1], "k-", linewidth=2, label="CC Contour", transform=transform)

        plt.axis("equal")
        plt.title(title, fontsize=14, fontweight="bold")
        # plt.legend(loc='best')
        plt.gca().invert_xaxis()
        plt.axis("off")
        if save_path is not None:
            self.__make_parent_folder(save_path)
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
        return fig

    

    @staticmethod
    def __create_cc_viewmat() -> "Matrix44":
        """Create the view matrix for a nice view of the corpus callosum.

        Returns
        -------
        Matrix44
            4x4 view matrix that provides a standard view of the corpus callosum (from pyrr).

        Notes
        -----
        The function:
        1. Creates a base view matrix looking from the left with top up
        2. Applies a series of rotations:
            - -10 degrees around x-axis
            - 35 degrees around y-axis
            - -8 degrees around z-axis
        3. Adds a small translation for better centering
        """

        if not HAS_PYRR:
            raise ImportError("Pyrr not installed, install pyrr with `pip install pyrr`.")

        viewLeft = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  # left w top up // right
        transl = Matrix44.from_translation((0, 0, 0.4))
        viewmat = transl * viewLeft

        # rotate 10 degrees around x axis
        rot = Matrix44.from_x_rotation(np.deg2rad(-10))
        viewmat = viewmat * rot

        # rotate 35 degrees around y axis
        rot = Matrix44.from_y_rotation(np.deg2rad(35))
        viewmat = viewmat * rot

        # rotate 10 degrees around z axis
        rot = Matrix44.from_z_rotation(np.deg2rad(-8))
        viewmat = viewmat * rot

        return viewmat

    def snap_cc_picture(
        self,
        output_path: Path | str,
        fssurf_file: Path | str | None = None,
        overlay_file: Path | str | None = None
    ) -> None:
        """Snap a picture of the corpus callosum mesh.

        Parameters
        ----------
        output_path : Path, str
            Path where to save the snapshot image.
        fssurf_file : Path, str, optional
            Path to a FreeSurfer surface file to use for the snapshot.
            If None, the mesh is saved to a temporary file.
        overlay_file : Path, str, optional
            Path to a FreeSurfer overlay file to use for the snapshot.
            If None, the mesh is saved to a temporary file.

        Raises
        ------
        Warning
            If the mesh has no faces and cannot create a snapshot.

        Notes
        -----
        The function:
        1. Creates temporary files for mesh and overlay data if needed.
        2. Uses whippersnappy to create a snapshot with:
        - Custom view matrix for standard orientation.
        - Ambient lighting and colorbar settings.
        - Thickness overlay if available.
        3. Cleans up temporary files after use.
        """
        try:
            from whippersnappy.core import snap1
        except ImportError:
            # whippersnappy not installed
            raise RuntimeError(
                "The snap_cc_picture method of CCMesh requires whippersnappy, but whippersnappy was not found. "
                "Please install whippersnappy!"
            ) from None
        self.__make_parent_folder(output_path)
        # Skip snapshot if there are no faces
        if len(self.t) == 0:
            logger.warning("Cannot create snapshot - no faces in mesh")
            return

        # create temp file
        if fssurf_file:
            fssurf_file = Path(fssurf_file)
        else:
            fssurf_file = tempfile.NamedTemporaryFile(suffix=".fssurf", delete=True)
            self.write_fssurf(fssurf_file.name)

        if overlay_file:
            overlay_path: str | None = Path(overlay_file).name
        elif hasattr(self, "mesh_vertex_colors"):
            overlay_file = tempfile.NamedTemporaryFile(suffix=".w", delete=True)
            # Write thickness values in FreeSurfer .w format
            nib.freesurfer.write_morph_data(overlay_file.name, self.mesh_vertex_colors)
            overlay_path = overlay_file.name
        else:
            overlay_path = None

        snap1(
            fssurf_file.name,
            overlaypath=overlay_path,
            view=None,
            viewmat=self.__create_cc_viewmat(),
            width=3 * 500,
            height=3 * 300,
            outpath=output_path,
            ambient=0.6,
            colorbar_scale=0.5,
            colorbar_y=0.88,
            colorbar_x=0.19,
            brain_scale=2.1,
            fthresh=0,
            caption="Corpus Callosum thickness (mm)",
            caption_y=0.85,
            caption_x=0.17,
            caption_scale=0.5,
        )

        if fssurf_file and hasattr(fssurf_file, "close"):
            fssurf_file.close()
        if overlay_file and hasattr(overlay_file, "close"):
            overlay_file.close()

    def smooth_(self, iterations: int = 1) -> None:
        """Smooth the mesh while preserving the z-coordinates.

        Parameters
        ----------
        iterations : int, optional
            Number of smoothing iterations, by default 1.

        Notes
        -----
        The function:
        1. Stores original z-coordinates.
        2. Applies Laplacian smoothing to x and y coordinates.
        3. Restores original z-coordinates to maintain slice structure.
        """
        z_values = self.v[:, 2]
        super().smooth_(iterations)
        self.v[:, 2] = z_values


    @staticmethod
    def __make_parent_folder(filename: Path | str) -> None:
        """Create the parent folder for a file if it doesn't exist.

        Parameters
        ----------
        filename : Path, str
            Path to the file whose parent folder should be created.

        Notes
        -----
        Creates parent directory with parents=False to avoid creating
        multiple levels of directories unintentionally.
        """
        Path(filename).parent.mkdir(parents=False, exist_ok=True)

    def to_fs_coordinates(
        self,
        vox2ras_tkr: np.ndarray,
    ) -> None:
        """Convert mesh coordinates to FreeSurfer coordinate system.

        Parameters
        ----------
        vox2ras_tkr : np.ndarray
            4x4 voxel to RAS tkr-space transformation matrix.

        Notes
        -----
        Mesh coordinates seem to be in ASR (Anterior-Superior-Right) orientation, with the coordinate system origin on
        *the* midslice.
        The function performs the following:
        1. Convert from mesh coordinates (LSA and voxel coordinates) to fsaverage voxel coordinates (LIA, origin).
        a. Convert coordinates from ASR to LSA orientation.
        b. Convert to voxel coordinates using voxel size.
        c. Center LR coordinates and flips SI coordinates.
        2. Apply vox2ras_tkr transformation to get final coordinates.
        """

        # to voxel coordinates
        v_vox = self.v.copy()
        
        # to LSA
        v_vox = v_vox[:, [2, 1, 0]]
        # to voxel
        # FIXME: why are the vertex positions multiplied by voxel size here?
        #        removed => for center LR, now dividing by resolution => convert fsaverage middle from mm to vox
        #                => remove the conversion back to mm in the end
        #        all other operations are independent of order of operations (distributive)
        # v_vox /= vox_size[0]
        # center LR
        v_vox[:, 0] += FSAVERAGE_MIDDLE / self.resolution
        # flip SI
        v_vox[:, 1] = -v_vox[:, 1]

        #v_vox_test = np.round(v_vox).astype(int)
        ## write volume for debugging
        # contour_img = np.zeros(orig.shape)
        # for i in range(v_vox_test.shape[0]):
        #     contour_img[v_vox_test[i, 0], v_vox_test[i, 1], v_vox_test[i, 2]] = 1

        # tkrRAS = Torig*[C R S 1]'
        # Torig: mri_info --vox2ras-tkr orig.mgz 
        # https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
        self.v = (vox2ras_tkr @ np.concatenate([v_vox, np.ones((self.v.shape[0], 1))], axis=1).T).T[:, :3]
        # FIXME: why are the vertex positions multiplied by voxel size here?
        # self.v = self.v * vox_size[0]

    def write_fssurf(self, filename: Path | str) -> None:
        """Write the mesh to a FreeSurfer surface file.

        Parameters
        ----------
        filename : Path, str
            Path where to save the FreeSurfer surface file.

        Notes
        -----
        Creates parent directory if needed before writing the file.
        """
        self.__make_parent_folder(filename)
        return super().write_fssurf(filename)

    def write_morph_data(self, filename: Path | str) -> None:
        """Write the thickness values as a FreeSurfer overlay file.

        Parameters
        ----------
        filename : Path, str
            Path where to save the overlay file.

        Notes
        -----
        Creates parent directory if needed before writing the file.
        """
        self.__make_parent_folder(filename)
        return nib.freesurfer.write_morph_data(filename, self.mesh_vertex_colors)
