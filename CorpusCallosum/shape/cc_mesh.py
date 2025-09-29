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
import pyrr
import scipy.interpolate
from scipy.ndimage import gaussian_filter1d
from whippersnappy.core import snap1

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.shape.cc_endpoint_heuristic import smooth_contour
from CorpusCallosum.shape.cc_thickness import HiddenPrints, make_mesh_from_contour

logger = logging.get_logger(__name__)


class CC_Mesh(lapy.TriaMesh):
    """A class for representing and manipulating corpus callosum (CC) meshes.

    This class extends lapy.TriaMesh to provide specialized functionality for working with
    corpus callosum meshes, including contour management, thickness measurements, and
    visualization capabilities.

    The mesh can be constructed from a series of 2D contours representing slices of the
    corpus callosum, with optional thickness measurements at various points along these
    contours.

    Attributes:
        contours (list): List of numpy arrays containing 2D contour points for each slice.
        thickness_values (list): List of thickness measurements for each contour point.
        start_end_idx (list): List of tuples containing start and end indices for each contour.
        ac_coords (numpy.ndarray): Coordinates of the anterior commissure.
        pc_coords (numpy.ndarray): Coordinates of the posterior commissure.
        resolution (float): Spatial resolution of the mesh.
        v (numpy.ndarray): Vertex coordinates of the mesh.
        t (numpy.ndarray): Triangle indices of the mesh.
        original_thickness_vertices (list): List of vertex indices where thickness was originally measured.
    """

    def __init__(self, num_slices):
        """Initialize a CC_Mesh object.

        Args:
            num_slices (int): Number of slices in the corpus callosum mesh.
        """
        self.contours = [None] * num_slices
        self.thickness_values = [None] * num_slices
        self.start_end_idx = [None] * num_slices
        self.ac_coords = None
        self.pc_coords = None
        self.resolution = None
        self.v = None
        self.t = None
        self.original_thickness_vertices = [None] * num_slices

    def add_contour(
        self,
        slice_idx: int,
        contour: np.ndarray,
        thickness_values: np.ndarray,
        start_end_idx: tuple[int, int] | None = None,
    ):
        """Add a contour and its associated thickness values for a specific slice.

        Args:
            slice_idx (int): 
                Index of the slice where the contour should be added.
            contour (numpy.ndarray): 
                Array of shape (N, 2) containing 2D contour points.
            thickness_values (numpy.ndarray): 
                Array of thickness measurements for each contour point.
            start_end_idx (tuple[int, int], optional): 
                Tuple containing start and end indices for the contour.
                If None, defaults to (0, len(contour)//2).
        """
        self.contours[slice_idx] = contour
        self.thickness_values[slice_idx] = thickness_values
        # write vertex indices where thickness values are not nan
        self.original_thickness_vertices[slice_idx] = np.where(~np.isnan(thickness_values))[0]

        if start_end_idx is None:
            self.start_end_idx[slice_idx] = (0, len(contour) // 2)
        else:
            self.start_end_idx[slice_idx] = start_end_idx

    def set_acpc_coords(self, ac_coords: np.ndarray, pc_coords: np.ndarray):
        """Set the coordinates of the anterior and posterior commissure.

        Args:
            ac_coords (numpy.ndarray): 3D coordinates of the anterior commissure.
            pc_coords (numpy.ndarray): 3D coordinates of the posterior commissure.
        """
        self.ac_coords = ac_coords
        self.pc_coords = pc_coords

    def set_resolution(self, resolution: float):
        """Set the spatial resolution of the mesh.

        Args:
            resolution (float): Spatial resolution in millimeters.
        """
        self.resolution = resolution

    def plot_mesh(
        self,
        output_path: str | None = None,
        colormap: str = "red_to_yellow",
        thickness_overlay: bool = True,
        show_contours: bool = False,
        show_grid: bool = False,
        color_range: tuple[float, float] | None = None,
        show_mesh_edges: bool = False,
        legend: str = "",
        threshold: tuple[float, float] | None = None,
    ):
        """Plot the mesh using Plotly for better performance and interactivity.

        Creates an interactive 3D visualization of the mesh with optional features like
        thickness overlay, contour display, and grid visualization. The plot can be saved
        to an HTML file or displayed in a web browser.

        Args:
            output_path (str, optional): 
                Path to save the plot. If None, displays the plot interactively.
            colormap (str, optional): 
                Which colormap to use. Options are:
                - "red_to_blue": Red -> Orange -> Grey -> Light Blue -> Blue
                - "red_to_yellow": Red -> Yellow -> Light Blue -> Blue
                - "yellow_to_red": Yellow -> Light Blue -> Blue -> Red
                - "blue_to_red": Blue -> Light Blue -> Grey -> Orange -> Red
            
                Defaults to "red_to_yellow".
            thickness_overlay (bool, optional): 
                Whether to overlay thickness values on the mesh.
                Defaults to True.
            show_contours (bool, optional): 
                Whether to show the contours. Defaults to False.
            show_grid (bool, optional): 
                Whether to show the grid. Defaults to False.
            color_range (tuple[float, float], optional): 
                Optional tuple of (min, max) to set fixed
                color range. Defaults to None.
            show_mesh_edges (bool, optional): 
                Whether to show the mesh edges. Defaults to False.
            legend (str, optional): 
                Legend text for the colorbar. Defaults to "".
            threshold (tuple[float, float], optional): 
                Values between these thresholds will be shown in grey.
                Defaults to (-0.2, 0.2).
        """
        assert self.v is not None and self.t is not None, "Mesh has not been created yet"

        if len(self.v) == 0:
            print("Warning: No vertices in mesh to plot")
            return

        if len(self.t) == 0:
            print("Warning: No faces in mesh to plot")
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
            print(f"Warning: Unknown colormap '{colormap}'. Using 'red_to_blue' instead.")
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

        if show_contours:
            # Add contour polylines for reference
            num_slices = len(self.contours)

            # Calculate z coordinates for each slice - use same calculation as in create_mesh
            lr_center = self.v[len(self.v) // 2][2]
            z_coordinates = np.arange(num_slices) * self.resolution - (num_slices // 2) * self.resolution + lr_center

            for i in range(num_slices):
                if self.contours[i] is not None:
                    # Use slice position for z coordinate
                    z_coord = z_coordinates[i]
                    contour = self.contours[i]

                    # Create 3D points with fixed z coordinate
                    v_i = np.hstack([contour, np.full((len(contour), 1), z_coord)])

                    # Close the contour by adding the first point at the end
                    v_i = np.vstack([v_i, v_i[0]])

                    fig.add_trace(
                        go.Scatter3d(
                            x=v_i[:, 0],
                            y=v_i[:, 1],
                            z=v_i[:, 2],
                            mode="lines",
                            line=dict(color="white", width=2),
                            opacity=0.5,
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
        if show_mesh_edges:  # show the mesh edges
            edge_color = "darkgray"
            vertices_in_first_contour = len(self.contours[0])

            vertices_to_plot_first = np.concatenate([self.v[:vertices_in_first_contour], self.v[None, 0]])
            # Add mesh edges for first 900 vertices as one continuous line
            fig.add_trace(
                go.Scatter3d(
                    x=vertices_to_plot_first[:, 0],
                    y=vertices_to_plot_first[:, 1],
                    z=vertices_to_plot_first[:, 2],
                    mode="lines",
                    line=dict(color=edge_color, width=8),
                    opacity=1,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

            vertices_in_last_contour = len(self.contours[-1])

            vertices_before_last_contour = np.sum([len(c) for c in self.contours[:-1]])
            vertices_to_plot_last = np.concatenate(
                [
                    self.v[vertices_before_last_contour : vertices_before_last_contour + vertices_in_last_contour],
                    self.v[None, vertices_before_last_contour],
                ]
            )
            fig.add_trace(
                go.Scatter3d(
                    x=vertices_to_plot_last[:, 0],
                    y=vertices_to_plot_last[:, 1],
                    z=vertices_to_plot_last[:, 2],
                    mode="lines",
                    line=dict(color=edge_color, width=8),
                    opacity=1,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

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
            fig.write_html(output_path)  # Save as interactive HTML
        else:
            # For non-interactive display, save to a temporary HTML and open in browser
            import os
            import tempfile
            import webbrowser

            temp_path = os.path.join(tempfile.gettempdir(), "cc_mesh_plot.html")
            fig.write_html(temp_path)
            webbrowser.open("file://" + temp_path)

    def get_contour_edge_lengths(self, contour_idx):
        """Get the lengths of the edges of a contour.

        Args:
            contour_idx (int): Index of the contour to get the edge lengths for.

        Returns:
            numpy.ndarray: Array of edge lengths for the contour.
        """
        edges = np.diff(self.contours[contour_idx], axis=0)
        return np.sqrt(np.sum(edges**2, axis=1))

    @staticmethod
    def make_triangles_between_contours(contour1, contour2):
        """Creates a triangular mesh between two contours using a robust method.

        This method creates triangles that connect two contours by matching points between them.
        It starts from the closest point on contour2 to the first point of contour1 and creates
        triangles by connecting corresponding points.

        Args:
            contour1 (numpy.ndarray): First contour points of shape (N, 2).
            contour2 (numpy.ndarray): Second contour points of shape (M, 2).

        Returns:
            numpy.ndarray: Array of triangle indices of shape (K, 3) where K is the number of triangles.
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

    def _create_levelpaths(self, contour_idx, points, trias, num_points=None):
        # # compute poisson
        with HiddenPrints():
            cc_tria = lapy.TriaMesh(points, trias)
        # extract boundary curve
        bdr = np.array(cc_tria.boundary_loops()[0])

        # find index of endpoints in bdr list
        iidx1 = np.where(bdr == self.start_end_idx[contour_idx][0])[0][0]
        iidx2 = np.where(bdr == self.start_end_idx[contour_idx][1])[0][0]

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
            fem = lapy.Solver(cc_tria)
            vfunc = fem.poisson(0, (bdr, dcond))
            if num_points is not None:
                # TODO: do midline stuff
                level = 0
                midline_equidistant, midline_length = cc_tria.level_path(vfunc, level, n_points=num_points + 2)
                midline_equidistant = midline_equidistant[:, :2]
                eval_points = midline_equidistant
            else:
                eval_points = self.contours[contour_idx]
            gf = lapy.diffgeo.compute_rotated_f(cc_tria, vfunc)

        # interpolate midline to get levels to evaluate
        gf_interp = scipy.interpolate.griddata(cc_tria.v[:, 0:2], gf, eval_points, method="nearest")

        # sort by value
        sorting_idx_gf = np.argsort(gf_interp)
        gf_interp = gf_interp[sorting_idx_gf]
        sorted_thickness_values = self.thickness_values[contour_idx][sorting_idx_gf]

        # get levels to evaluate
        # level_length = tria.level_length(gf, gf_interp)

        levelpaths = []
        thickness_values = []

        for i in range(0, len(eval_points)):
            level = gf_interp[i]
            # levelpath starts at index zero
            if level == 0:
                continue
            lvlpath, lvlpath_length, tria_idx = cc_tria.level_path(gf, level, get_tria_idx=True)

            levelpaths.append(lvlpath)
            thickness_values.append(sorted_thickness_values[i])

        return levelpaths, thickness_values

    def _create_cap(self, points, trias, contour_idx):
        levelpaths, thickness_values = self._create_levelpaths(contour_idx, points, trias)

        # Create mesh from level paths
        level_vertices = []
        level_faces = []
        level_colors = []
        vertex_counter = 0
        sorted_thickness_values = np.array(thickness_values)

        # smooth thickness values
        from scipy.ndimage import gaussian_filter1d

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

    def create_mesh(self, lr_center: float = 0, closed: bool = False, smooth: int = 0):
        """Creates a surface mesh by triangulating between consecutive contours.

        This method constructs a 3D mesh from the stored contours by creating triangles between
        adjacent slices. It can optionally create a closed mesh by adding caps at the ends and
        apply smoothing.

        Args:
            lr_center (float, optional): Center position in the left-right axis. Defaults to 0.
            closed (bool, optional): Whether to create a closed mesh by adding caps. Defaults to False.
            smooth (int, optional): Number of smoothing iterations to apply. Defaults to 0.
        """
        # Filter out None contours and get their indices
        valid_contours = [(i, c) for i, c in enumerate(self.contours) if c is not None]
        if not valid_contours:
            print("Warning: No valid contours found")
            self.v = np.array([])
            self.t = np.array([])
            return

        # Calculate z coordinates for each slice
        z_coordinates = (
            np.arange(len(valid_contours)) * self.resolution - (len(valid_contours) // 2) * self.resolution + lr_center
        )

        # Build vertices list with z-coordinates
        vertices = []
        faces = []
        vertex_start_indices = []  # Track starting index for each contour
        current_index = 0

        for i, (_, contour) in enumerate(valid_contours):
            vertex_start_indices.append(current_index)
            vertices.append(np.hstack([contour, np.full((len(contour), 1), z_coordinates[i])]))

            # Check if there's a next valid contour to connect to
            if i + 1 < len(valid_contours):
                next_idx, contour2 = valid_contours[i + 1]
                faces_between = self.make_triangles_between_contours(contour, contour2)
                faces.append(faces_between + current_index)

            current_index += len(contour)

        self.set_mesh(vertices, faces, self.thickness_values)

        if smooth > 0:
            self.smooth_(smooth)

        if closed:
            # Close the mesh by creating caps on both ends
            # Left cap (first slice) - use counterclockwise orientation
            left_side_points, left_side_trias = make_mesh_from_contour(self.v[: vertex_start_indices[1]][..., :2])
            left_side_points = np.hstack([left_side_points, np.full((len(left_side_points), 1), z_coordinates[0])])

            # Right cap (last slice) - reverse points for proper orientation
            right_side_points, right_side_trias = make_mesh_from_contour(self.v[vertex_start_indices[-1] :][..., :2])
            right_side_points = np.hstack([right_side_points, np.full((len(right_side_points), 1), z_coordinates[-1])])

            color_sides = True
            if color_sides:
                left_side_points, left_side_trias, left_side_colors = self._create_cap(
                    left_side_points, left_side_trias, 0
                )
                right_side_points, right_side_trias, right_side_colors = self._create_cap(
                    right_side_points, right_side_trias, len(self.contours) - 1
                )

                # reverse right side trias
                right_side_trias = right_side_trias[:, ::-1]

            left_side_trias = left_side_trias + current_index
            current_index += len(left_side_points)

            right_side_trias = right_side_trias + current_index
            current_index += len(right_side_points)

            self.set_mesh(
                [self.v, left_side_points, right_side_points],
                [self.t, left_side_trias, right_side_trias],
                [self.mesh_vertex_colors, left_side_colors, right_side_colors],
            )

    def fill_thickness_values(self):
        """
        Interpolate missing thickness values on the contours by weighted average of nearest known thickness values.
        """

        # For each contour with missing thickness values
        for i in range(len(self.contours)):
            if self.contours[i] is None or self.thickness_values[i] is None:
                continue

            thickness = self.thickness_values[i]
            edge_lengths = self.get_contour_edge_lengths(i)

            # Find indices of points with known thickness
            known_idx = np.where(~np.isnan(thickness))[0]

            # For each point with unknown thickness
            for j in range(len(thickness)):
                if not np.isnan(thickness[j]):
                    continue

                # Find two closest points with known thickness
                distances = np.zeros(len(known_idx))
                for k, idx in enumerate(known_idx):
                    # Calculate distance along contour by summing edge lengths
                    if idx > j:
                        distances[k] = np.sum(edge_lengths[j:idx])
                    else:
                        distances[k] = np.sum(edge_lengths[idx:j])

                # Get indices of two closest points
                closest_indices = known_idx[np.argsort(distances)[:2]]
                closest_distances = np.sort(distances)[:2]

                # Calculate weights based on inverse distance
                weights = 1.0 / closest_distances
                weights = weights / np.sum(weights)

                # Calculate weighted average thickness
                thickness[j] = np.sum(weights * thickness[closest_indices])

            self.thickness_values[i] = thickness

    def smooth_thickness_values(self, iterations: int = 1):
        """
        Smooth the thickness values using a Gaussian filter
        """

        for i in range(len(self.thickness_values)):
            if self.thickness_values[i] is not None:
                self.thickness_values[i] = gaussian_filter1d(self.thickness_values[i], sigma=5)

    def plot_contour(self, slice_idx: int, output_path: str):
        """Plot a single contour with thickness values.

        Creates a 2D visualization of a specific contour slice with points colored according
        to their thickness values. The plot is saved to the specified output path.

        Args:
            slice_idx (int): Index of the slice to plot.
            output_path (str): Path where to save the plot.

        Raises:
            ValueError: If the contour for the specified slice is not set.
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

    def smooth_contour(self, contour_idx, window_size=5):
        """
        Smooth a contour using a moving average filter.

        Parameters
        ----------
        contour : tuple of arrays
            The contour coordinates (x, y).
        window_size : int
            Size of the smoothing window.

        Returns
        -------
        tuple of arrays
            The smoothed contour coordinates (x, y).
        """
        x, y = self.contours[contour_idx].T

        x, y = smooth_contour(x, y, window_size)

        self.contours[contour_idx] = np.array([x, y]).T

    def plot_cc_contour_with_levelsets(self, contour_idx=0, levelpaths=None, title=None, save_path=None, colorbar=True):
        """Plot a contour with levelset visualization.

        Creates a visualization of a contour with interpolated levelsets, useful for
        analyzing the thickness distribution across the corpus callosum.

        Args:
            contour_idx (int, optional): Index of the contour to plot. Defaults to 0.
            levelpaths (list, optional): List of levelset paths. If None, uses stored levelpaths.
            title (str, optional): Title for the plot. Defaults to None.
            save_path (str, optional): Path to save the plot. If None, displays interactively.
            colorbar (bool, optional): Whether to show the colorbar. Defaults to True.

        Returns:
            matplotlib.figure.Figure: The created figure object.
        """

        plot_values = np.array(self.thickness_values[contour_idx][~np.isnan(self.thickness_values[contour_idx])][:100])[
            ::-1
        ]
        # double plot values with linear interpolation

        # Create bar plot of thickness values
        # fig, ax = plt.subplots(figsize=(10, 4))
        # ax.bar(range(len(plot_values)), plot_values)
        # ax.set_xlabel('Point Index')
        # ax.set_ylabel('Thickness (mm)')
        # ax.set_title('Thickness Distribution')
        # ax.set_ylim(0, 0.06)
        # ax.invert_xaxis()
        # plt.tight_layout()
        # plt.show()

        points, trias = make_mesh_from_contour(self.contours[contour_idx], max_volume=0.5, min_angle=25, verbose=False)

        # make points 3D by adding zero
        points = np.column_stack([points, np.zeros(len(points))])

        levelpaths, _ = self._create_levelpaths(contour_idx, points, trias, num_points=99)

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
            (all_level_points_x, all_level_points_y), all_level_values, (x_grid, y_grid), method="linear", fill_value=0
        )

        # smooth the grid_values
        grid_values = scipy.ndimage.gaussian_filter(grid_values, sigma=5, radius=5)

        # Apply the mask to only show values inside the contour
        masked_values = np.where(mask, grid_values, np.nan)

        # Sample colormaps (e.g., 'binary' and 'gist_heat_r')
        colors1 = plt.cm.binary([0.4] * 128)
        colors2 = plt.cm.hot(np.linspace(0.8, 0.1, 128))

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
            extent=[x_min - margin, x_max + margin, y_min - margin, y_max + margin],
            origin="lower",
            cmap=cmap,
            alpha=1,
            interpolation="bilinear",
            vmin=0,
            vmax=0.10,
            transform=transform,
        )

        plt.imshow(
            masked_values,
            extent=[x_min - margin, x_max + margin, y_min - margin, y_max + margin],
            origin="lower",
            cmap=cmap,
            alpha=1,
            interpolation="bilinear",
            vmin=0,
            vmax=0.10,
            # norm=LogNorm(vmin=1e-3, vmax=0.1),  # Set minimum to avoid log(0)
            transform=transform,
        )

        if colorbar:
            # Add a colorbar
            cbar = plt.colorbar(aspect=10)
            cbar.ax.set_ylim(0.001, 0.054)
            cbar.ax.set_yticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
            # cbar.ax.set_yticks([0.001, 0.01, 0.05])
            # cbar.ax.set_yticklabels(['0.001', '0.01', '0.05'])
            cbar.set_label("p-value (log scale)")

        # Plot the outside contour on top for clear boundary
        plt.plot(outside_contour[0], outside_contour[1], "k-", linewidth=2, label="CC Contour", transform=transform)

        # plot levelpaths
        # for i, path in enumerate(levelpaths):
        #    plt.plot(path[:,0], path[:,1], 'k--', linewidth=1, alpha=0.2, transform=transform)
        # plot midline
        # if midline_equidistant is not None:
        #     midline_x, midline_y = zip(*midline_equidistant)
        #     plt.plot(midline_x, midline_y, 'k--', linewidth=2, transform=transform, alpha=0.2)

        plt.axis("equal")
        plt.title(title, fontsize=14, fontweight="bold")
        # plt.legend(loc='best')
        plt.gca().invert_xaxis()
        plt.axis("off")
        # plt.tight_layout()
        # plt.ylim(-105, -75)
        # plt.xlim(181, 101)
        if save_path is not None:
            self.__make_parent_folder(save_path)
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
        return fig

    def set_mesh(self, vertices, faces, thickness_values=None):
        """Set the mesh vertices, faces, and optional thickness values.

        Args:
            vertices (list or numpy.ndarray): List of vertex coordinates or array of shape (N, 3).
            faces (list or numpy.ndarray): List of face indices or array of shape (M, 3).
            thickness_values (list or numpy.ndarray, optional): Thickness values for each vertex.
        """
        # Handle case when there are no faces (single contour)
        if not faces:
            # For single contour, just store vertices without creating a mesh
            vertices_array = np.vstack(vertices) if vertices else np.array([]).reshape(0, 3)
            self.v = vertices_array
            self.t = np.array([]).reshape(0, 3)
            # Initialize fsinfo attribute that lapy expects
            self.fsinfo = None
            # Skip parent initialization since we have no faces
        else:
            super().__init__(np.vstack(vertices), np.vstack(faces))

        if thickness_values is not None:
            # Filter out empty thickness arrays and concatenate
            valid_thickness = [tv for tv in thickness_values if tv is not None and len(tv) > 0]
            if valid_thickness:
                self.mesh_vertex_colors = np.concatenate(valid_thickness)
            else:
                self.mesh_vertex_colors = np.array([])

    @staticmethod
    def __create_cc_viewmat():
        """
        Create the view matrix for a nice view of the corpus callosum.
        """
        viewLeft = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  # left w top up // right
        transl = pyrr.Matrix44.from_translation((0, 0, 0.4))
        viewmat = transl * viewLeft

        # rotate 10 degrees around x axis
        rot = pyrr.Matrix44.from_x_rotation(np.deg2rad(-10))
        viewmat = viewmat * rot

        # rotate 35 degrees around y axis
        rot = pyrr.Matrix44.from_y_rotation(np.deg2rad(35))
        viewmat = viewmat * rot

        # rotate 10 degrees around z axis
        rot = pyrr.Matrix44.from_z_rotation(np.deg2rad(-8))
        viewmat = viewmat * rot

        return viewmat

    def snap_cc_picture(self, output_path: str, fssurf_file: str | None = None, overlay_file: str | None = None):
        """Snap a picture of the corpus callosum mesh.

        Takes a snapshot of the mesh from a predefined viewpoint, with optional thickness
        overlay. The image is saved to the specified output path.

        Args:
            output_path (str): 
                Path where to save the snapshot image.
            fssurf_file (str | None): Path to a FreeSurfer surface file to use for the snapshot. If None,
                the mesh is saved to a temporary file. Defaults to None.
            overlay_file (str | None): Path to a FreeSurfer overlay file to use for the snapshot. If None,
                the mesh is saved to a temporary file. Defaults to None.

        Note:
            This method uses a temporary file to store the mesh and overlay data during
            the snapshot process.
        """
        self.__make_parent_folder(output_path)
        # Skip snapshot if there are no faces
        if len(self.t) == 0:
            print("Warning: Cannot create snapshot - no faces in mesh")
            return

        # create temp file
        if fssurf_file is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".fssurf", delete=True)
            self.write_fssurf(temp_file.name)
        else:
            temp_file = Path(fssurf_file)

        if overlay_file is None:
            if hasattr(self, "mesh_vertex_colors"):
                overlay_file = tempfile.NamedTemporaryFile(suffix=".w", delete=True)
                # Write thickness values in FreeSurfer .w format
                nib.freesurfer.write_morph_data(overlay_file.name, self.mesh_vertex_colors)
                overlaypath = overlay_file.name
            else:
                overlaypath = None
        else:
            overlaypath = Path(overlay_file).name

        snap1(
            temp_file.name,
            overlaypath=overlaypath,
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

        temp_file.close()
        overlay_file.close()

    def smooth_(self, iterations: int = 1):
        """Smooth the mesh while preserving the z-coordinates.

        This method applies Laplacian smoothing to the mesh vertices while keeping
        the z-coordinates unchanged to maintain the slice structure.

        Args:
            iterations (int, optional): Number of smoothing iterations. Defaults to 1.
        """
        z_values = self.v[:, 2]
        super().smooth_(iterations)
        self.v[:, 2] = z_values

    def save_contours(self, output_path: str):
        """Save the contours to a CSV file.

        Saves all contours and their associated endpoint indices to a CSV file.
        The file format is:
        slice_idx,x,y
        where each point of each contour gets its own row, with special lines indicating
        the start of new contours and their endpoint indices.

        Args:
            output_path (str): Path where to save the CSV file.
        """
        logger.info(f"Saving contours to CSV file: {output_path}")
        with open(output_path, "w") as f:
            # Write header
            f.write("slice_idx,x,y\n")
            # Write data
            for slice_idx, contour in enumerate(self.contours):
                if contour is not None:  # Skip empty slices
                    f.write(
                        f"New contour, anterior_endpoint_idx={self.start_end_idx[slice_idx][0]}, "
                        f"posterior_endpoint_idx={self.start_end_idx[slice_idx][1]}\n"
                    )
                    for point in contour:
                        f.write(f"{slice_idx},{point[0]},{point[1]}\n")

    def load_contours(self, input_path: str):
        """Load contours from a CSV file.

        Loads contours and their associated endpoint indices from a CSV file.
        The file format should match that produced by save_contours:
        slice_idx,x,y with special lines for endpoint indices.

        Args:
            input_path (str): Path to the CSV file containing the contours.

        Note:
            This method will reset any existing contours and endpoint indices.
        """
        current_points = []
        self.contours = []
        self.start_end_idx = []

        with open(input_path) as f:
            # Skip header
            next(f)

            for line in f:
                if line.startswith("New contour"):
                    # If we have points from previous contour, save them
                    if current_points:
                        self.contours.append(np.array(current_points))
                        current_points = []

                    # Extract anterior and posterior endpoint indices
                    # Format: "New contour, anterior_endpoint_idx=X,posterior_endpoint_idx=Y"
                    parts = line.strip().split(",")
                    anterior_idx = int(parts[1].split("=")[1])
                    posterior_idx = int(parts[2].split("=")[1])
                    self.start_end_idx.append((anterior_idx, posterior_idx))
                else:
                    # Parse point data
                    slice_idx, x, y = line.strip().split(",")
                    current_points.append([float(x), float(y)])

            # Don't forget to add the last contour
            if current_points:
                self.contours.append(np.array(current_points))

        # Convert lists to fixed-size arrays
        max_slices = max(len(self.contours), len(self.start_end_idx))
        self.contours = self.contours + [None] * (max_slices - len(self.contours))
        self.start_end_idx = self.start_end_idx + [None] * (max_slices - len(self.start_end_idx))

    def save_thickness_values(self, output_path: str):
        """Save thickness values to a CSV file.

        Saves all thickness values to a CSV file in the format:
        slice_idx,thickness
        where each thickness value gets its own row.

        Args:
            output_path (str): Path where to save the CSV file.
        """
        logger.info(f"Saving thickness data to CSV file: {output_path}")
        with open(output_path, "w") as f:
            # Write header
            f.write("slice_idx,thickness\n")
            # Write data
            for slice_idx, thickness in enumerate(self.thickness_values):
                if thickness is not None:  # Skip empty slices
                    for value in thickness:
                        f.write(f"{slice_idx},{value}\n")

    def load_thickness_values(self, input_path: str, original_thickness_vertices_path: str | None = None):
        """Load thickness values from a CSV file.

        Loads thickness values from a CSV file and optionally associates them with specific
        vertices using a measurement points file.

        Args:
            input_path (str): 
                Path to the CSV file containing thickness values.
            original_thickness_vertices_path (str, optional): 
                Path to a file containing the
                indices of vertices where thickness was measured. If None, assumes thickness
                values correspond to all vertices in order.

        Raises:
            ValueError: 
                If the number of thickness values doesn't match the number of
                measurement points, or if the number of slices is inconsistent.
        """
        data = np.loadtxt(input_path, delimiter=",", skiprows=1)
        slice_indices = data[:, 0].astype(int)
        values = data[:, 1]

        # Group values by slice_idx
        unique_slices = np.unique(slice_indices)

        # split data into slices
        loaded_thickness_values = [None] * (max(unique_slices) + 1)
        for slice_idx in unique_slices:
            mask = slice_indices == slice_idx
            loaded_thickness_values[slice_idx] = values[mask]

        if original_thickness_vertices_path is None:
            # check that the number of thickness values for each slice is equal to the number of points in the contour
            for slice_idx, thickness in enumerate(loaded_thickness_values):
                if thickness is not None:
                    assert len(thickness) == len(self.contours[slice_idx]), (
                        "Number of thickness values does not match number of points in the contour, maybe you need to "
                        "provide the measurement points file"
                    )
            # fill original_thickness_vertices with all indices
            self.original_thickness_vertices = [
                np.arange(len(self.contours[slice_idx])) for slice_idx in range(len(self.contours))
            ]
        else:
            loaded_original_thickness_vertices = self._load_thickness_measurement_points(
                original_thickness_vertices_path
            )

            if len(loaded_original_thickness_vertices) != len(loaded_thickness_values):
                raise ValueError(
                    "Number of slices in measurement points does not match number of "
                    "slices in provided thickness values"
                )

            # check that original_thickness_vertices is equal to number of measurement points for each slice
            for slice_idx, vertex_indices in enumerate(loaded_original_thickness_vertices):
                if len(vertex_indices) // 2 == len(loaded_thickness_values[slice_idx]) or len(
                    vertex_indices
                ) // 2 == np.sum(~np.isnan(loaded_thickness_values[slice_idx])):
                    is_thickness_profile = True
                elif len(vertex_indices) == len(loaded_thickness_values[slice_idx]) or len(vertex_indices) == np.sum(
                    ~np.isnan(loaded_thickness_values[slice_idx])
                ):
                    is_thickness_profile = False
                else:
                    raise ValueError("Number of measurement points does not match number of thickness values")

            # create nan thickness value array for each slice
            new_thickness_values = [
                np.full(len(self.contours[slice_idx]), np.nan) for slice_idx in range(len(self.contours))
            ]
            for slice_idx, vertex_indices in enumerate(loaded_original_thickness_vertices):
                if is_thickness_profile:
                    new_thickness_values[slice_idx][vertex_indices] = np.concatenate(
                        [loaded_thickness_values[slice_idx], loaded_thickness_values[slice_idx][::-1]]
                    )
                else:
                    try:
                        new_thickness_values[slice_idx][vertex_indices] = loaded_thickness_values[slice_idx][
                            ~np.isnan(loaded_thickness_values[slice_idx])]
                    except IndexError as err:
                        logger.error(
                            f"Tried to load "
                            f"{loaded_thickness_values[slice_idx][~np.isnan(loaded_thickness_values[slice_idx])]} "
                            f"values, but template has {new_thickness_values[slice_idx][vertex_indices]} values, "
                            "supply a correct template to visualize the thickness values"
                        )
                        raise ValueError(
                            f"Tried to load "
                            f"{loaded_thickness_values[slice_idx][~np.isnan(loaded_thickness_values[slice_idx])]} "
                            f"values, but template has {new_thickness_values[slice_idx][vertex_indices]} values, "
                            "supply a correct template to visualize the thickness values"
                        ) from err
            self.thickness_values = new_thickness_values

    @staticmethod
    def __make_parent_folder(filename: str):
        """Make the parent folder of the given filename.
        """
        output_folder = Path(filename).parent
        output_folder.mkdir(parents=False, exist_ok=True)

    def to_fs_coordinates(self, vox_size: tuple[int, int, int], image_size: tuple[int, int, int]):
        """Convert mesh coordinates to FreeSurfer coordinate system.

        Transforms the mesh vertices from the original coordinate system to the
        FreeSurfer coordinate system by reordering axes and applying appropriate offsets.
        """
        self.v = self.v[:, [2, 0, 1]] # LIA to ALI?
        self.v *= (vox_size[0] **2) ## ???
        self.v[:, 1] -= image_size[1] * vox_size[1] // 2 # move 0 to center of image
        self.v[:, 2] += image_size[2] * vox_size[2] // 2 
        self.v[:, 0] += vox_size[0] / 2
        
        

    def write_fssurf(self, filename):
        """Write the mesh to a FreeSurfer surface file.

        Args:
            filename (str): Path where to save the FreeSurfer surface file.

        Returns:
            The result of the parent class's write_fssurf method.
        """
        self.__make_parent_folder(filename)
        return super().write_fssurf(filename)

    def write_overlay(self, filename):
        """Write the thickness values as a FreeSurfer overlay file.

        Args:
            filename (str): Path where to save the overlay file.

        Returns:
            The result of writing the morph data using nibabel.
        """
        self.__make_parent_folder(filename)
        return nib.freesurfer.write_morph_data(filename, self.mesh_vertex_colors)

    def save_thickness_measurement_points(self, filename):
        """Write the thickness measurement points to a CSV file.

        Saves the indices of vertices where thickness was measured for each slice
        in CSV format: slice_idx,vertex_idx

        Args:
            filename (str): Path where to save the CSV file.
        """
        self.__make_parent_folder(filename)
        logger.info(f"Saving thickness measurement points to CSV file: {filename}")
        with open(filename, "w") as f:
            f.write("slice_idx,vertex_idx\n")
            for slice_idx, vertex_indices in enumerate(self.original_thickness_vertices):
                if vertex_indices is not None:
                    for vertex_idx in vertex_indices:
                        f.write(f"{slice_idx},{vertex_idx}\n")

    @staticmethod
    def _load_thickness_measurement_points(filename):
        """Load thickness measurement points from a CSV file.

        Args:
            filename (str): Path to the CSV file containing measurement points.

        Returns:
            list: List of arrays containing vertex indices for each slice where
                thickness was measured.
        """
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
        slice_indices = data[:, 0].astype(int)
        vertex_indices = data[:, 1].astype(int)

        # Group values by slice_idx
        unique_slices = np.unique(slice_indices)

        # split data into slices
        original_thickness_vertices = [None] * (max(unique_slices) + 1)
        for slice_idx in unique_slices:
            mask = slice_indices == slice_idx
            original_thickness_vertices[slice_idx] = vertex_indices[mask]
        return original_thickness_vertices
