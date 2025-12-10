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

"""
This module provides the ``CCContour`` class for reading, writing, and
manipulating 2D corpus callosum contours together with per-vertex thickness
values. Typical template outputs (from ``fastsurfer_cc.py --save_template``)
emit one set per slice:

- ``contour_<idx>.txt``: CSV with header ``New contour, anterior_endpoint_idx=<a>, posterior_endpoint_idx=<p>`` followed
  by ``x,y`` rows.
- ``thickness_values_<idx>.txt``: CSV with header ``thickness`` and one value per contour vertex.
- ``thickness_measurement_points_<idx>.txt``: CSV with header ``vertex_idx`` listing the vertices where thickness was
  measured.
"""

import re
from pathlib import Path
from typing import Literal

import lapy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from scipy.ndimage import gaussian_filter1d

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.shape.endpoint_heuristic import smooth_contour
from CorpusCallosum.shape.thickness import cc_thickness, make_mesh_from_contour

logger = logging.get_logger(__name__)


class CCContour:
    """A class for representing and manipulating corpus callosum (CC) contours.

    This class provides functionality for manipulating and analyzing corpus callosum contours.

    Attributes
    ----------
    contour : np.ndarray
        Array of shape (N, 2) containing 2D contour points.
    thickness_values : np.ndarray
        Array of shape (N,) for thickness measurements for each contour point.
    endpoint_idxs : tuple[int, int]
        Tuple containing start and end indices for the contour.

    Examples
    --------
    >>> from CorpusCallosum.shape.contour import CCContour
    >>>
    >>> contour = CCContour(contour_points, thickness_values,
    >>>                     endpoint_idxs=(anterior_idx, posterior_idx),
    >>>                     resolution=1.0)
    >>> contour.fill_thickness_values()   # interpolate missing values
    >>> contour.smooth_contour(window_size=5)
    >>> contour.save_contour("contour_0.txt")
    >>> contour.save_thickness_values("thickness_values_0.txt")
    >>> contour.save_thickness_measurement_points("thickness_measurement_points_0.txt")
    """
    
    def __init__(
        self,
        contour: np.ndarray[tuple[Literal["N", 2]], np.dtype[float]],
        thickness_values: np.ndarray[tuple[Literal["N"]], np.dtype[float]],
        endpoint_idxs: tuple[int, int] | None = None,
        resolution: float = 1.0
    ):
        """Initialize a CCContour object.

        Parameters
        ----------
        contour : np.ndarray
            Array of shape (N, 2) containing 2D contour points.
        thickness_values : np.ndarray
            Array of thickness measurements for each contour point.
        endpoint_idxs : tuple[int, int], optional
            Tuple containing start and end indices for the contour.
        resolution : float, default=1.0
            The left-right spacing.
        """
        self.contour = contour
        if self.contour.shape[1] != 2:
            raise ValueError(f"Contour must be a 2D array, but is {self.contour.shape}")
        self.thickness_values = thickness_values
        if self.contour.shape[0] != len(thickness_values):
            raise ValueError(
                f"Number of contour points ({self.contour.shape[0]}) does not match number of thickness values "
                f"({len(thickness_values)})",
            )
        # write vertex indices where thickness values are not nan
        self.original_thickness_vertices = np.where(~np.isnan(thickness_values))[0]
        self.resolution = resolution

        if endpoint_idxs is None:
            self.endpoint_idxs = (0, len(contour) // 2)
        else:
            self.endpoint_idxs = endpoint_idxs

    def smooth_contour(self, window_size: int = 5) -> None:
        """Smooth a contour using a moving average filter.

        Parameters
        ----------
        window_size : int, default=5
            Size of the smoothing window.

        Notes
        -----
        Uses smooth_contour from cc_endpoint_heuristic module to:
        1. Extract x and y coordinates.
        2. Apply moving average smoothing.
        3. Update contour with smoothed coordinates.
        """
        x, y = self.contour.T
        x, y = smooth_contour(x, y, window_size)
        self.contour = np.array([x, y]).T

    def copy(self) -> "CCContour":
        """Copy the contour.
        """
        return CCContour(self.contour.copy(), self.thickness_values.copy(), self.endpoint_idxs, self.resolution)
    
    def get_contour_edge_lengths(self) -> np.ndarray:
        """Get the lengths of the edges of a contour.

        Returns
        -------
        np.ndarray
            Array of edge lengths for the contour.

        Notes
        -----
        Edge lengths are calculated as Euclidean distances between consecutive points
        in the contour.
        """
        edges = np.diff(self.contour, axis=0)
        return np.sqrt(np.sum(edges**2, axis=1))

    def create_levelpaths(self, 
                           num_points: int,
                           update_data: bool = True
                           ) -> tuple[list[np.ndarray], list[float]]:
        midline_len, thickness, curvature, midline_equi, \
            levelpaths, contour_with_thickness, endpoint_idxs = cc_thickness(
                self.contour,
                self.endpoint_idxs,
                n_points=num_points,
            )
        
        if update_data:
            self.contour = contour_with_thickness[:, :2]
            self.thickness_values = contour_with_thickness[:,2]
            self.original_thickness_vertices = np.where(~np.isnan(self.thickness_values))[0]
            self.endpoint_idxs = endpoint_idxs

        return levelpaths, thickness
    
    def set_thickness_values(self, thickness_values: np.ndarray, use_measurement_points: bool = False) -> None:
        """Set the thickness values for the contour.
        This is useful to update the thickness values for specific plots.
        
        Parameters
        ----------
        thickness_values : np.ndarray
            Array of thickness values for the contour.
        use_measurement_points : bool, optional
            Whether to use the measurement points to set the thickness values, by default False.
        """
        if use_measurement_points:
            if len(thickness_values) == len(self.original_thickness_vertices):
                self.thickness_values = np.full(len(self.contour), np.nan)
                self.thickness_values[self.original_thickness_vertices] = thickness_values
            else:
                raise ValueError(
                    "Number of thickness values does not match number of measurement points "
                    f"{len(self.original_thickness_vertices)}.",
                )
        else:
            if len(thickness_values) != len(self.contour):
                raise ValueError(
                    f"The number of thickness values does not match number of points in the contour "
                    f"{len(self.contour)}.",
                )
            self.thickness_values = thickness_values

    def fill_thickness_values(self) -> None:
        """Interpolate missing thickness values using weighted averaging.

        Notes
        -----
        The function:
        1. Processes each contour with missing thickness values.
        2. For each missing value:
        - Finds two closest points with known thickness.
        - Calculates distances along contour.
        - Computes weighted average based on inverse distance.
        3. Updates thickness values in place.

        The weights are calculated as inverse distances to ensure closer
        points have more influence on the interpolated value.

        """
        thickness = self.thickness_values
        edge_lengths = self.get_contour_edge_lengths()

        # Find indices of points with known thickness
        known_idx = np.where(~np.isnan(thickness))[0]

        if len(known_idx) == 0:
            logger.warning("No known thickness values; skipping interpolation")
            return
        if len(known_idx) == 1:
            logger.warning("Only one known thickness value; skipping interpolation")
            thickness[np.isnan(thickness)] = thickness[known_idx[0]]
            self.thickness_values = thickness
            return

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

        self.thickness_values = thickness

    def smooth_thickness_values(self, iterations: int = 1) -> None:
        """Smooth the thickness values using a Gaussian filter.

        Parameters
        ----------
        iterations : int, optional
            Number of smoothing iterations, by default 1.

        Notes
        -----
        Applies Gaussian smoothing with sigma=5 to thickness values
        for each slice that has measurements.
        """
        for i in range(len(self.thickness_values)):
            if self.thickness_values[i] is not None:
                self.thickness_values[i] = gaussian_filter1d(self.thickness_values[i], sigma=5)
    
    def plot_contour(self, output_path: str | None = None) -> None:
        """Plot a single contour with thickness values.

        Parameters
        ----------
        output_path : str
            Path where to save the plot.

        Notes
        -----
        Creates a 2D visualization with:
        - Points colored by thickness values.
        - Gray points for missing thickness values.
        - Connected contour line.
        - Grid, labels, and legend.
        """
        if output_path is not None:
            self.__make_parent_folder(output_path)

        contour = self.contour

        plt.figure(figsize=(10, 10))
        # Get thickness values for this slice
        thickness = self.thickness_values

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
        plt.title("CC contour")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=300)
        else:
            plt.show()


    def plot_contour_colorfill(
        self,
        plot_values: np.ndarray,
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
        plot_values : np.ndarray
            Array of values to plot on CC from anterior to posterior (left to right in the plot).
        title : str, optional
            Title for the plot.
        save_path : str, optional
            Path to save the plot. If None, displays interactively.
        colorbar : bool, default=True
            Whether to show the colorbar.
        mode : {"p-value", "icc", "thickness"}, default="p-value"
            Mode of the plot.
        
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object.
        """
        plot_values = plot_values[::-1] # make sure values are plotted left to right (anterior to posterior)

        points, _ = make_mesh_from_contour(self.contour, max_volume=0.5, min_angle=25, verbose=False)

        # make points 3D by adding zero
        points = np.column_stack([points, np.zeros(len(points))])

        levelpaths, _ = self.create_levelpaths(num_points=len(plot_values)-1, update_data=False)

        outside_contour = self.contour.T

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

            # add third dimension to path
            path = np.column_stack([path, np.zeros(len(path))])

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
        elif mode == "thickness":
            # Blue to red colormap for thickness values
            cmap = plt.cm.coolwarm
        else:
            raise ValueError(f"Invalid mode '{mode}'")

        # Combine the color samples for p-value and icc modes
        if mode != "thickness":
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
            vmin=0 if mode != "thickness" else np.nanmin(plot_values),
            vmax=0.10 if mode == "p-value" else (1 if mode == "icc" else np.nanmax(plot_values)),
            transform=transform,
        )

        plt.imshow(
            masked_values,
            extent=(x_min - margin, x_max + margin, y_min - margin, y_max + margin),
            origin="lower",
            cmap=cmap,
            alpha=1,
            interpolation="bilinear",
            vmin=0 if mode != "thickness" else np.nanmin(plot_values),
            vmax=0.10 if mode == "p-value" else (1 if mode == "icc" else np.nanmax(plot_values)),
            # norm=LogNorm(vmin=1e-3, vmax=0.1),  # Set minimum to avoid log(0)
            transform=transform,
        )

        if colorbar:
            # Add a colorbar
            cbar = plt.colorbar(aspect=15)
            if mode == "p-value":
                cbar.ax.set_ylim(0.001, 0.054)
                cbar.ax.set_yticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
                cbar.set_label("p-value (log scale)")
            elif mode == "icc":
                cbar.ax.set_ylim(0, 1)
                cbar.ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                cbar.ax.set_label("Intraclass correlation coefficient")
            elif mode == "thickness":
                # Set limits based on actual thickness values
                thickness_min = np.nanmin(plot_values)
                thickness_max = np.nanmax(plot_values)
                cbar.ax.set_ylim(thickness_min, thickness_max)
                cbar.set_label("Thickness (mm)")

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

    def save_contour(self, output_path: Path | str) -> None:
        """Save the contours to a CSV file.

        Parameters
        ----------
        output_path : Path, str
            Path to save the CSV file.

        Notes
        -----
        The function saves contours in CSV format with:
        - Header: slice_idx,x,y.
        - Special lines indicating new contours with endpoint indices.
        - Each point gets its own row with slice index and coordinates.
        """
        self.__make_parent_folder(output_path)
        logger.info(f"Saving contours to CSV file: {output_path}")
        with open(output_path, "w") as f:
            
            f.write(
                f"New contour, anterior_endpoint_idx={self.endpoint_idxs[0]}, "
                f"posterior_endpoint_idx={self.endpoint_idxs[1]}\n"
            )
            f.write("x,y\n")
            for point in self.contour:
                f.write(f"{point[0]},{point[1]}\n")

    def load_contour(self, input_path: str) -> None:
        """Load contour from a CSV file.

        Parameters
        ----------
        input_path : str
            Path to the CSV file containing the contours.

        Raises
        ------
        ValueError
            If the file format doesn't match expected structure.

        Notes
        -----
        The function:
        1. Reads CSV file with format matching save_contours output.
        2. Processes special lines for endpoint indices.
        3. Reconstructs contours and endpoint indices for each slice.
        4. Converts lists to fixed-size arrays with None padding.
        """
        current_points = []
        self.contours = []
        self.endpoint_idxs = []

        with open(input_path) as f:
            header = next(f).strip()
            # Parse endpoint indices from header
            anterior_match = re.search(r'anterior_endpoint_idx=(\d+)', header)
            posterior_match = re.search(r'posterior_endpoint_idx=(\d+)', header)
            assert anterior_match and posterior_match, "Header does not contain endpoint indices"

            anterior_idx = int(anterior_match.group(1))
            posterior_idx = int(posterior_match.group(1))
            self.endpoint_idxs = (anterior_idx, posterior_idx)

            # Skip column names
            next(f)

            for line in f:
                x, y = line.strip().split(",")
                current_points.append([float(x), float(y)])
        self.contour = np.array(current_points)

    def save_thickness_values(self, output_path: Path | str) -> None:
        """Save thickness values to a CSV file.

        Parameters
        ----------
        output_path : Path, str
            Path to save the CSV file.

        Notes
        -----
        The function saves thickness values in CSV format with:
        - Header: thickness.
        - Each thickness value gets its own row with slice index.
        - Skips slices with no thickness values.
        """
        self.__make_parent_folder(output_path)
        logger.info(f"Saving thickness data to CSV file: {output_path}")
        with open(output_path, "w") as f:
            f.write("thickness\n")
            for value in self.thickness_values:
                f.write(f"{value}\n")

    def load_thickness_values(
        self,
        input_path: str,
    ) -> None:
        """Load thickness values from a CSV file.

        Parameters
        ----------
        input_path : str
            Path to the CSV file containing thickness values.
        original_thickness_vertices_path : str or None, optional
            Path to a file containing the indices of vertices where thickness
            was measured, by default None.

        Raises
        ------
        ValueError
            If number of thickness values doesn't match measurement points
            or if number of slices is inconsistent.

        Notes
        -----
        The function:
        1. Reads thickness values from CSV file.
        2. Groups values by slice index.
        3. Optionally associates values with specific vertices.
        4. Handles both full contour and profile measurements.

        
        """
        data = np.loadtxt(input_path, delimiter=",", skiprows=1)
        if data.ndim == 0:
            values = np.array([float(data)])
        elif data.ndim == 1:
            values = data.astype(float)
        else:
            raise ValueError("Thickness values file must contain a single column")

        if len(values) != len(self.contour):
            if np.sum(~np.isnan(values)) == len(self.original_thickness_vertices):
                new_values = np.full(len(self.contour), np.nan)
                new_values[self.original_thickness_vertices] = values[~np.isnan(values)]
            else:
                raise ValueError(
                    f"Number of thickness values {len(values)} does not match number of points in the "
                    f"contour {len(self.contour)} and current number of measururement points "
                    f"{len(self.original_thickness_vertices)} does not match the number of set thickness values "
                    f"{np.sum(~np.isnan(values))}."
                )
        else:
            raise ValueError(f"Number of thickness values in {input_path} does not match the vertices of the path!")

        self.thickness_values = new_values
