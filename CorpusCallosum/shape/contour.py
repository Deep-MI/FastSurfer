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

from pathlib import Path
from typing import Literal

import lapy
import numpy as np
import scipy.interpolate
from scipy.ndimage import gaussian_filter1d

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.shape.endpoint_heuristic import smooth_contour
from FastSurferCNN.utils.common import suppress_stdout

logger = logging.get_logger(__name__)


class CCContour:
    """A class for representing and manipulating corpus callosum (CC) contours.

    This class provides functionality for manipulating and analyzing corpus callosum contours.

    Attributes
    ----------
    contour : np.ndarray
        Array of shape (N, 2) containing 2D contour points.
    endpoint_idxs : tuple[int, int]
        Tuple containing start and end indices for the contour.
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
        endpoint_idxs : tuple[int, int]
            Tuple containing start and end indices for the contour.
        """
        self.contour = contour
        self.thickness_values = thickness_values
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
        contour_idx : int
            Index of the contour to smooth.
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

    
    def get_contour_edge_lengths(self) -> np.ndarray:
        """Get the lengths of the edges of a contour.

        Parameters
        ----------
        contour_idx : int
            Index of the contour to get the edge lengths for.

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
    


    def _create_levelpaths(
        self,
        points: np.ndarray,
        trias: np.ndarray,
        num_points: int | None = None
    ) -> tuple[list[np.ndarray], list[float]]:
        """Create level paths for thickness measurements.

        Parameters
        ----------
        contour_idx : int
            Index of the contour to process
        points : np.ndarray
            Array of shape (N, 2) containing mesh points
        trias : np.ndarray
            Array of shape (M, 3) containing triangle indices
        num_points : int or None, optional
            Number of points to sample along the midline, by default None

        Returns
        -------
        tuple[list[np.ndarray], list[float]]
            - levelpaths : List of arrays containing level path coordinates
            - thickness_values : List of thickness values for each level path

        Notes
        -----
        The function:
        1. Creates a triangular mesh from the points
        2. Finds boundary points and endpoints
        3. Solves Poisson equation for level sets
        4. Extracts level paths and interpolates thickness values
        """

        with suppress_stdout():
            cc_tria = lapy.TriaMesh(points, trias)
        # extract boundary curve
        bdr = np.array(cc_tria.boundary_loops()[0])

        # find index of endpoints in bdr list
        iidx1 = np.where(bdr == self.endpoint_idxs[0])[0][0]
        iidx2 = np.where(bdr == self.endpoint_idxs[1])[0][0]

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
        with suppress_stdout():
            fem = lapy.Solver(cc_tria)
            vfunc = fem.poisson(0, (bdr, dcond))
            if num_points is not None:
                # TODO: do midline stuff
                level = 0
                midline_equidistant, midline_length = cc_tria.level_path(vfunc, level, n_points=num_points + 2)
                midline_equidistant = midline_equidistant[:, :2]
                eval_points = midline_equidistant
            else:
                eval_points = self.contour
            gf = lapy.diffgeo.compute_rotated_f(cc_tria, vfunc)

        # interpolate midline to get levels to evaluate
        gf_interp = scipy.interpolate.griddata(cc_tria.v[:, 0:2], gf, eval_points, method="nearest")

        # sort by value
        sorting_idx_gf = np.argsort(gf_interp)
        gf_interp = gf_interp[sorting_idx_gf]
        sorted_thickness_values = self.thickness_values[sorting_idx_gf]

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
    

    def save_thickness_measurement_points(self, filename: Path | str) -> None:
        """Write the thickness measurement points to a CSV file.

        Parameters
        ----------
        filename : Path, str
            Path where to save the CSV file.

        Notes
        -----
        The function saves measurement points in CSV format with:
        - Header: slice_idx,vertex_idx.
        - Each measurement point gets its own row.
        - Skips slices with no measurement points.
        """
        self.__make_parent_folder(filename)
        logger.info(f"Saving thickness measurement points to CSV file: {filename}")
        with open(filename, "w") as f:
            f.write("vertex_idx\n")
            for vertex_idx in self.original_thickness_vertices:
                f.write(f"{vertex_idx}\n")

    @staticmethod
    def _load_thickness_measurement_points(filename: str) -> list[np.ndarray | None]:
        """Load thickness measurement points from a CSV file.

        Parameters
        ----------
        filename : str
            Path to the CSV file containing measurement points.

        Returns
        -------
        list[np.ndarray | None]
            List of arrays containing vertex indices for each slice where
            thickness was measured. None for slices without measurements.

        Notes
        -----
        The function:
        1. Reads CSV file with format: slice_idx,vertex_idx
        2. Groups vertex indices by slice index
        3. Creates a list with length matching max slice index
        4. Fills list with vertex indices arrays or None for missing slices
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
            f.write("x,y\n")
            f.write(
                f"New contour, anterior_endpoint_idx={self.endpoint_idxs[0]}, "
                f"posterior_endpoint_idx={self.endpoint_idxs[1]}\n"
            )
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
        self.start_end_idx = []

        with open(input_path) as f:
            # Skip header
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
        original_thickness_vertices_path: str | None = None
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
        values = data[:, 0]

        if original_thickness_vertices_path is None:
            # check that the number of thickness values for each slice is equal to the number of points in the contour
            assert len(values) == len(self.contour), (
                "Number of thickness values does not match number of points in the contour, maybe you need to "
                "provide the measurement points file"
            )
            # fill original_thickness_vertices with all indices
            self.original_thickness_vertices = np.arange(len(self.contour))
        else:
            loaded_original_thickness_vertices = self._load_thickness_measurement_points(
                original_thickness_vertices_path
            )

            if len(loaded_original_thickness_vertices) != len(values):
                raise ValueError(
                    "Number of measurement points does not match number of thickness values"
                )

            self.thickness_values = values
            logger.error(
                f"Tried to load {len(values[~np.isnan(values)])} values, but template has {len(values)} values, "
                "supply a correct template to visualize the thickness values"
            )