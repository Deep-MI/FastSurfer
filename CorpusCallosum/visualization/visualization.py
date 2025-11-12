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

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def plot_standardized_space(
    ax_row: list[plt.Axes], 
    vol: np.ndarray, 
    ac_coords: np.ndarray, 
    pc_coords: np.ndarray
) -> None:
    """Plot standardized space visualization across three views.

    Parameters
    ----------
    ax_row : list[plt.Axes]
        Row of axes to plot on (should be length 3)
    vol : np.ndarray
        Volume data to visualize
    ac_coords : np.ndarray
        AC coordinates in standardized space
    pc_coords : np.ndarray
        PC coordinates in standardized space

    Notes
    -----
    Creates three views:
    - Axial (top view)
    - Sagittal (side view)
    - Coronal (front view)
    """
    ax_row[0].set_title("Standardized")

    for i, (a, b, _) in ((2, 1, "Axial"), (2, 0, "Sagittal"), (1, 0, "Coronal")):
        ax_row[i].scatter(ac_coords[a], ac_coords[b], color="red", marker="x")
        ax_row[i].scatter(pc_coords[a], pc_coords[b], color="blue", marker="x")
        ax_row[i].imshow(vol[(slice(None),) * i + (vol.shape[i] // 2,)], cmap="gray")


def visualize_coordinate_spaces(
    orig: "nib.Nifti1Image",
    upright: np.ndarray,
    standardized: np.ndarray,
    ac_coords_orig: np.ndarray,
    pc_coords_orig: np.ndarray,
    ac_coords_3d: np.ndarray,
    pc_coords_3d: np.ndarray,
    ac_coords_standardized: np.ndarray,
    pc_coords_standardized: np.ndarray,
    output_plot_path: str | Path,
) -> None:
    """Visualize the AC and PC coordinates in different coordinate spaces.

    Creates a figure showing the anterior and posterior commissure points
    in three different coordinate spaces for testing/debugging.

    Parameters
    ----------
    orig : nibabel.Nifti1Image
        Original image volume
    upright : np.ndarray
        Volume in fsaverage space
    standardized : np.ndarray
        Volume in standardized space
    ac_coords_orig : np.ndarray
        AC coordinates in original space
    pc_coords_orig : np.ndarray
        PC coordinates in original space
    ac_coords_3d : np.ndarray
        AC coordinates in fsaverage space
    pc_coords_3d : np.ndarray
        PC coordinates in fsaverage space
    ac_coords_standardized : np.ndarray
        AC coordinates in standardized space
    pc_coords_standardized : np.ndarray
        PC coordinates in standardized space
    output_plot_path : str or Path
        Directory to save visualization

    Notes
    -----
    Saves a visualization of the anterior (red) and posterior (blue) commisure in three different view: 
    1. the orig image (orig), 
    2. fs-average standardized image space, and
    3. standardized image space
    as a single image named 'ac_pc_spaces.png' in `output_dir`.
    """
    fig, ax = plt.subplots(3, 4)
    ax = ax.T

    # Original space - using plot_standardized_space
    plot_standardized_space(ax[0], orig.get_fdata(), ac_coords_orig, pc_coords_orig)
    ax[0, 0].set_title("Orig")

    # Fsaverage space
    plot_standardized_space(ax[1], upright, ac_coords_3d, pc_coords_3d)
    ax[1, 0].set_title("Fsaverage")

    # Standardized space
    plot_standardized_space(ax[2], standardized, ac_coords_standardized, pc_coords_standardized)
    ax[2, 0].set_title("Standardized")
    # Format all subplots
    for a in ax.flatten():
        a.set_aspect("equal", adjustable="box")
        a.axis("off")

    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_contours(
    transformed: np.ndarray,
    split_contours: list[np.ndarray] | None = None,
    midline_equidistant: np.ndarray | None = None,
    levelpaths: list[np.ndarray] | None = None,
    output_path: str | Path | None = None,
    ac_coords: np.ndarray | None = None,
    pc_coords: np.ndarray | None = None,
    vox_size: float | None = None,
    title: str = "",
) -> None:
    """Plot contours and subdivisions of the corpus callosum.

    Parameters
    ----------
    transformed : np.ndarray
        Transformed image data
    split_contours : list[np.ndarray], optional
        List of contour arrays for each subdivision, by default None
    midline_equidistant : np.ndarray, optional
        Midline points at equidistant spacing, by default None
    levelpaths : list[np.ndarray], optional
        List of level paths for visualization, by default None
    output_path : str or Path, optional
        Path to save the plot, by default None
    ac_coords : np.ndarray, optional
        AC coordinates for visualization, by default None
    pc_coords : np.ndarray, optional
        PC coordinates for visualization, by default None
    vox_size : float, optional
        Voxel size for scaling, by default None
    title : str, optional
        Title for the plot, by default ""
    debug : bool, optional
        Whether to show debug information, by default False

    Notes
    -----
    Creates a visualization of the corpus callosum contours and their subdivisions.
    If output_path is provided, saves the plot to that location.
    """

    # scale contour data by vox_size
    if split_contours:
        split_contours = np.stack(split_contours, axis=0) / vox_size
    if midline_equidistant:
        midline_equidistant = midline_equidistant / vox_size
    if levelpaths:
        levelpaths = np.stack(levelpaths, axis=0) / vox_size

    NO_PLOTS = 1 + int(split_contours is not None)

    _, ax = plt.subplots(1, NO_PLOTS, sharex=True, sharey=True, figsize=(15, 10))

    # NOTE: For all plots imshow shows y inverted
    current_plot = 0


    if split_contours is not None:
        ax[current_plot].imshow(transformed[transformed.shape[0] // 2], cmap="gray")
        # ax[0].imshow(cc_mask, cmap='autumn')
        ax[current_plot].set_title(title)
        for i in range(len(split_contours)):
            ax[current_plot].fill(split_contours[i][0, :], -split_contours[i][1, :], color="steelblue", alpha=0.25)
            ax[current_plot].plot(
                split_contours[i][0, :], -split_contours[i][1, :], color="mediumblue", linestyle="dotted", linewidth=0.7
            )

        ax[current_plot].plot(split_contours[0][0, :], -split_contours[0][1, :], color="mediumblue", linewidth=0.7)
        ax[current_plot].scatter(ac_coords[1], ac_coords[0], color="red", marker="x")
        ax[current_plot].scatter(pc_coords[1], pc_coords[0], color="blue", marker="x")
        current_plot += 1

    reference_contour = split_contours[0]

    ax[current_plot].imshow(transformed[transformed.shape[0] // 2], cmap="gray")
    # ax[2].imshow(cc_mask, cmap='autumn')
    for i in range(len(levelpaths)):
        ax[current_plot].plot(levelpaths[i][:, 0], -levelpaths[i][:, 1], color="brown", linewidth=0.8)
    ax[current_plot].set_title("Midline & Levelpaths")
    ax[current_plot].plot(midline_equidistant[:, 0], -midline_equidistant[:, 1], color="red")
    ax[current_plot].plot(reference_contour[0, :], -reference_contour[1, :], color="red", linewidth=0.5)

    for a in ax.flatten():
        a.set_aspect("equal", adjustable="box")
        a.axis("off")

    # get bounding box of contours
    padding = 30
    ax[0].set_xlim(reference_contour[0, :].min() - padding, reference_contour[0, :].max() + padding)
    ax[0].set_ylim((-reference_contour[1, :]).max() + padding, (-reference_contour[1, :]).min() - padding)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # plt.show()


def plot_midplane(grid_orig: np.ndarray, orig: np.ndarray) -> None:
    """Create a 3D visualization of grid points in original image space.

    Parameters
    ----------
    grid_orig : np.ndarray
        Grid points in original space, shape (3, N)
    orig : np.ndarray
        Original image for dimension reference

    Notes
    -----
    The function:
    1. Creates a 3D scatter plot of grid points
    2. Samples every 40th point to avoid overcrowding
    3. Sets axis limits based on original image dimensions
    4. Shows the plot interactively
    """
    # Create a figure showing grid points in original space

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot every 10th point to avoid overcrowding
    sample_idx = np.arange(0, grid_orig.shape[1], 40)
    ax.scatter(*grid_orig[:3, sample_idx], c="r", alpha=0.1, marker=".")

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Grid Points in Original Image Space")

    # Set axis limits to image dimensions
    ax.set_xlim(0, orig.shape[0])
    ax.set_ylim(0, orig.shape[1])
    ax.set_zlim(0, orig.shape[2])

    # Save plot
    plt.show()
    # plt.savefig('grid_points.png')
    # plt.close()
