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
        Row of axes to plot on (should be length 3).
    vol : np.ndarray
        Volume data to visualize.
    ac_coords : np.ndarray
        AC coordinates in standardized space.
    pc_coords : np.ndarray
        PC coordinates in standardized space.

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
        Original image volume.
    upright : np.ndarray
        Volume in fsaverage space.
    standardized : np.ndarray
        Volume in standardized space.
    ac_coords_orig : np.ndarray
        AC coordinates in original space.
    pc_coords_orig : np.ndarray
        PC coordinates in original space.
    ac_coords_3d : np.ndarray
        AC coordinates in fsaverage space.
    pc_coords_3d : np.ndarray
        PC coordinates in fsaverage space.
    ac_coords_standardized : np.ndarray
        AC coordinates in standardized space.
    pc_coords_standardized : np.ndarray
        PC coordinates in standardized space.
    output_plot_path : str or Path
        Directory to save visualization.

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
    plot_standardized_space(ax[0], np.asarray(orig.dataobj), ac_coords_orig, pc_coords_orig)
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
    """Creates a figure of the contours (shape) and the subdivisions of the corpus callosum.

    Parameters
    ----------
    transformed : np.ndarray
        Transformed image data.
    split_contours : list[np.ndarray], optional
        List of contour arrays for each subdivision (ignore contours on None).
    midline_equidistant : np.ndarray, optional
        Midline points at equidistant spacing (ignore midline on None).
    levelpaths : list[np.ndarray], optional
        List of level paths for visualization (ignore level paths on None).
    output_path : str or Path, optional
        Path to save the plot (do not save on None).
    ac_coords : np.ndarray, optional
        AC coordinates for visualization (ignore AC on None).
    pc_coords : np.ndarray, optional
        PC coordinates for visualization (ignore PC on None).
    vox_size : float, optional
        Voxel size for scaling.
    title : str, default=""
        Title for the plot.

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

    has_first_plot = bool(split_contours) or bool(ac_coords) or bool(pc_coords)
    num_plots = 1 + int(has_first_plot)

    _, ax = plt.subplots(1, num_plots, sharex=True, sharey=True, figsize=(15, 10))

    # NOTE: For all plots imshow shows y inverted
    current_plot = 0

    if has_first_plot:
        ax[current_plot].imshow(transformed[transformed.shape[0] // 2], cmap="gray")
        ax[current_plot].set_title(title)
    if split_contours:
        for i, this_contour in enumerate(split_contours):
            ax[current_plot].fill(this_contour[0, :], -this_contour[1, :], color="steelblue", alpha=0.25)
            kwargs = {"color": "mediumblue", "linewidth": 0.7, "linestyle": "solid" if i != 0 else "dotted"}
            ax[current_plot].plot(this_contour[0, :], -this_contour[1, :], **kwargs)
    if ac_coords:
        ax[current_plot].scatter(ac_coords[1], ac_coords[0], color="red", marker="x")
    if pc_coords:
        ax[current_plot].scatter(pc_coords[1], pc_coords[0], color="blue", marker="x")
    current_plot += int(has_first_plot)

    reference_contour = split_contours[0]
    ax[current_plot].imshow(transformed[transformed.shape[0] // 2], cmap="gray")
    for this_path in levelpaths:
        ax[current_plot].plot(this_path[:, 0], -this_path[:, 1], color="brown", linewidth=0.8)
    ax[current_plot].set_title("Midline & Levelpaths")
    ax[current_plot].plot(midline_equidistant[:, 0], -midline_equidistant[:, 1], color="red")
    ax[current_plot].plot(reference_contour[0, :], -reference_contour[1, :], color="red", linewidth=0.5)

    padding = 30
    for a in ax.flatten():
        a.set_aspect("equal", adjustable="box")
        a.axis("off")
        # get bounding box of contours
        a.set_xlim(reference_contour[0, :].min() - padding, reference_contour[0, :].max() + padding)
        a.set_ylim((-reference_contour[1, :]).max() + padding, (-reference_contour[1, :]).min() - padding)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
