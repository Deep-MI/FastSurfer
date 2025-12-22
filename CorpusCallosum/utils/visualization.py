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

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from CorpusCallosum.utils.types import ContourList, Polygon2dType
from FastSurferCNN.utils import AffineMatrix4x4, Image3d, Vector2d


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
    slice_or_slab: Image3d,
    split_contours: ContourList | None = None,
    midline_equidistant: Polygon2dType | None = None,
    levelpaths: list[Polygon2dType] | None = None,
    output_path: str | Path | list[Path | str] | None = None,
    ac_coords_vox: Vector2d | None = None,
    pc_coords_vox: Vector2d | None = None,
    vox2ras: AffineMatrix4x4 | None = None,
    title: str = "",
) -> None:
    """Creates a figure of the contours (shape) and the subdivisions of the corpus callosum.

    Parameters
    ----------
    slice_or_slab : np.ndarray
        Intensities of the current slice, midslice or midslab (will plot middle slice).
    split_contours : list[np.ndarray], optional
        List of contour arrays for each subdivision (ignore contours on None) in upright AS coordinates each with shape
        (N, 2).
    midline_equidistant : np.ndarray, optional
        Midline points at equidistant spacing (ignore midline on None) in upright AS coordinates with shape (2, N).
    levelpaths : list[np.ndarray], optional
        List of level paths for visualization (ignore level paths on None) in upright AS coordinates each with shape
        (2, N).
    output_path : str or Path or list of Paths, optional
        Path to save the plot (show and do not save on None).
    ac_coords_vox : np.ndarray, optional
        AC coordinates for visualization (ignore AC on None) in LIA voxel coordinates.
    pc_coords_vox : np.ndarray, optional
        PC coordinates for visualization (ignore PC on None) in LIA voxel coordinates.
    vox2ras : AffineMatrix4x4, optional
        Slice vox2ras transformation matrix.
    title : str, default=""
        Title for the plot.

    Notes
    -----
    Creates a visualization of the corpus callosum contours and their subdivisions.
    If output_path is provided, saves the plot to that location.
    """
    from functools import partial

    from nibabel.affines import apply_affine

    if vox2ras is None and None in (split_contours, midline_equidistant, levelpaths):
        raise ValueError("vox_size must be provided if split_contours, midline_equidistant, or levelpaths are given.")
    
    if output_path is not None:
        matplotlib.use('Agg')  # Use non-GUI backend

    # convert vox_size from LIA to AS
    ras2vox = partial(apply_affine, np.linalg.inv(vox2ras)[1:, 1:])

    # scale contour data by vox_size to convert from AS to AS-aligned voxel space
    _split_contours = [] if split_contours is None else [ras2vox(sp.T).T for sp in split_contours]
    _midline_equi = np.zeros((0, 2)) if midline_equidistant is None else ras2vox(midline_equidistant)
    _levelpaths = [] if levelpaths is None else [ras2vox(lp) for lp in levelpaths]

    has_first_plot = not (len(_split_contours) == 0 and ac_coords_vox is None and pc_coords_vox is None)
    num_plots = 1 + int(has_first_plot)

    fig, ax = plt.subplots(1, num_plots, sharex=True, sharey=True, figsize=(15, 10))

    # NOTE: For all plots imshow shows y inverted
    current_plot = 0

    # This visualization uses voxel coordinates in fsaverage space...
    if has_first_plot:
        ax[current_plot].imshow(slice_or_slab[slice_or_slab.shape[0] // 2], cmap="gray")
        ax[current_plot].set_title(title)
    if _split_contours:
        for i, this_contour in enumerate(_split_contours):
            ax[current_plot].fill(this_contour[1, :], this_contour[0, :], color="steelblue", alpha=0.25)
            kwargs = {"color": "mediumblue", "linewidth": 0.7, "linestyle": "solid" if i != 0 else "dotted"}
            ax[current_plot].plot(this_contour[1, :], this_contour[0, :], **kwargs)
    if ac_coords_vox is not None:
        ax[current_plot].scatter(ac_coords_vox[1], ac_coords_vox[0], color="red", marker="x")
    if pc_coords_vox is not None:
        ax[current_plot].scatter(pc_coords_vox[1], pc_coords_vox[0], color="blue", marker="x")
    current_plot += int(has_first_plot)

    ax[current_plot].imshow(slice_or_slab[slice_or_slab.shape[0] // 2], cmap="gray")
    for this_path in _levelpaths:
        ax[current_plot].plot(this_path[:, 1], this_path[:, 0], color="brown", linewidth=0.8)
    ax[current_plot].set_title("Midline & Levelpaths")
    if _midline_equi.shape[0] > 0:
        ax[current_plot].plot(_midline_equi[:, 1], _midline_equi[:, 0], color="red")
    if _split_contours:
        reference_contour = _split_contours[0]
        ax[current_plot].plot(reference_contour[1, :], reference_contour[0, :], color="red", linewidth=0.5)

    padding = 30
    for a in ax.flatten():
        a.set_aspect("equal", adjustable="box")
        a.axis("off")
        if _split_contours:
            reference_contour = _split_contours[0]
            # get bounding box of contours
            a.set_xlim(reference_contour[1, :].min() - padding, reference_contour[1, :].max() + padding)
            a.set_ylim((reference_contour[0, :]).max() + padding, (reference_contour[0, :]).min() - padding)

    if output_path is None:
        return plt.show()
    for _output_path in (output_path if isinstance(output_path, (list, tuple)) else [output_path]):
        Path(_output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(_output_path, dpi=300, bbox_inches="tight")
    return None
