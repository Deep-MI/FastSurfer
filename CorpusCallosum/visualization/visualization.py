from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_standardized_space(ax_row, vol, ac_coords, pc_coords):
    """Plot standardized space visualization across three views.

    Args:
        ax_row: Row of axes to plot on (should be length 3)
        vol: Volume data to visualize
        ac_coords: AC coordinates in standardized space
        pc_coords: PC coordinates in standardized space
    """
    ax_row[0].set_title("Standardized")

    # Axial view
    ax_row[0].scatter(ac_coords[2], ac_coords[1], color="red", marker="x")
    ax_row[0].scatter(pc_coords[2], pc_coords[1], color="blue", marker="x")
    ax_row[0].imshow(vol[vol.shape[0] // 2], cmap="gray")

    # Sagittal view
    ax_row[1].scatter(ac_coords[2], ac_coords[0], color="red", marker="x")
    ax_row[1].scatter(pc_coords[2], pc_coords[0], color="blue", marker="x")
    ax_row[1].imshow(vol[:, vol.shape[1] // 2], cmap="gray")

    # Coronal view
    ax_row[2].scatter(ac_coords[1], ac_coords[0], color="red", marker="x")
    ax_row[2].scatter(pc_coords[1], pc_coords[0], color="blue", marker="x")
    ax_row[2].imshow(vol[:, :, vol.shape[2] // 2], cmap="gray")


def visualize_coordinate_spaces(
    orig,
    upright,
    standardized,
    ac_coords_orig,
    pc_coords_orig,
    ac_coords_3d,
    pc_coords_3d,
    ac_coords_standardized,
    pc_coords_standardized,
    output_dir,
):
    """
    Visualize the AC and PC coordinates in different coordinate spaces for testing/debugging.

    Args:
        orig: Original image volume
        vol: Volume in fsaverage space
        vol2: Volume after nodding correction
        vol3: Volume after translation
        ac_coords_*: AC coordinates in different spaces
        pc_coords_*: PC coordinates in different spaces
        output_dir: Directory to save visualization
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

    plt.savefig(Path(output_dir) / "ac_pc_spaces.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_contours(
    transformed: np.ndarray,
    split_contours: list[np.ndarray],
    split_contours_hofer_frahm: list[np.ndarray],
    midline_equidistant: np.ndarray,
    levelpaths: list[np.ndarray],
    output_path: str,
    ac_coords: np.ndarray,
    pc_coords: np.ndarray,
    vox_size: float,
    title: str = None,
) -> None:
    """Plots corpus callosum contours and segmentations.

    Creates a figure with three subplots showing:
    1. Midline-based subsegmentation
    2. Hofer-Frahm segmentation scheme
    3. Midline and levelpaths visualization

    Args:
        transformed: The transformed brain image array
        split_contours: List of contour arrays for midline-based segmentation
        split_contours_hofer_frahm: List of contour arrays for Hofer-Frahm segmentation
        midline_equidistant: Array of midline points
        levelpaths: List of levelpath arrays
        output_dir: Directory to save the output plot
        ac_coords: Anterior commissure coordinates
        pc_coords: Posterior commissure coordinates
    """

    # scale contour data by vox_size
    split_contours = (
        [split_contour * vox_size for split_contour in split_contours] if split_contours is not None else None
    )
    split_contours_hofer_frahm = (
        [split_contour * vox_size for split_contour in split_contours_hofer_frahm]
        if split_contours_hofer_frahm is not None
        else None
    )
    midline_equidistant = midline_equidistant * vox_size
    levelpaths = [levelpath * vox_size for levelpath in levelpaths]

    NO_PLOTS = 1
    if split_contours is not None:
        NO_PLOTS += 1
    if split_contours_hofer_frahm is not None:
        NO_PLOTS += 1

    fig, ax = plt.subplots(1, NO_PLOTS, sharex=True, sharey=True, figsize=(15, 10))

    PLT_NUM = 0

    if split_contours is not None:
        ax[PLT_NUM].imshow(transformed[transformed.shape[0] // 2], cmap="gray")
        # ax[0].imshow(cc_mask, cmap='autumn')
        ax[PLT_NUM].set_title(title)
        for i in range(len(split_contours)):
            ax[PLT_NUM].fill(split_contours[i][0, :], -split_contours[i][1, :], color="steelblue", alpha=0.25)
            ax[PLT_NUM].plot(
                split_contours[i][0, :], -split_contours[i][1, :], color="mediumblue", linestyle="dotted", linewidth=0.7
            )
        ax[PLT_NUM].plot(split_contours[0][0, :], -split_contours[0][1, :], color="mediumblue", linewidth=0.7)
        ax[PLT_NUM].scatter(ac_coords[1], ac_coords[0], color="red", marker="x")
        ax[PLT_NUM].scatter(pc_coords[1], pc_coords[0], color="blue", marker="x")
        PLT_NUM += 1

    if split_contours_hofer_frahm is not None:
        ax[PLT_NUM].imshow(transformed[transformed.shape[0] // 2], cmap="gray")
        # ax[1].imshow(cc_mask, cmap='autumn')
        ax[PLT_NUM].set_title("Hofer-Frahm Jaenecke")
        for i in range(len(split_contours_hofer_frahm)):
            ax[PLT_NUM].fill(
                split_contours_hofer_frahm[i][0, :], -split_contours_hofer_frahm[i][1, :], color="steelblue", alpha=0.25
            )
            ax[PLT_NUM].plot(
                [split_contours_hofer_frahm[i][0, 0], split_contours_hofer_frahm[i][0, -1]],
                [-split_contours_hofer_frahm[i][1, 0], -split_contours_hofer_frahm[i][1, -1]],
                color="mediumblue",
                linestyle="dotted",
                linewidth=0.7,
            )
        ax[PLT_NUM].plot(
            split_contours_hofer_frahm[0][0, :], -split_contours_hofer_frahm[0][1, :], color="mediumblue", linewidth=0.7
        )
        ax[PLT_NUM].scatter(ac_coords[1], ac_coords[0], color="red", marker="x")
        ax[PLT_NUM].scatter(pc_coords[1], pc_coords[0], color="blue", marker="x")
        PLT_NUM += 1

    reference_contour = split_contours[0] if split_contours is not None else split_contours_hofer_frahm[0]

    ax[PLT_NUM].imshow(transformed[transformed.shape[0] // 2], cmap="gray")
    # ax[2].imshow(cc_mask, cmap='autumn')
    for i in range(len(levelpaths)):
        ax[PLT_NUM].plot(levelpaths[i][:, 0], -levelpaths[i][:, 1], color="brown", linewidth=0.8)
    ax[PLT_NUM].set_title("Midline & Levelpaths")
    ax[PLT_NUM].plot(midline_equidistant[:, 0], -midline_equidistant[:, 1], color="red")
    ax[PLT_NUM].plot(reference_contour[0, :], -reference_contour[1, :], color="red", linewidth=0.5)

    for a in ax.flatten():
        a.set_aspect("equal", adjustable="box")
        a.axis("off")

    # get bounding box of countours
    padding = 30
    ax[0].set_xlim(reference_contour[0, :].min() - padding, reference_contour[0, :].max() + padding)
    ax[0].set_ylim((-reference_contour[1, :]).max() + padding, (-reference_contour[1, :]).min() - padding)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # plt.show()


def plot_midplane(grid_orig, orig):
    """
    Creates a 3D visualization of grid points in original image space.

    Args:
        grid_orig: Grid points in original space
        orig: Original image for dimension reference
    """
    # Create a figure showing grid points in original space

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot every 10th point to avoid overcrowding
    sample_idx = np.arange(0, grid_orig.shape[1], 40)
    ax.scatter(
        grid_orig[0, sample_idx], grid_orig[1, sample_idx], grid_orig[2, sample_idx], c="r", alpha=0.1, marker="."
    )

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
