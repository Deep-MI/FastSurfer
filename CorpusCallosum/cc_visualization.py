import argparse
from pathlib import Path

import numpy as np

from CorpusCallosum.data.fsaverage_cc_template import load_fsaverage_cc_template
from CorpusCallosum.shape.cc_mesh import CC_Mesh


def options_parse() -> argparse.Namespace:
    """Parse command line arguments for the visualization pipeline."""
    parser = argparse.ArgumentParser(description="Visualize corpus callosum from template files.")
    parser.add_argument("--contours", type=str, required=False, help="Path to contours.txt file", default=None)
    parser.add_argument("--thickness", type=str, required=True, help="Path to thickness_values.txt file")
    parser.add_argument(
        "--measurement_points",
        type=str,
        required=True,
        help="Path to measurement points file containing the original vertex indices where thickness was measured",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files")
    parser.add_argument("--resolution", type=float, default=1.0, help="Resolution in mm for the mesh")
    parser.add_argument(
        "--smooth_iterations", type=int, default=1, help="Number of smoothing iterations to apply to the mesh"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="red_to_yellow",
        choices=["red_to_blue", "blue_to_red", "red_to_yellow", "yellow_to_red"],
        help="Colormap to use for thickness visualization",
    )
    parser.add_argument(
        "--color_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional fixed range for the colorbar (min max)",
    )
    parser.add_argument("--legend", type=str, default="Thickness (mm)", help="Legend for the colorbar")
    parser.add_argument("--twoD", action="store_true", help="Generate 2D visualization instead of 3D mesh")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    return args


def main(
    contours_path: str | Path | None,
    thickness_path: str | Path,
    measurement_points_path: str | Path,
    output_dir: str | Path,
    resolution: float = 1.0,
    smooth_iterations: int = 1,
    colormap: str = "red_to_yellow",
    color_range: tuple[float, float] | None = None,
    legend: str | None = None,
    twoD: bool = False,
) -> None:
    """Main function to visualize corpus callosum from template files.

    This function:
    1. Loads contours and thickness values from template files
    2. Creates a CC_Mesh object
    3. Generates and saves visualizations

    Args:
        contours_path: Path to contours.txt file
        thickness_path: Path to thickness_values.txt file
        measurement_points_path: Path to file containing the original vertex indices where thickness was measured
        output_dir: Directory for output files
        resolution: Resolution in mm for the mesh
        smooth_iterations: Number of smoothing iterations to apply to the mesh
        colormap: Which colormap to use. Options are:
        - "red_to_blue": Red -> Orange -> Grey -> Light Blue -> Blue
        - "blue_to_red": Blue -> Light Blue -> Grey -> Orange -> Red
        - "red_to_yellow": Red -> Yellow -> Light Blue -> Blue
        - "yellow_to_red": Yellow -> Light Blue -> Blue -> Red

        color_range: Optional tuple of (min, max) to set fixed color range for the colorbar
        twoD: If True, generate 2D visualization instead of 3D mesh
    """
    # Convert paths to Path objects
    contours_path = Path(contours_path) if contours_path is not None else None
    thickness_path = Path(thickness_path)
    measurement_points_path = Path(measurement_points_path)
    output_dir = Path(output_dir)

    # Load data and create mesh
    cc_mesh = CC_Mesh(num_slices=1)  # Will be resized when loading data

    if contours_path is not None:
        cc_mesh.load_contours(str(contours_path))
    else:
        cc_contour, anterior_endpoint_idx, posterior_endpoint_idx = load_fsaverage_cc_template()
        cc_mesh.contours[0] = np.stack(cc_contour).T
        cc_mesh.start_end_idx[0] = [anterior_endpoint_idx, posterior_endpoint_idx]

    cc_mesh.load_thickness_values(str(thickness_path), str(measurement_points_path))
    cc_mesh.set_resolution(resolution)

    if twoD:
        # cc_mesh.smooth_contour(contour_idx=0, window_size=5)
        cc_mesh.plot_cc_contour_with_levelsets(
            contour_idx=0, levelpaths=None, title=None, save_path=str(output_dir / "cc_thickness_2d.png"), colorbar=True
        )
    else:
        cc_mesh.fill_thickness_values()
        # Create and process mesh
        cc_mesh.create_mesh(smooth=smooth_iterations, closed=False)

        # Generate visualizations
        cc_mesh.plot_mesh(
            colormap=colormap,
            color_range=color_range,
            thickness_overlay=True,
            show_contours=False,
            show_mesh_edges=True,
            legend=legend,
        )
        cc_mesh.plot_mesh(str(output_dir / "cc_mesh.html"), thickness_overlay=True)

        cc_mesh.plot_cc_contour_with_levelsets(
            contour_idx=len(cc_mesh.contours) // 2, save_path=str(output_dir / "midslice_2d.png")
        )

        cc_mesh.to_fs_coordinates()
        cc_mesh.write_vtk(str(output_dir / "cc_mesh.vtk"))
        cc_mesh.write_fssurf(str(output_dir / "cc_mesh.fssurf"))
        cc_mesh.write_overlay(str(output_dir / "cc_mesh_overlay.curv"))
        cc_mesh.snap_cc_picture(str(output_dir / "cc_mesh_snap.png"))


if __name__ == "__main__":
    options = options_parse()
    main_args = {
        "contours_path": options.contours,
        "thickness_path": options.thickness,
        "measurement_points_path": options.measurement_points,
        "output_dir": options.output_dir,
        "resolution": options.resolution,
        "smooth_iterations": options.smooth_iterations,
        "colormap": options.colormap,
        "color_range": options.color_range,
        "legend": options.legend,
        "twoD": options.twoD,
    }
    main(**main_args)
