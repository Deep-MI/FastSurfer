import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np

from CorpusCallosum.data.constants import FSAVERAGE_DATA_PATH
from CorpusCallosum.data.fsaverage_cc_template import load_fsaverage_cc_template
from CorpusCallosum.data.read_write import load_fsaverage_data
from CorpusCallosum.shape.cc_mesh import CCMesh


def make_parser() -> argparse.ArgumentParser:
    """Create a command line parser for the visualization pipeline."""
    parser = argparse.ArgumentParser(description="Visualize corpus callosum from template files.")
    parser.add_argument(
        "--contours", 
        type=str, 
        required=False, 
        help="Path to contours.txt file if not provided, uses fsaverage template.", 
        metavar="CONTOURS_PATH", 
        default=None
    )
    parser.add_argument(
        "--thickness", 
        type=str, 
        required=True, 
        help="Path to thickness_values.txt file.",
        metavar="THICKNESS_VALUES_PATH"
    )
    parser.add_argument(
        "--measurement_points",
        type=str,
        required=True,
        help="Path to measurement points file containing the original vertex indices where thickness was measured.",
    )
    parser.add_argument("--output_dir", 
        type=str, 
        required=True, 
        help="Directory for output files. Writes: \\\
            cc_mesh.html - Interactive 3D mesh visualization (HTML file) \\\
            midslice_2d.png - 2D midslice visualization of the corpus callosum \\\
            cc_mesh.vtk - VTK mesh file format \\\
            cc_mesh.fssurf - FreeSurfer surface file \\\
            cc_mesh_overlay.curv - FreeSurfer curvature overlay file \\\
            cc_mesh_snap.png - Screenshot/snapshot of the 3D mesh (requires whippersnappy>=1.3.1)",
        metavar="OUTPUT_DIR"
    )
    parser.add_argument(
        "--resolution", 
        type=float, 
        default=1.0, 
        help="Resolution in mm for the mesh.",
        metavar="RESOLUTION"
    )
    parser.add_argument(
        "--smoothing_window", 
        type=int, 
        default=5, 
        help="Window size for smoothing the contour.",
        metavar="SMOOTHING_WINDOW"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="red_to_yellow",
        choices=["red_to_blue", "blue_to_red", "red_to_yellow", "yellow_to_red"],
        help="Colormap to use for thickness visualization, lower to higher values.",
    )
    parser.add_argument(
        "--color_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        required=False,
        help="Specify the range for the colorbar (2 values: min max). Defaults to automatic choice. \
              (e.g. --color_range 0 10).",
    )
    parser.add_argument(
        "--legend", 
        type=str, 
        default="Thickness (mm)", 
        help="Legend for the colorbar.",
        metavar="LEGEND")
    parser.add_argument(
        "--twoD", 
        action="store_true", 
        help="Generate 2D visualization instead of 3D mesh.",
    )

    return parser

def options_parse() -> argparse.Namespace:
    """Parse command line arguments for the pipeline."""
    args = make_parser().parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    return args


def main(
    contours_path: str | Path | None,
    thickness_path: str | Path,
    measurement_points_path: str | Path,
    output_dir: str | Path,
    resolution: float = 1.0,
    smoothing_window: int = 5,
    colormap: str = "red_to_yellow",
    color_range: tuple[float, float] | None = None,
    legend: str | None = None,
    twoD: bool = False,
) -> Literal[0] | str:
    """Main function to visualize corpus callosum from template files.

    This function loads contours and thickness values from template files,
    creates a CC_Mesh object, and generates visualizations.

    Parameters
    ----------
    contours_path : str or Path or None
        Path to contours.txt file.
    thickness_path : str or Path
        Path to thickness_values.txt file.
    measurement_points_path : str or Path
        Path to file containing original vertex indices where thickness was measured.
    output_dir : str or Path
        Directory for output files.
    resolution : float, optional
        Resolution in mm for the mesh, by default 1.0.
    smoothing_window : int, optional
        Window size for smoothing the contour, by default 5.
    colormap : str, optional
        Colormap to use for visualization, by default "red_to_yellow".
        Options:
        - "red_to_blue": Red -> Orange -> Grey -> Light Blue -> Blue
        - "blue_to_red": Blue -> Light Blue -> Grey -> Orange -> Red
        - "red_to_yellow": Red -> Yellow -> Light Blue -> Blue
        - "yellow_to_red": Yellow -> Light Blue -> Blue -> Red
    color_range : tuple[float, float], optional
        Fixed range (min, max) for the colorbar, by default None.
    legend : str, optional
        Legend for the colorbar, by default None.
    twoD : bool, optional
        If True, generate 2D visualization instead of 3D mesh, by default False.
    """
    # Convert paths to Path objects
    contours_path = Path(contours_path) if contours_path is not None else None
    thickness_path = Path(thickness_path)
    measurement_points_path = Path(measurement_points_path)
    output_dir = Path(output_dir)

    # Load data and create mesh
    cc_mesh = CCMesh(num_slices=1)  # Will be resized when loading data

    _, _, vox2ras_tkr = load_fsaverage_data(FSAVERAGE_DATA_PATH)

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
        cc_mesh.create_mesh(smooth=smoothing_window, closed=False)

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

        cc_mesh.to_fs_coordinates(vox_size=[resolution, resolution, resolution], vox2ras_tkr=vox2ras_tkr)
        cc_mesh.write_vtk(str(output_dir / "cc_mesh.vtk"))
        cc_mesh.write_fssurf(str(output_dir / "cc_mesh.fssurf"))
        cc_mesh.write_overlay(str(output_dir / "cc_mesh_overlay.curv"))
        try:
            cc_mesh.snap_cc_picture(str(output_dir / "cc_mesh_snap.png"))
        except RuntimeError:
            return ("The cc_visualization script requires whippersnappy>=1.3.1 to makes screenshots, install with "
                    "`pip install whippersnappy>=1.3.1` !")
    return 0

if __name__ == "__main__":
    options = make_parser().parse_args()
    sys.exit(main(
        contours_path=options.contours,
        thickness_path=options.thickness,
        measurement_points_path=options.measurement_points,
        output_dir=options.output_dir,
        resolution=options.resolution,
        smoothing_window=options.smoothing_window,
        colormap=options.colormap,
        color_range=options.color_range,
        legend=options.legend,
        twoD=options.twoD,
    ))
