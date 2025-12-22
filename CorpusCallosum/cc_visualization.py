import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np

from CorpusCallosum.data.constants import FSAVERAGE_MIDDLE
from CorpusCallosum.data.fsaverage_cc_template import load_fsaverage_cc_template
from CorpusCallosum.shape.contour import CCContour
from CorpusCallosum.shape.mesh import CCMesh
from FastSurferCNN.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def make_parser() -> argparse.ArgumentParser:
    """Create a command line parser for the visualization pipeline."""
    parser = argparse.ArgumentParser(description="Visualize corpus callosum from template files.")
    parser.add_argument(
        "--template_dir",
        type=str,
        required=True,
        help=(
            "Path to a template directory containing per-slice files named "
            "thickness_values_<idx>.txt, and optionally contour_<idx>.txt "
            "and thickness_measurement_points_<idx>.txt. If contour_<idx>.txt "
            "and thickness_measurement_points_<idx>.txt are not provided, "
            "uses fsaverage template."
        ),
        metavar="TEMPLATE_DIR",
        default=None,
    )
    parser.add_argument("--output_dir", 
        type=str, 
        required=True, 
        help="Directory for output files. Writes: "
            "cc_mesh.html - Interactive 3D mesh visualization (HTML file) "
            "midslice_2d.png - 2D midslice visualization of the corpus callosum "
            "cc_mesh.vtk - VTK mesh file format "
            "cc_mesh.fssurf - FreeSurfer surface file "
            "cc_mesh_overlay.curv - FreeSurfer curvature overlay file "
            "cc_mesh_snap.png - Screenshot/snapshot of the 3D mesh (requires whippersnappy>=1.3.1)",
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose (pass twice for debug-output).",
    )
    return parser


def options_parse() -> argparse.Namespace:
    """Parse command line arguments for the pipeline."""
    parser = make_parser()
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args


def load_contours_from_template_dir(
    template_dir: Path, resolution: float, smoothing_window: int
) -> list[CCContour]:
    """Load all contours and thickness data from a template directory."""
    thickness_files = sorted(template_dir.glob("thickness_values_*.txt"))
    if not thickness_files:
        raise FileNotFoundError(
            f"No thickness files found in template directory {template_dir}. "
            "Expected files named thickness_values_<idx>.txt and "
            "optionally contour_<idx>.txt and thickness_measurement_points_<idx>.txt."
        )
    
    fsaverage_contour = None
    contours: list[CCContour] = []
    # First pass: collect all indices to determine the range
    indices = []
    for thickness_file in thickness_files:
        try:
            idx = int(thickness_file.stem.split("_")[-1])
            indices.append(idx)
        except ValueError:
            # skip files that do not follow the expected naming
            continue
    
    # Calculate z_positions centered around the middle slice
    num_slices = len(indices)
    middle_idx = num_slices // 2
    
    for i, thickness_file in enumerate(thickness_files):
        try:
            idx = int(thickness_file.stem.split("_")[-1])
        except ValueError:
            # skip files that do not follow the expected naming
            continue

        # Calculate z_position: use the index offset from middle, scaled by resolution
        z_position = (idx - indices[middle_idx]) * resolution
        
        contour_file = template_dir / f"contour_{idx}.txt"

        if not contour_file.exists():
            # get length of thickness values
            thickness_values = np.loadtxt(thickness_file, dtype=str)
            # get the non nan thickness values (excluding header), so we know how many points to sample
            num_thickness_values = np.sum(~np.isnan(np.array(thickness_values[1:],dtype=float)))
            if fsaverage_contour is None:
                fsaverage_contour = load_fsaverage_cc_template()
                # create measurement points (points = 2 x levelpaths) according to number of thickness values
                fsaverage_contour.create_levelpaths(num_points=num_thickness_values // 2, update_data=True)
            current_contour = fsaverage_contour.copy()
            current_contour.z_position = z_position
            current_contour.load_thickness_values(thickness_file)
            
        else:
            current_contour = CCContour.from_contour_file(contour_file, thickness_file, z_position=z_position)
        
        current_contour.fill_thickness_values()
        contours.append(current_contour)

    if not contours:
        raise ValueError(f"No valid contours could be loaded from {template_dir}")
    return contours


def main(
    template_dir: str | Path,
    output_dir: str | Path,
    resolution: float = 1.0,
    smoothing_window: int = 5,
    colormap: str = "red_to_yellow",
    color_range: tuple[float, float] | None = None,
    legend: str | None = None,
    twoD: bool = False,
) -> Literal[0] | str:
    """Visualize corpus callosum templates in 2D or 3D."""
    output_dir = Path(output_dir)
    color_range = tuple(color_range) if color_range is not None else None

    contours = load_contours_from_template_dir(
        Path(template_dir), resolution=resolution, smoothing_window=smoothing_window,
    )

    # 2D visualization
    mid_contour = contours[len(contours) // 2]

    # for now, we only support thickness visualization, this is preparing to plot also p-values and icc values
    mode = "thickness"
    logger.info(f"Writing output to {output_dir / 'cc_thickness_2d.png'}")

    if mode == "thickness":
        raw_thickness_values = mid_contour.thickness_values[~np.isnan(mid_contour.thickness_values)]
        # values are duplicated because they have two measurement points per levelpath
        raw_thickness_values = raw_thickness_values[len(raw_thickness_values) // 2:] 
    mid_contour.plot_contour_colorfill(
        plot_values=raw_thickness_values,
        title=None,
        save_path=str(output_dir / "cc_thickness_2d.png"),
        colorbar=True,
        mode=mode
    )
    if twoD:
        return 0

    # 3D visualization
    # FIXME: This function would need contours[i].z_position to be properly initialized!
    cc_mesh = CCMesh.from_contours(contours, smooth=0)

    plot_kwargs = dict(
        colormap=colormap,
        color_range=color_range,
        thickness_overlay=True,
        legend=legend or "",
    )
    cc_mesh.plot_mesh(**plot_kwargs)
    cc_mesh.plot_mesh(output_path=str(output_dir / "cc_mesh.html"), **plot_kwargs)

    #FIXME: needs to be adapted to new interface of CCMesh.to_fs_coordinates / to_vox_coordinates
    cc_mesh = cc_mesh.to_vox_coordinates(lr_offset=FSAVERAGE_MIDDLE / resolution)
    logger.info(f"Writing vtk file to {output_dir / 'cc_mesh.vtk'}")
    cc_mesh.write_vtk(str(output_dir / "cc_mesh.vtk"))
    logger.info(f"Writing freesurfer surface file to {output_dir / 'cc_mesh.fssurf'}")
    cc_mesh.write_fssurf(str(output_dir / "cc_mesh.fssurf"))
    logger.info(f"Writing freesurfer overlay file to {output_dir / 'cc_mesh_overlay.curv'}")
    cc_mesh.write_morph_data(str(output_dir / "cc_mesh_overlay.curv"))
    try:
        cc_mesh.snap_cc_picture(str(output_dir / "cc_mesh_snap.png"))
        logger.info(f"Writing 3D snapshot image to {output_dir / 'cc_mesh_snap.png'}")
    except RuntimeError:
        logger.warning("The cc_visualization script requires whippersnappy>=1.3.1 to makes screenshots, install with "
                "`pip install whippersnappy>=1.3.1` !")
    return 0

if __name__ == "__main__":
    options = options_parse()

    # Set up logging if verbose mode is enabled
    setup_logging(None, options.verbose)  # Log to stdout only

    sys.exit(main(
        template_dir=options.template_dir,
        output_dir=options.output_dir,
        resolution=options.resolution,
        smoothing_window=options.smoothing_window,
        colormap=options.colormap,
        color_range=options.color_range,
        legend=options.legend,
        twoD=options.twoD,
    ))
