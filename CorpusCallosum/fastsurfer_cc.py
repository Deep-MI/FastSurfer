import argparse
import json
from pathlib import Path

# import warnings warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
import nibabel as nib
import numpy as np
import torch

import FastSurferCNN.utils.logging as logging
from CorpusCallosum.data.constants import (
    CC_LABEL,
    FSAVERAGE_CENTROIDS_PATH,
    FSAVERAGE_DATA_PATH,
    FSAVERAGE_MIDDLE,
    STANDARD_OUTPUT_PATHS,
    WEIGHTS_PATH,
)
from CorpusCallosum.data.read_write import (
    convert_numpy_to_json_serializable,
    get_centroids_from_nib,
    load_fsaverage_centroids,
    load_fsaverage_data,
    run_in_background,
    save_nifti_background,
)
from CorpusCallosum.localization import localization_inference
from CorpusCallosum.registration.mapping_helpers import (
    apply_transform_to_pt,
    apply_transform_to_volume,
    get_mapping_to_standard_space,
    interpolate_midplane,
    map_softlabels_to_orig,
)
from CorpusCallosum.segmentation import segmentation_inference, segmentation_postprocessing
from CorpusCallosum.shape.cc_postprocessing import (
    check_area_changes,
    create_visualization,
    make_subdivision_mask,
    process_slices,
)
from FastSurferCNN.data_loader.conform import is_conform
from recon_surf import lta
from recon_surf.align_points import find_rigid

logger = logging.get_logger(__name__)


def options_parse() -> argparse.Namespace:
    """Parse command line arguments for the pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_mri",
        type=str,
        required=False,
        help="Input MRI file path. If not provided, defaults to subject_dir/mri/orig.mgz",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even when CUDA is available",
    )
    parser.add_argument(
        "--aseg",
        type=str,
        required=False,
        help="Input segmentation file path. If not provided, defaults to subject_dir/mri/aparc.DKTatlas+aseg.deep.mgz",
    )
    parser.add_argument(
        "--subject_dir",
        type=str,
        required=False,
        help="Subject directory containing standard FreeSurfer structure. "
        "Required if --in_mri and --aseg are not both provided.",
        default=None,
    )
    parser.add_argument("--debug_output_dir", type=str, required=False, default=None,
                        help="Directory for debug output (default: subject_dir/qc_snapshots)")
    parser.add_argument(
        "--num_thickness_points", type=int, default=100, help="Number of points for thickness estimation."
    )
    parser.add_argument(
        "--subdivisions",
        type=float,
        nargs="+",
        default=[1/6, 1/2, 2/3, 3/4],
        help="List of subdivision fractions for the corpus callosum subsegmentation.",
    )
    parser.add_argument(
        "--subdivision_method",
        type=str,
        default="shape",
        help="Method for contour subdivision. \
                        Options: shape (Intercallosal subdivision perpendicular to intercallosal line), vertical \
                        (orthogonal to the most anterior and posterior points in the AC/PC standardized CC contour), \
                        angular (subdivision based on equally spaced angles, as proposed by Hampel and colleagues), \
                        eigenvector (primary direction, same as FreeSurfers mri_cc)",
        choices=["shape", "vertical", "angular", "eigenvector"],
    )
    parser.add_argument(
        "--contour_smoothing",
        type=float,
        default=5,
        help="Window size for smoothing during contour detection. Default is 5, higher values mean a smoother"
        "outline, at the cost of precision.",
    )
    parser.add_argument(
        "--slice_selection",
        type=str,
        default="all",
        help="Which slices to process. Options: 'middle' (default), 'all', or a specific slice number.",
    )
    parser.add_argument(
        "--upright_volume_path",
        type=str,
        help=f"Path for upright volume output (default: subject_dir/{STANDARD_OUTPUT_PATHS['upright_volume']})",
        default=None,
    )
    parser.add_argument(
        "--segmentation_path",
        type=str,
        help=f"Path for segmentation output (default: subject_dir/{STANDARD_OUTPUT_PATHS['segmentation']})",
        default=None,
    )
    parser.add_argument(
        "--postproc_results_path",
        type=str,
        help=f"Path for postprocessing results (default: subject_dir/{STANDARD_OUTPUT_PATHS['postproc_results']})",
        default=None,
    )
    parser.add_argument(
        "--cc_markers_path",
        type=str,
        help=f"Path for CC markers output (default: subject_dir/{STANDARD_OUTPUT_PATHS['cc_markers']})",
        default=None,
    )
    parser.add_argument(
        "--upright_lta_path",
        type=str,
        help=f"Path for upright LTA transform (default: subject_dir/{STANDARD_OUTPUT_PATHS['upright_lta']})",
        default=None,
    )
    parser.add_argument(
        "--orient_volume_lta_path",
        type=str,
        help="Path for orientation volume LTA transform "
             f"(default: subject_dir/{STANDARD_OUTPUT_PATHS['orient_volume_lta']})",
        default=None,
    )
    parser.add_argument(
        "--orig_space_segmentation_path",
        type=str,
        help="Path for segmentation in original space "
             f"(default: subject_dir/{STANDARD_OUTPUT_PATHS['orig_space_segmentation']})",
        default=None,
    )
    parser.add_argument(
        "--debug_image_path",
        type=str,
        help=f"Path for debug visualization image (default: subject_dir/{STANDARD_OUTPUT_PATHS['debug_image']})",
        default=None,
    )
    parser.add_argument(
        "--save_template",
        type=str,
        help="Directory path where to save contours.txt and thickness_values.txt files",
        default=None,
    )
    parser.add_argument(
        "--thickness_image_path",
        type=str,
        help=f"Path for thickness image (default: subject_dir/{STANDARD_OUTPUT_PATHS['thickness_image']})",
        default=None,
    )
    parser.add_argument(
        "--surf_file_path",
        type=str,
        help=f"Path for surf file (default: subject_dir/{STANDARD_OUTPUT_PATHS['surf_file']})",
        default=None,
    )
    parser.add_argument(
        "--overlay_file_path",
        type=str,
        help=f"Path for overlay file (default: subject_dir/{STANDARD_OUTPUT_PATHS['overlay_file']})",
        default=None,
    )
    parser.add_argument(
        "--cc_html_path",
        type=str,
        help=f"Path for CC HTML file (default: subject_dir/{STANDARD_OUTPUT_PATHS['cc_html']})",
        default=None,
    )
    parser.add_argument(
        "--vtk_file_path",
        type=str,
        help=f"Path for vtk file (default: subject_dir/{STANDARD_OUTPUT_PATHS['vtk_file']})",
        default=None,
    )
    parser.add_argument(
        "--softlabels_cc_path",
        type=str,
        help=f"Path for cc softlabels (default: subject_dir/{STANDARD_OUTPUT_PATHS['softlabels_cc']})",
        default=None,
    )
    parser.add_argument(
        "--softlabels_fn_path",
        type=str,
        help=f"Path for fornix softlabels (default: subject_dir/{STANDARD_OUTPUT_PATHS['softlabels_fn']})",
        default=None,
    )
    parser.add_argument(
        "--softlabels_background_path",
        type=str,
        help=f"Path for background softlabels (default: subject_dir/{STANDARD_OUTPUT_PATHS['softlabels_background']})",
        default=None,
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (shows output paths)", default=False)


    args = parser.parse_args()

    # Validation logic: either subject_dir OR both in_mri and aseg must be provided
    if not args.subject_dir and (not args.in_mri or not args.aseg):
        parser.error("You must specify either --subject_dir OR both --in_mri and --aseg arguments.")

    # If subject_dir is provided, set default paths for missing arguments
    if args.subject_dir:
        subject_dir_path = Path(args.subject_dir)

        # Create standard FreeSurfer subdirectories
        (subject_dir_path / "mri").mkdir(parents=True, exist_ok=True)
        (subject_dir_path / "stats").mkdir(parents=True, exist_ok=True)
        (subject_dir_path / "transforms").mkdir(parents=True, exist_ok=True)

        if not args.in_mri:
            args.in_mri = str(subject_dir_path / "mri" / "orig.mgz")

        if not args.aseg:
            args.aseg = str(subject_dir_path / "mri" / "aparc.DKTatlas+aseg.deep.mgz")

        # Set default output paths if not provided
        for key, value in STANDARD_OUTPUT_PATHS.items():
            if not getattr(args, f"{key}_path") and value is not None:
                setattr(args, f"{key}_path", str(subject_dir_path / value))

        # Set output_dir to subject_dir
        args.output_dir = str(subject_dir_path)

    # Create parent directories for all output paths
    for path_name in STANDARD_OUTPUT_PATHS.keys():
        path = getattr(args, f"{path_name}_path")
        if path is not None:
            Path(path).parent.mkdir(parents=False, exist_ok=True)

    return args


def centroid_registration(aseg_nib: nib.Nifti1Image, verbose: bool = False) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, nib.Nifti1Header, np.ndarray
]:
    """Perform centroid-based registration between subject and fsaverage space.

    Computes a rigid transformation between the subject's segmentation and fsaverage space
    by aligning centroids of corresponding anatomical structures.

    Parameters
    ----------
    aseg_nib : nibabel.Nifti1Image
        Subject's segmentation image.
    verbose : bool, optional
        Whether to print progress information, by default False.

    Returns
    -------
    orig_fsaverage_vox2vox : np.ndarray
        Transformation matrix from original to fsaverage voxel space.
    orig_fsaverage_ras2ras : np.ndarray
        Transformation matrix from original to fsaverage RAS space.
    fsaverage_hires_affine : np.ndarray
        High-resolution fsaverage affine matrix.
    fsaverage_header : nibabel.Nifti1Header
        FSAverage header fields for LTA writing.
    vox2ras_tkr : np.ndarray
        Voxel to RAS tkr-space transformation matrix.

    Notes
    -----
    The function uses pre-computed fsaverage centroids and data from static files
    to perform the registration. It matches corresponding anatomical structures
    between the subject's segmentation and fsaverage space.
    """
    if verbose:
        print("Centroid registration")

    # Load pre-computed fsaverage centroids and data from static files
    centroids_dst = load_fsaverage_centroids(FSAVERAGE_CENTROIDS_PATH)
    fsaverage_affine, fsaverage_header, vox2ras_tkr = load_fsaverage_data(FSAVERAGE_DATA_PATH)

    centroids_mov, ids_not_found = get_centroids_from_nib(aseg_nib, label_ids=list(centroids_dst.keys()))

    # delete not found labels from centroids_mov
    for id in ids_not_found:
        del centroids_dst[id]

    centroids_mov = np.array(list(centroids_mov.values())).T
    centroids_dst = np.array(list(centroids_dst.values())).T

    orig_fsaverage_ras2ras = find_rigid(p_mov=centroids_mov.T, p_dst=centroids_dst.T)

    # make affine that increases resolution to orig resolution
    resolution_orig = aseg_nib.header.get_zooms()[0]
    resolution_trans = np.eye(4)
    resolution_trans[0, 0] = resolution_orig
    resolution_trans[1, 1] = resolution_orig
    resolution_trans[2, 2] = resolution_orig

    orig_fsaverage_vox2vox = (
        np.linalg.inv(resolution_trans @ fsaverage_affine) @ orig_fsaverage_ras2ras @ aseg_nib.affine
    )
    fsaverage_hires_affine = resolution_trans @ fsaverage_affine

    return orig_fsaverage_vox2vox, orig_fsaverage_ras2ras, fsaverage_hires_affine, fsaverage_header, vox2ras_tkr


def localize_ac_pc(
    midslices: np.ndarray,
    aseg_nib: "nib.Nifti1Image",
    orig_fsaverage_vox2vox: np.ndarray,
    model_localization: "torch.nn.Module",
    slices_to_analyze: int
) -> tuple[np.ndarray, np.ndarray]:
    """Localize anterior and posterior commissure points in the brain.

    Uses a trained model to detect AC and PC points in mid-sagittal slices,
    using the third ventricle as an anatomical reference.

    Parameters
    ----------
    midslices : np.ndarray
        Array of mid-sagittal slices.
    aseg_nib : nibabel.Nifti1Image
        Subject's segmentation image.
    orig_fsaverage_vox2vox : np.ndarray
        Transformation matrix to fsaverage space.
    model_localization : torch.nn.Module
        Trained model for AC-PC detection.
    slices_to_analyze : int
        Number of slices to process.

    Returns
    -------
    ac_coords : np.ndarray
        Coordinates of the anterior commissure.
    pc_coords : np.ndarray
        Coordinates of the posterior commissure.
    """

    # get center of third ventricle from aseg and map to fsaverage space
    third_ventricle_mask = aseg_nib.get_fdata() == 4
    third_ventricle_center = np.argwhere(third_ventricle_mask).mean(axis=0)
    third_ventricle_center_vox = apply_transform_to_pt(third_ventricle_center, orig_fsaverage_vox2vox, inv=False)

    # get 5 mm of slices output with 3 slices per inference
    midslices_middle = midslices.shape[0] // 2
    middle_slices_localization = midslices[
        midslices_middle - slices_to_analyze // 2 - 1 : midslices_middle + slices_to_analyze // 2 + 2
    ]
    ac_coords, pc_coords = localization_inference.run_inference_on_slice(
        model_localization, middle_slices_localization, third_ventricle_center_vox[1:]
    )

    return ac_coords, pc_coords


def segment_cc(
    midslices: np.ndarray,
    ac_coords: np.ndarray,
    pc_coords: np.ndarray,
    aseg_nib: "nib.Nifti1Image",
    model_segmentation: "torch.nn.Module",
    slices_to_analyze: int
) -> tuple[np.ndarray, np.ndarray]:
    """Segment the corpus callosum using a trained model.

    Performs corpus callosum segmentation on mid-sagittal slices using a trained model,
    with AC-PC points as anatomical references. Includes post-processing to clean 
    the segmentation.

    Parameters
    ----------
    midslices : np.ndarray
        Array of mid-sagittal slices.
    ac_coords : np.ndarray
        Anterior commissure coordinates.
    pc_coords : np.ndarray
        Posterior commissure coordinates.
    aseg_nib : nibabel.Nifti1Image
        Subject's segmentation image.
    model_segmentation : torch.nn.Module
        Trained model for CC segmentation.
    slices_to_analyze : int
        Number of slices to process.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - segmentation : Binary segmentation of the corpus callosum.
        - outputs_soft : Soft segmentation probabilities.
    """
    # get 5 mm of slices output with 9 slices per inference
    midslices_middle = midslices.shape[0] // 2
    middle_slices_segmentation = midslices[
        midslices_middle - slices_to_analyze // 2 - 4 : midslices_middle + slices_to_analyze // 2 + 5
    ]
    segmentation, inputs, outputs_avg, outputs_soft = segmentation_inference.run_inference_on_slice(
        model_segmentation,
        middle_slices_segmentation,
        AC_center=ac_coords,
        PC_center=pc_coords,
        voxel_size=aseg_nib.header.get_zooms()[0],
    )

    pre_clean_segmentation = segmentation.copy()
    segmentation, cc_volume_mask = segmentation_postprocessing.clean_cc_segmentation(segmentation)

    # print a warning if the cc_volume_mask touches the edge of the segmentation
    if (
        np.any(cc_volume_mask[:, 0, :])
        or np.any(cc_volume_mask[:, -1, :])
        or np.any(cc_volume_mask[:, :, 0])
        or np.any(cc_volume_mask[:, :, -1])
    ):
        print("Warning: CC voume mask touches the edge of the segmentation field-of-view, CC might be truncated")

    # get voxels that were removed during cleaning
    removed_voxels = pre_clean_segmentation != segmentation
    outputs_soft[removed_voxels, 1] = 0

    return segmentation, outputs_soft




def main(
    in_mri_path: str | Path,
    aseg_path: str | Path,
    output_dir: str | Path,
    slice_selection: str = "middle",
    debug_output_dir: str | Path = None,
    verbose: bool = False,
    num_thickness_points: int = 100,
    subdivisions: list[float] | None = None,
    subdivision_method: str = "shape",
    contour_smoothing: float = 5,
    save_template: str | Path | None = None,
    cpu: bool = False,
    upright_volume_path: str | Path = None,
    segmentation_path: str | Path = None,
    postproc_results_path: str | Path = None,
    cc_markers_path: str | Path = None,
    upright_lta_path: str | Path = None,
    orient_volume_lta_path: str | Path = None,
    surf_file_path: str | Path = None,
    overlay_file_path: str | Path = None,
    cc_html_path: str | Path = None,
    vtk_file_path: str | Path = None,
    orig_space_segmentation_path: str | Path = None,
    debug_image_path: str | Path = None,
    thickness_image_path: str | Path = None,
    softlabels_cc_path: str | Path = None,
    softlabels_fn_path: str | Path = None,
    softlabels_background_path: str | Path = None,
) -> None:
    """Main pipeline function for corpus callosum analysis.

    This function performs the complete corpus callosum analysis pipeline including
    registration, landmark detection, segmentation, and morphometry analysis.

    Parameters
    ----------
    in_mri_path : str or Path
        Path to input MRI file.
    aseg_path : str or Path
        Path to input segmentation file.
    output_dir : str or Path
        Directory for output files.
    slice_selection : str, optional
        Which slices to process ('middle', 'all', or specific slice number), by default 'middle'.
    debug_output_dir : str or Path, optional
        Directory for debug outputs, by default None.
    verbose : bool, optional
        Flag for verbose output, by default False.
    num_thickness_points : int, optional
        Number of points for thickness estimation, by default 100.
    subdivisions : list[float], optional
        List of subdivision fractions for CC subsegmentation, by default None.
    subdivision_method : str, optional
        Method for contour subdivision ('shape', 'vertical', 'angular', 'eigenvector'), by default 'shape'.
    contour_smoothing : float, optional
        Gaussian sigma for smoothing during contour detection, by default 5.
    save_template : str or Path, optional
        Directory path where to save contours.txt and thickness_values.txt files, by default None.
    cpu : bool, optional
        Force CPU usage even when CUDA is available, by default False.
    upright_volume_path : str or Path, optional
        Path to save upright volume, by default None.
    segmentation_path : str or Path, optional
        Path to save segmentation, by default None.
    postproc_results_path : str or Path, optional
        Path to save post-processing results, by default None.
    cc_markers_path : str or Path, optional
        Path to save CC markers, by default None.
    upright_lta_path : str or Path, optional
        Path to save upright LTA transform, by default None.
    orient_volume_lta_path : str or Path, optional
        Path to save orientation transform, by default None.
    surf_file_path : str or Path, optional
        Path to save surface file, by default None.
    overlay_file_path : str or Path, optional
        Path to save overlay file, by default None.
    cc_html_path : str or Path, optional
        Path to save HTML visualization, by default None.
    vtk_file_path : str or Path, optional
        Path to save VTK file, by default None.
    orig_space_segmentation_path : str or Path, optional
        Path to save segmentation in original space, by default None.
    debug_image_path : str or Path, optional
        Path to save debug images, by default None.
    thickness_image_path : str or Path, optional
        Path to save thickness visualization, by default None.
    softlabels_cc_path : str or Path, optional
        Path to save CC soft labels, by default None.
    softlabels_fn_path : str or Path, optional
        Path to save fornix soft labels, by default None.
    softlabels_background_path : str or Path, optional
        Path to save background soft labels, by default None.

    Notes
    -----
    The function saves multiple outputs to specified paths or default locations in output_dir:
    - cc_markers.json: Contains detected landmarks and measurements.
    - midplane_slices.mgz: Extracted midplane slices.
    - upright_volume.mgz: Volume aligned to standard orientation.
    - segmentation.mgz: Corpus callosum segmentation.
    - cc_postproc_results.json: Enhanced postprocessing results.
    - Various visualization plots and transformation matrices.

    The pipeline consists of the following steps:
    1. Initializes environment and loads models.
    2. Registers input image to fsaverage space.
    3. Detects AC and PC points.
    4. Segments the corpus callosum.
    5. Performs enhanced post-processing analysis.
    6. Saves results and visualizations.
    """

    if subdivisions is None:
        subdivisions = [1 / 6, 1 / 2, 2 / 3, 3 / 4]

    # Set up logging if verbose mode is enabled
    if verbose:
        logging.setup_logging(None)  # Log to stdout only
    
    logger.info("Starting corpus callosum analysis pipeline")
    logger.info(f"Input MRI: {in_mri_path}")
    logger.info(f"Input segmentation: {aseg_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Convert all paths to Path objects
    in_mri_path = Path(in_mri_path)
    aseg_path = Path(aseg_path)
    output_dir = Path(output_dir)
    debug_output_dir = Path(debug_output_dir) if debug_output_dir else None
    save_template = Path(save_template) if save_template else None

    # Validate subdivision fractions
    for i in subdivisions:
        if i < 0 or i > 1:
            logger.error(f"Error: Subdivision fractions must be between 0 and 1, but got: {i}")
            raise ValueError(f"Subdivision fractions must be between 0 and 1, but got: {i}")

    #### setup variables
    IO_processes = []

    orig = nib.load(in_mri_path)

    # 5 mm around the midplane
    slices_to_analyze = int(np.ceil(5 / orig.header.get_zooms()[0]))
    if slices_to_analyze % 2 == 0:
        slices_to_analyze += 1

    if verbose:
        logger.info(
            f"Segmenting {slices_to_analyze} slices (5 mm width at {orig.header.get_zooms()[0]} mm resolution, "
            "center around the mid-sagittal plane)"
        )

    if not is_conform(orig, vox_size='min', img_size=None):
        logger.error("Error: MRI is not conformed, please run conform.py or mri_convert to conform the image.")
        raise ValueError("MRI is not conformed, please run conform.py or mri_convert to conform the image.")

    # load models
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Loading localization model")
    model_localization = localization_inference.load_model(
        str(Path(WEIGHTS_PATH) / "localization_weights_acpc.pth"), device=device
    )
    logger.info("Loading segmentation model")
    model_segmentation = segmentation_inference.load_model(
        str(Path(WEIGHTS_PATH) / "segmentation_weights_cc_fn.pth"), device=device
    )

    aseg_nib = nib.load(aseg_path)

    logger.info("Performing centroid registration to fsaverage space")
    (orig_fsaverage_vox2vox, orig_fsaverage_ras2ras, 
     fsaverage_hires_affine, fsaverage_header, fsaverage_vox2ras_tkr) = centroid_registration(
        aseg_nib, verbose=False
    )

    if verbose:
        logger.info("Interpolating midplane")

    logger.info("Interpolating midplane slices")
    # this is a fast interpolation to not block the main thread
    midslices = interpolate_midplane(orig, orig_fsaverage_vox2vox, slices_to_analyze)

    # start saving upright volume
    IO_processes.append(
        run_in_background(
            apply_transform_to_volume,
            False,
            orig.get_fdata(),
            orig_fsaverage_vox2vox,
            fsaverage_hires_affine,
            None,
            upright_volume_path,
            output_size=np.array([256, 256, 256]),
        )
    )

    #### do localization and segmentation inference
    logger.info("Starting AC/PC localization")
    ac_coords, pc_coords = localize_ac_pc(
        midslices, aseg_nib, orig_fsaverage_vox2vox, model_localization, slices_to_analyze
    )
    logger.info("Starting corpus callosum segmentation")
    segmentation, outputs_soft = segment_cc(
        midslices, ac_coords, pc_coords, aseg_nib, model_segmentation, slices_to_analyze
    )

    # calculate affine for segmentation volume
    orig_to_seg = np.eye(4)
    orig_to_seg[0, 3] = -FSAVERAGE_MIDDLE + slices_to_analyze // 2
    seg_affine = fsaverage_hires_affine
    seg_affine = seg_affine @ np.linalg.inv(orig_to_seg)

    # save softlabels
    if softlabels_background_path is not None:
        if verbose:
            logger.info(f"Saving background softlabels to {softlabels_background_path}")
        save_nifti_background(IO_processes, outputs_soft[..., 0], seg_affine, orig.header, softlabels_background_path)
    if softlabels_cc_path is not None:
        if verbose:
            logger.info(f"Saving cc softlabels to {softlabels_cc_path}")
        save_nifti_background(IO_processes, outputs_soft[..., 1], seg_affine, orig.header, softlabels_cc_path)
    if softlabels_fn_path is not None:
        if verbose:
            logger.info(f"Saving fornix softlabels to {softlabels_fn_path}")
        save_nifti_background(IO_processes, outputs_soft[..., 2], seg_affine, orig.header, softlabels_fn_path)
    

    # Create a temporary segmentation image with proper affine for enhanced postprocessing
    temp_seg_affine = fsaverage_hires_affine @ np.linalg.inv(np.eye(4))

    # Process slices based on selection mode
    logger.info(f"Processing slices with selection mode: {slice_selection}")
    slice_results, slice_io_processes = process_slices(
        segmentation=segmentation,
        slice_selection=slice_selection,
        temp_seg_affine=temp_seg_affine,
        midslices=midslices,
        ac_coords=ac_coords,
        pc_coords=pc_coords,
        num_thickness_points=num_thickness_points,
        subdivisions=subdivisions,
        subdivision_method=subdivision_method,
        contour_smoothing=contour_smoothing,
        debug_image_path=debug_image_path,
        one_debug_image=True,
        surf_file_path=surf_file_path,
        overlay_file_path=overlay_file_path,
        cc_html_path=cc_html_path,
        vtk_file_path=vtk_file_path,
        thickness_image_path=thickness_image_path,
        vox_size=orig.header.get_zooms(),
        vox2ras_tkr=fsaverage_vox2ras_tkr,
        verbose=verbose,
        save_template=save_template,
    )


    outer_contours = [slice_result['split_contours'][0] for slice_result in slice_results]

    if len(outer_contours) > 1 and not check_area_changes(outer_contours, verbose=True):
        logger.warning("Large area changes detected between consecutive slices, "
                       "this is likely due to a segmentation error.")

    IO_processes.extend(slice_io_processes)

    # Get middle slice result for backward compatibility
    middle_slice_result = slice_results[len(slice_results) // 2]

    if len(middle_slice_result['split_contours']) <= 5:
        subdivision_mask = make_subdivision_mask(segmentation.shape[1:], 
                                                 middle_slice_result['split_contours'], 
                                                 orig.header.get_zooms())
    else:
        logger.warning("Too many subsegments for lookup table, skipping sub-divion of output segmentation.")
        subdivision_mask = None


    # map soft labels to original space (in parallel because this takes a while)
    IO_processes.append(
        run_in_background(
            map_softlabels_to_orig,
            debug=False,
            outputs_soft=outputs_soft,
            orig_fsaverage_vox2vox=orig_fsaverage_vox2vox,
            orig=orig,
            slices_to_analyze=slices_to_analyze,
            orig_space_segmentation_path=orig_space_segmentation_path,
            fsaverage_middle=FSAVERAGE_MIDDLE,
            subdivision_mask=subdivision_mask,
        )
    )

    # Save middle slice visualization
    IO_processes.append(
        create_visualization(
            subdivision_method,
            {
            "split_contours": middle_slice_result["split_contours"],
            "midline_equidistant": middle_slice_result["midline_equidistant"],
            "levelpaths": middle_slice_result["levelpaths"],
            },
            midslices,
            output_dir,
            ac_coords,
            pc_coords,
            orig.header.get_zooms()[0],
            " (Middle Slice)",
        )
    )

    save_nifti_background(IO_processes, segmentation, seg_affine, orig.header, segmentation_path)


    METRICS = [
        "areas",
        "thickness",
        "curvature",
        "midline_length",
        "circularity",
        "cc_index",
        "total_area",
        "total_perimeter",
        "thickness_profile",
    ]

    # Record key metrics for middle slice
    output_metrics_middle_slice = {
        metric: middle_slice_result[metric] for metric in METRICS
    }

    # Create enhanced output dictionary with all slice results
    per_slice_output_dict = {
        "slices": [
            convert_numpy_to_json_serializable(
                {
                    metric: result[metric] for metric in METRICS
                }
            )
            for result in slice_results
        ],
    }

    ########## Save outputs ##########

    additional_metrics = {}
    if len(outer_contours) > 1:
        cc_volume_voxel = segmentation_postprocessing.get_cc_volume_voxel(
            desired_width_mm=5, 
            cc_mask=segmentation == CC_LABEL, 
            voxel_size=orig.header.get_zooms()
        )
        cc_volume_contour = segmentation_postprocessing.get_cc_volume_contour(
            cc_contours=outer_contours, 
            voxel_size=orig.header.get_zooms()
        )
        if verbose:
            logger.info(f"CC volume voxel: {cc_volume_voxel}")
            logger.info(f"CC volume contour: {cc_volume_contour}")
        
        additional_metrics["cc_5mm_volume"] = cc_volume_voxel
        additional_metrics["cc_5mm_volume_pv_corrected"] = cc_volume_contour

    

    # get ac and pc in all spaces
    ac_coords_3d = np.hstack((FSAVERAGE_MIDDLE, ac_coords))
    pc_coords_3d = np.hstack((FSAVERAGE_MIDDLE, pc_coords))
    standardized_to_orig_vox2vox, ac_coords_standardized, pc_coords_standardized, ac_coords_orig, pc_coords_orig = (
        get_mapping_to_standard_space(orig, ac_coords_3d, pc_coords_3d, orig_fsaverage_vox2vox)
    )

    # write output dict as csv
    additional_metrics["ac_center"] = ac_coords_orig
    additional_metrics["pc_center"] = pc_coords_orig
    additional_metrics["ac_center_oriented_volume"] = ac_coords_standardized
    additional_metrics["pc_center_oriented_volume"] = pc_coords_standardized
    additional_metrics["ac_center_upright"] = ac_coords_3d
    additional_metrics["pc_center_upright"] = pc_coords_3d
    additional_metrics["slices_in_segmentation"] = slices_to_analyze
    additional_metrics["voxel_size"] = [float(x) for x in orig.header.get_zooms()]
    additional_metrics["num_thickness_points"] = num_thickness_points
    additional_metrics["subdivision_method"] = subdivision_method
    additional_metrics["subdivision_ratios"] = subdivisions
    additional_metrics["contour_smoothing"] = contour_smoothing
    additional_metrics["slice_selection"] = slice_selection

    # Convert numpy arrays to lists for JSON serialization
    output_metrics_middle_slice = convert_numpy_to_json_serializable(output_metrics_middle_slice | additional_metrics)

    logger.info(f"Saving CC markers to {cc_markers_path}")
    with open(cc_markers_path, "w") as f:
        json.dump(output_metrics_middle_slice, f, indent=4)


    per_slice_output_dict = convert_numpy_to_json_serializable(per_slice_output_dict | additional_metrics)

    # Save slice-wise postprocessing results to JSON
    with open(postproc_results_path, "w") as f:
        json.dump(per_slice_output_dict, f, indent=4)

    if verbose:
        logger.info(f"Multiple slice post-processing results saved to {postproc_results_path}")

    # save lta to fsaverage space
    logger.info(f"Saving LTA to fsaverage space: {upright_lta_path}")
    lta.writeLTA(upright_lta_path, orig_fsaverage_ras2ras, aseg_path, aseg_nib.header, "fsaverage", fsaverage_header)

    # save lta to standardized space (fsaverage + nodding + ac to center)
    orig_to_standardized_ras2ras = (
        orig.affine @ np.linalg.inv(standardized_to_orig_vox2vox) @ np.linalg.inv(orig.affine)
    )
    logger.info(f"Saving LTA to standardized space: {orient_volume_lta_path}")
    lta.writeLTA(
        orient_volume_lta_path, orig_to_standardized_ras2ras, in_mri_path, orig.header, in_mri_path, orig.header
    )

    for process in IO_processes:
        if process is not None:
            process.join()
    
    logger.info("CorpusCallosum analysis pipeline completed successfully")


if __name__ == "__main__":
    options = options_parse()
    main_args = vars(options)

    # Rename keys to match main function parameters
    main_args["in_mri_path"] = main_args.pop("in_mri")
    main_args["aseg_path"] = main_args.pop("aseg")
    main_args["output_dir"] = main_args.pop("subject_dir", ".")

    main(**main_args)
