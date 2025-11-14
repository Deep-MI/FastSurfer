#!/usr/bin/env python3
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

import argparse
import json
from pathlib import Path
from typing import Literal, cast

import nibabel as nib
import numpy as np
import torch
from monai.networks.nets import DenseNet
from numpy import typing as npt

from CorpusCallosum.data.constants import (
    CC_LABEL,
    FSAVERAGE_CENTROIDS_PATH,
    FSAVERAGE_DATA_PATH,
    FSAVERAGE_MIDDLE,
    STANDARD_INPUT_PATHS,
    STANDARD_OUTPUT_PATHS,
)
from CorpusCallosum.data.read_write import (
    FSAverageHeader,
    convert_numpy_to_json_serializable,
    get_centroids_from_nib,
    load_fsaverage_centroids,
    load_fsaverage_data,
)
from CorpusCallosum.localization import localization_inference
from CorpusCallosum.registration.mapping_helpers import (
    apply_transform_to_pt,
    apply_transform_to_volume,
    calc_mapping_to_standard_space,
    interpolate_midplane,
    map_softlabels_to_orig,
)
from CorpusCallosum.segmentation import segmentation_inference, segmentation_postprocessing
from CorpusCallosum.shape.cc_postprocessing import (
    SubdivisionMethod,
    check_area_changes,
    make_subdivision_mask,
    recon_cc_surf_measures_multi,
)
from FastSurferCNN.data_loader.conform import is_conform
from FastSurferCNN.utils import logging
from FastSurferCNN.utils.common import SubjectDirectory, find_device
from FastSurferCNN.utils.common import thread_executor as executor
from recon_surf import lta
from recon_surf.align_points import find_rigid

logger = logging.get_logger(__name__)
SliceSelection = Literal["middle", "all"] | int


def make_parser() -> argparse.ArgumentParser:
    """Create the argument parse object for the pipeline."""
    from FastSurferCNN.utils.parser_defaults import add_arguments

    parser = argparse.ArgumentParser()

    # Specify subject directory + subject ID, OR specify individual MRI and segmentation files + output paths
    add_arguments(parser, ["sd", "sid", "conformed_name", "aseg_name", "device"])

    parser.add_argument(
        "--num_thickness_points",
        type=int,
        default=100,
        help="Number of points for thickness estimation."
    )
    parser.add_argument(
        "--subdivisions",
        type=float,
        metavar="FRAC",
        nargs=4,
        default=[1/6, 1/2, 2/3, 3/4],
        help="List of FOUR subdivision fractions for the corpus callosum subsegmentation.",
    )
    parser.add_argument(
        "--subdivision_method",
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
        help="Gaussian sigma for smoothing during contour detection. Higher values mean a smoother"
        " CC outline, at the cost of precision. (default: 5)",
    )
    def _slice_selection(a: str) -> SliceSelection:
        if a.lower() in ("middle", "all"):
            return a.lower()
        return int(a)
    parser.add_argument(
        "--slice_selection",
        type=_slice_selection,
        default="all",
        help="Which slices to process. Options: 'middle', 'all', or a specific slice number. \
              (default: 'all')",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose (pass twice for debug-output).",
    )

    ######## OUTPUT PATHS #########
    # 4. Options for advanced, technical parameters
    advanced = parser.add_argument_group(
        title="Advanced options",
        description="Custom output paths, useful if no standard case directory is used.",
    )
    advanced.add_argument("--qc_output_dir",
        type=Path,
        required=False,
        default=None,
        help="Directory for quality control output (default: subject_dir/qc_snapshots)")
    advanced.add_argument(
        "--upright_volume_path",
        type=Path,
        help="Path for upright volume output (default: No output)",
        default=None,
    )
    advanced.add_argument(
        "--seg",
        dest="segmentation_path",
        type=Path,
        help=f"Path for segmentation output (default: subject_dir/{STANDARD_OUTPUT_PATHS['segmentation']})",
        default=None,
    )
    advanced.add_argument(
        "--postproc_results_path",
        type=Path,
        help=f"Path for postprocessing results. Contains metrics describing CC shape and volume for each slice \
               (default: subject_dir/{STANDARD_OUTPUT_PATHS['postproc_results']})",
        default=None,
    )
    advanced.add_argument(
        "--cc_markers_path",
        type=Path,
        help=f"Path for CC markers output. Contains metrics describing CC shape and volume \
               (default: subject_dir/{STANDARD_OUTPUT_PATHS['cc_markers']})",
        default=None,
    )
    advanced.add_argument(
        "--upright_lta_path",
        type=Path,
        help=f"Path for upright LTA transform. This makes sure the midplane is at 128 in LR direction, but no nodding \
               correction is applied (default: subject_dir/{STANDARD_OUTPUT_PATHS['upright_lta']})",
        default=None,
    )
    advanced.add_argument(
        "--orient_volume_lta_path",
        type=Path,
        help=f"Path for orientation volume LTA transform. This makes sure the midplane is at 128 in LR direction, \
              and the AC & PC are on the coordinate line, standardizing the head orientation. \
              (default: subject_dir/{STANDARD_OUTPUT_PATHS['orient_volume_lta']})",
        default=None,
    )
    advanced.add_argument(
        "--orig_space_segmentation_path",
        type=Path,
        help="Path for segmentation in the input MRI space "
             f"(default: subject_dir/{STANDARD_OUTPUT_PATHS['orig_space_segmentation']})",
        default=None,
    )
    advanced.add_argument(
        "--qc_image_path",
        type=Path,
        help=f"Path for QC visualization image (default: subject_dir/{STANDARD_OUTPUT_PATHS['qc_image']})",
        default=None,
    )
    advanced.add_argument(
        "--save_template_dir",
        type=Path,
        help="Directory path where to save contours.txt and thickness_values.txt files. \
              These files can be used to visualize the CC shape and volume in 3D.",
        default=None,
    )
    advanced.add_argument(
        "--thickness_image_path",
        type=Path,
        help=f"Path for thickness image (default: subject_dir/{STANDARD_OUTPUT_PATHS['thickness_image']})",
        default=None,
    )
    advanced.add_argument(
        "--surf_file_path",
        type=Path,
        help=f"Path for surf file (default: subject_dir/{STANDARD_OUTPUT_PATHS['surf_file']})",
        default=None,
    )
    advanced.add_argument(
        "--overlay_file_path",
        type=Path,
        help=f"Path for overlay file (default: subject_dir/{STANDARD_OUTPUT_PATHS['overlay_file']})",
        default=None,
    )
    advanced.add_argument(
        "--cc_html_path",
        type=Path,
        help=f"Path to CC 3D visualization for CC HTML file (default: subject_dir/{STANDARD_OUTPUT_PATHS['cc_html']})",
        default=None,
    )
    advanced.add_argument(
        "--vtk_file_path",
        type=Path,
        help=f"Path for vtk file, showing the CC 3D mesh (default: subject_dir/{STANDARD_OUTPUT_PATHS['vtk_file']})",
        default=None,
    )
    advanced.add_argument(
        "--softlabels_cc_path",
        type=Path,
        help=f"Path for cc softlabels. Contains the probability of each voxel being part of the CC \
              (default: subject_dir/{STANDARD_OUTPUT_PATHS['softlabels_cc']})",
        default=None,
    )
    advanced.add_argument(
        "--softlabels_fn_path",
        type=Path,
        help=f"Path for fornix softlabels. Contains the probability of each voxel being part of the Fornix \
              (default: subject_dir/{STANDARD_OUTPUT_PATHS['softlabels_fn']})",
        default=None,
    )
    advanced.add_argument(
        "--softlabels_background_path",
        type=Path,
        help=f"Path for background softlabels. Contains the probability of each voxel being part of the background \
              (default: subject_dir/{STANDARD_OUTPUT_PATHS['softlabels_background']})",
        default=None,
    )
    ############ END OF OUTPUT PATHS ############
    return parser


def options_parse() -> argparse.Namespace:
    """Parse command line arguments for the pipeline."""
    parser = make_parser()
    args = parser.parse_args()

    # Reconstruct subject_dir from sd and sid (but sd might be stored as out_dir by parser_defaults)
    sd_value = getattr(args, 'sd', getattr(args, 'out_dir', None))
    if sd_value and hasattr(args, 'sid') and args.sid:
        args.subject_dir = Path(sd_value) / args.sid
    else:
        args.subject_dir = None

    # Validation logic: must use either directory approach (--sd + --sid) OR file approach (--conf_name + --aseg_name)
    if sd_value:
        # Using directory approach - make sure sid was also provided
        if not (hasattr(args, 'sid') and args.sid):
            parser.error("When using --sd, you must also provide --sid.")
    elif hasattr(args, 'sid') and args.sid:
        # If sid is provided without sd, that's an error
        if not sd_value:
            parser.error("When using --sid, you must also provide --sd.")
    elif hasattr(args, 'conf_name') and args.conf_name:
        # Using file approach - make sure aseg_name was also provided
        if not (hasattr(args, 'aseg_name') and args.aseg_name):
            parser.error("When using --conf_name, you must also provide --aseg_name.")
    elif hasattr(args, 'aseg_name') and args.aseg_name:
        # If aseg_name is provided without conf_name, that's an error
        if not (hasattr(args, 'conf_name') and args.conf_name):
            parser.error("When using --aseg_name, you must also provide --conf_name.")
    else:
        parser.error("You must specify either --sd and --sid OR both --conf_name and --aseg_name.")

    # If subject_dir is provided, set default paths for missing arguments
    if args.subject_dir:
        subject_dir_path = args.subject_dir

        # Create standard FreeSurfer subdirectories
        (subject_dir_path / "mri").mkdir(parents=True, exist_ok=True)
        (subject_dir_path / "stats").mkdir(parents=True, exist_ok=True)
        (subject_dir_path / "transforms").mkdir(parents=True, exist_ok=True)

        if not args.conf_name:
            args.conf_name = str(subject_dir_path / STANDARD_INPUT_PATHS["conf_name"])

        if not args.aseg_name:
            args.aseg_name = str(subject_dir_path / STANDARD_INPUT_PATHS["aseg_name"])

        # Set default output paths if not provided
        for key, value in STANDARD_OUTPUT_PATHS.items():
            if not getattr(args, f"{key}_path") and value is not None:
                setattr(args, f"{key}_path", subject_dir_path / value)

        # Set output_dir to subject_dir
        args.output_dir = str(subject_dir_path)

    # Create parent directories for all output paths
    for path_name in STANDARD_OUTPUT_PATHS.keys():
        path = getattr(args, f"{path_name}_path")
        if path is not None:
            Path(path).parent.mkdir(parents=False, exist_ok=True)

    return args


def centroid_registration(aseg_nib: nib.analyze.SpatialImage) -> tuple[
    npt.NDArray[float], npt.NDArray[float], npt.NDArray[float], FSAverageHeader, npt.NDArray[float]
]:
    """Perform centroid-based registration between subject and fsaverage space.

    Computes a rigid transformation between the subject's segmentation and fsaverage space
    by aligning centroids of corresponding anatomical structures.

    Parameters
    ----------
    aseg_nib : nibabel.analyze.SpatialImage
        Subject's segmentation image.

    Returns
    -------
    orig_fsaverage_vox2vox : np.ndarray
        Transformation matrix from original to fsaverage voxel space.
    orig_fsaverage_ras2ras : np.ndarray
        Transformation matrix from original to fsaverage RAS space.
    fsaverage_hires_affine : np.ndarray
        High-resolution fsaverage affine matrix.
    fsaverage_header : FSAverageHeader
        FSAverage header fields for LTA writing.
    vox2ras_tkr : np.ndarray
        Voxel to RAS tkr-space transformation matrix.

    Notes
    -----
    The function uses pre-computed fsaverage centroids and data from static files
    to perform the registration. It matches corresponding anatomical structures
    between the subject's segmentation and fsaverage space.
    """
    logger.info("Starting centroid registration")

    # Load pre-computed fsaverage centroids and data from static files
    centroids_dst = load_fsaverage_centroids(FSAVERAGE_CENTROIDS_PATH)
    fsaverage_affine, fsaverage_header, vox2ras_tkr = load_fsaverage_data(FSAVERAGE_DATA_PATH)

    centroids_mov = get_centroids_from_nib(aseg_nib, label_ids=list(centroids_dst.keys()))

    # get the set of joint labels
    joint_centroid_labels = [lbl for lbl, v in centroids_mov.items() if v is not None]

    centroids_mov = np.array([centroids_mov[lbl] for lbl in joint_centroid_labels]).T
    centroids_dst = np.array([centroids_dst[lbl] for lbl in joint_centroid_labels]).T

    orig_fsaverage_ras2ras: npt.NDArray[float]  = find_rigid(p_mov=centroids_mov.T, p_dst=centroids_dst.T)

    # make affine that increases resolution to orig resolution
    resolution_trans: npt.NDArray[float] = np.diagflat(list(aseg_nib.header.get_zooms()[:3]) + [1]).astype(float)

    orig_fsaverage_vox2vox: npt.NDArray[float] = (
        np.linalg.inv(resolution_trans @ fsaverage_affine) @ orig_fsaverage_ras2ras @ aseg_nib.affine
    )
    fsaverage_hires_affine: npt.NDArray[float] = resolution_trans @ fsaverage_affine
    logger.info("Centroid registration successful!")
    return orig_fsaverage_vox2vox, orig_fsaverage_ras2ras, fsaverage_hires_affine, fsaverage_header, vox2ras_tkr


def localize_ac_pc(
    midslices: np.ndarray,
    aseg_nib: nib.analyze.SpatialImage,
    orig_fsaverage_vox2vox: npt.NDArray[float],
    model_localization: DenseNet,
    num_slices_to_analyze: int
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Localize anterior and posterior commissure points in the brain.

    Uses a trained model to detect AC and PC points in mid-sagittal slices,
    using the third ventricle as an anatomical reference.

    Parameters
    ----------
    midslices : np.ndarray
        Array of mid-sagittal slices.
    aseg_nib : nibabel.analyze.SpatialImage
        Subject's segmentation image in native subject space.
    orig_fsaverage_vox2vox : np.ndarray
        Transformation matrix from subject/native space to fsaverage space (in lia).
    model_localization : DenseNet
        Trained model for AC-PC detection.
    num_slices_to_analyze : int
        Number of slices to process.

    Returns
    -------
    ac_coords : np.ndarray
        Coordinates of the anterior commissure.
    pc_coords : np.ndarray
        Coordinates of the posterior commissure.
    """

    # get center of third ventricle from aseg and map to fsaverage space
    third_ventricle_mask = np.asarray(aseg_nib.dataobj) == 4
    third_ventricle_center = np.argwhere(third_ventricle_mask).mean(axis=0)
    third_ventricle_center_vox = apply_transform_to_pt(third_ventricle_center, orig_fsaverage_vox2vox, inv=False)

    # get 5 mm of slices output with 3 slices per inference
    midslices_start = midslices.shape[0] // 2 - num_slices_to_analyze // 2 - 1
    middle_slices_localization = midslices[midslices_start:midslices_start + num_slices_to_analyze + 3]
    ac_coords, pc_coords = localization_inference.run_inference_on_slice(
        model_localization, middle_slices_localization, third_ventricle_center_vox[1:],
    )

    return ac_coords, pc_coords


def segment_cc(
    midslices: np.ndarray,
    ac_coords: npt.NDArray[float],
    pc_coords: npt.NDArray[float],
    aseg_nib: "nib.Nifti1Image",
    model_segmentation: "torch.nn.Module",
    slices_to_analyze: int,
) -> tuple[npt.NDArray[bool], npt.NDArray[float]]:
    """Segment the corpus callosum using a trained model.

    Performs corpus callosum segmentation on mid-sagittal slices using a trained model, with AC-PC points as anatomical
    references. Includes post-processing to clean the cc_seg_labels.

    Parameters
    ----------
    midslices : np.ndarray
        Array of mid-sagittal slices.
    ac_coords : np.ndarray
        Anterior commissure coordinates.
    pc_coords : np.ndarray
        Posterior commissure coordinates.
    aseg_nib : nibabel.Nifti1Image
        Subject's cc_seg_labels image.
    model_segmentation : torch.nn.Module
        Trained model for CC cc_seg_labels.
    slices_to_analyze : int
        Number of slices to process.

    Returns
    -------
    cc_seg_labels : np.ndarray
        Binary cc_seg_labels of the corpus callosum.
    cc_softlabels : np.ndarray
        Soft cc_seg_labels probabilities.
    """
    # get 5 mm of slices output with 9 slices per inference
    midslices_start = midslices.shape[0] // 2 - slices_to_analyze // 2 - 4
    middle_slices_slab = midslices[midslices_start:midslices_start + slices_to_analyze + 9]
    pre_clean_segmentation, inputs, cc_softlabels = segmentation_inference.run_inference_on_slice(
        model_segmentation,
        middle_slices_slab,
        ac_center=ac_coords,
        pc_center=pc_coords,
        voxel_size=aseg_nib.header.get_zooms()[0],
    )

    cc_seg_labels, cc_volume_mask = segmentation_postprocessing.clean_cc_segmentation(pre_clean_segmentation)

    # print a warning if the cc_volume_mask touches the edge of the segmentation
    if np.any(cc_volume_mask[:, [0, -1]]) or np.any(cc_volume_mask[:, :, [0, -1]]):
        logger.warning("CC volume mask touches the edge of the cc_seg_labels field-of-view, CC might be truncated")

    # get voxels that were removed during cleaning
    cleaned_mask = pre_clean_segmentation != cc_seg_labels
    cc_softlabels[cleaned_mask, 1] = 0
    cc_softlabels[cleaned_mask, :] /= np.sum(cc_softlabels[cleaned_mask, :], axis=-1, keepdims=True) + 1e-6

    return cc_seg_labels, cc_softlabels


def main(
    conf_name: str | Path,
    aseg_name: str | Path,
    subject_dir: str | Path,
    slice_selection: SliceSelection = "middle",
    #TODO: qc_output_dir is currently unused ?!
    qc_output_dir: str | Path | None = None,
    num_thickness_points: int = 100,
    subdivisions: list[float] | None = None,
    subdivision_method: SubdivisionMethod = "shape",
    contour_smoothing: float = 5,
    save_template_dir: str | Path | None = None,
    device: str | torch.device = "auto",
    upright_volume_path: str | Path | None = None,
    segmentation_path: str | Path | None = None,
    postproc_results_path: str | Path | None = None,
    cc_markers_path: str | Path | None = None,
    upright_lta_path: str | Path | None = None,
    orient_volume_lta_path: str | Path | None = None,
    surf_file_path: str | Path | None = None,
    overlay_file_path: str | Path | None = None,
    cc_html_path: str | Path | None = None,
    vtk_file_path: str | Path | None = None,
    orig_space_segmentation_path: str | Path | None = None,
    qc_image_path: str | Path | None = None,
    thickness_image_path: str | Path | None = None,
    softlabels_cc_path: str | Path | None = None,
    softlabels_fn_path: str | Path | None = None,
    softlabels_background_path: str | Path | None = None,
) -> None:
    """Main pipeline function for corpus callosum analysis.

    This function performs the complete corpus callosum analysis pipeline including
    registration, landmark detection, segmentation, and morphometry analysis.

    Parameters
    ----------
    conf_name : str or Path
        Path to input MRI file.
    aseg_name : str or Path
        Path to input segmentation file.
    subject_dir : str or Path
        FastSurfer/FreeSurfer subject directory and directory for output files.
    slice_selection : "middle", "all" or int, default="middle"
        Which slices to process.
    qc_output_dir : str or Path, optional
        Directory for quality control outputs, None deactivates qc snapshots.
    num_thickness_points : int, default=100
        Number of points for thickness estimation.
    subdivisions : list[float], optional
        List of subdivision fractions for CC subsegmentation.
    subdivision_method : any of "shape", "vertical", "angular", "eigenvector", default="shape"
        Method for contour subdivision.
    contour_smoothing : float, default=5
        Gaussian sigma for smoothing during contour detection.
    save_template_dir : str or Path, optional
        Directory path where to save contours.txt and thickness_values.txt files. These files can be used to visualize
        the CC shape and volume in 3D. Files are only saved, if a valid directory path is passed.
    device : str, default="auto"
        Device to run inference on ('auto', 'cpu', 'cuda', or 'cuda:X').
    upright_volume_path : str or Path, optional
        Path to save upright volume.
    segmentation_path : str or Path, optional
        Path to save segmentation.
    postproc_results_path : str or Path, optional
        Path to save post-processing results.
    cc_markers_path : str or Path, optional
        Path to save CC markers.
    upright_lta_path : str or Path, optional
        Path to save upright LTA transform.
    orient_volume_lta_path : str or Path, optional
        Path to save orientation transform.
    surf_file_path : str or Path, optional
        Path to save surface file.
    overlay_file_path : str or Path, optional
        Path to save overlay file.
    cc_html_path : str or Path, optional
        Path to save HTML visualization.
    vtk_file_path : str or Path, optional
        Path to save VTK file.
    orig_space_segmentation_path : str or Path, optional
        Path to save segmentation in original space.
    qc_image_path : str or Path, optional
        Path to save QC images.
    thickness_image_path : str or Path, optional
        Path to save thickness visualization.
    softlabels_cc_path : str or Path, optional
        Path to save CC soft labels.
    softlabels_fn_path : str or Path, optional
        Path to save fornix soft labels.
    softlabels_background_path : str or Path, optional
        Path to save background soft labels.

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
    import sys

    if subdivisions is None:
        subdivisions = [1 / 6, 1 / 2, 2 / 3, 3 / 4]

    subject_dir = Path(subject_dir)

    logger.info("Starting corpus callosum analysis pipeline")
    logger.info(f"Input MRI: {conf_name}")
    logger.info(f"Input segmentation: {aseg_name}")
    logger.info(f"Output directory: {subject_dir}")
    
    # Convert all paths to Path objects
    sd = SubjectDirectory(
        subject_dir.parent,
        id=subject_dir.name,
        conf_name=conf_name,
        aseg_name=aseg_name,
        save_template_dir=save_template_dir,
        upright_volume=upright_volume_path,
        cc_segmentation=segmentation_path,
        cc_postproc_results=postproc_results_path,
        cc_markers=cc_markers_path,
        upright_lta=upright_lta_path,
        cc_orient_volume_lta=orient_volume_lta_path,
        cc_surf=surf_file_path,
        cc_overlay=overlay_file_path,
        cc_html=cc_html_path,
        cc_mesh=vtk_file_path,
        cc_orig_segfile=orig_space_segmentation_path,
        cc_qc_images=qc_image_path,
        cc_thickness_image=thickness_image_path,
        cc_softlabels_cc=softlabels_cc_path,
        cc_softlabels_fn=softlabels_fn_path,
        cc_softlabels_background=softlabels_background_path,
    )

    # Validate subdivision fractions
    if any(i < 0 or i > 1 for i in subdivisions):
        logger.error(f"Subdivision fractions must be between 0 and 1, but got: {subdivisions}")
        sys.exit(1)

    #### setup variables
    io_futures = []

    orig = nib.load(sd.conf_name)

    # 5 mm around the midplane (making sure to get rl by as_closest_canonical)
    slices_to_analyze = int(np.ceil(5 / nib.as_closest_canonical(orig).header.get_zooms()[0])) // 2 * 2 + 1

    logger.info(
        f"Segmenting {slices_to_analyze} slices (5 mm width at {orig.header.get_zooms()[0]} mm resolution, "
        "center around the mid-sagittal plane)"
    )

    if not is_conform(orig, vox_size='min', img_size=None):
        if is_conform(orig, vox_size=None, img_size=None):
            logger.warning("fastsurfer_cc currently requires isotropic images.")
        logger.error("MRI is not conformed, please run conform.py or mri_convert to conform the image.")
        sys.exit(1)

    # load models
    device = find_device(device)
    logger.info(f"Using device: {device}")
    
    logger.info("Loading models")
    model_localization = localization_inference.load_model(device=device)
    model_segmentation = segmentation_inference.load_model(device=device)

    aseg_nib = cast(nib.analyze.SpatialImage, nib.load(sd.filename_by_attribute("aseg_name")))

    logger.info("Performing centroid registration to fsaverage space")
    orig2fsavg_vox2vox, orig2fsavg_ras2ras, fsavg_affine, fsavg_header, fsavg_vox2ras_tkr = centroid_registration(
        aseg_nib
    )
    logger.info("Interpolating midplane slices")
    # this is a fast interpolation to not block the main thread
    midslices = interpolate_midplane(orig, orig2fsavg_vox2vox, slices_to_analyze)

    # start saving upright volume
    if sd.has_attribute("upright_volume"):
        io_futures.append(
            executor().submit(
                apply_transform_to_volume,
                orig,
                orig2fsavg_vox2vox,
                fsavg_affine,
                output_path=sd.filename_by_attribute("upright_volume"),
                output_size=np.array([256, 256, 256]),
            )
        )

    #### do localization and segmentation inference
    logger.info("Starting AC/PC localization")
    ac_coords, pc_coords = localize_ac_pc(
        midslices, aseg_nib, orig2fsavg_vox2vox, model_localization, slices_to_analyze,
    )
    logger.info("Starting corpus callosum segmentation")
    cc_fn_seg_labels, cc_fn_softlabels = segment_cc(
        midslices, ac_coords, pc_coords, aseg_nib, model_segmentation, slices_to_analyze,
    )

    # calculate affine for segmentation volume
    orig_to_seg = np.eye(4)
    orig_to_seg[0, 3] = -FSAVERAGE_MIDDLE + slices_to_analyze // 2
    seg_affine = fsavg_affine @ np.linalg.inv(orig_to_seg)

    # save softlabels
    for i, (attr, name) in enumerate((("background",) * 2, ("cc", "Corpus Callosum"), ("fn", "Fornix"))):
        if sd.has_attribute(f"cc_softlabels_{attr}"):
            logger.info(f"Saving {name} softlabels to {sd.filename_by_attribute(f'cc_softlabels_{attr}')}")
            io_futures.append(executor().submit(
                nib.save,
                nib.MGHImage(cc_fn_softlabels[..., i], seg_affine, orig.header),
                sd.filename_by_attribute(f"cc_softlabels_{attr}"),
            ))

    # Create a temporary segmentation image with proper affine for enhanced postprocessing
    # Process slices based on selection mode
    logger.info(f"Processing slices with selection mode: {slice_selection}")
    slice_results, slice_io_futures = recon_cc_surf_measures_multi(
        segmentation=cc_fn_seg_labels,
        slice_selection=slice_selection,
        temp_seg_affine=fsavg_affine,
        midslices=midslices,
        ac_coords=ac_coords,
        pc_coords=pc_coords,
        num_thickness_points=num_thickness_points,
        subdivisions=subdivisions,
        subdivision_method=subdivision_method,
        contour_smoothing=contour_smoothing,
        vox_size=orig.header.get_zooms(),
        vox2ras_tkr=fsavg_vox2ras_tkr,
        subject_dir=sd,
    )
    io_futures.extend(slice_io_futures)

    outer_contours = [slice_result['split_contours'][0] for slice_result in slice_results]

    if len(outer_contours) > 1 and not check_area_changes(outer_contours):
        logger.warning(
            "Large area changes detected between consecutive slices, this is likely due to a segmentation error."
        )

    # Get middle slice result for backward compatibility
    middle_slice_result = slice_results[len(slice_results) // 2]

    if len(middle_slice_result['split_contours']) <= 5:
        subdivision_mask = make_subdivision_mask(
            cc_fn_seg_labels.shape[1:],
            middle_slice_result['split_contours'],
            orig.header.get_zooms(),
        )
    else:
        logger.warning("Too many subsegments for lookup table, skipping sub-divion of output segmentation.")
        subdivision_mask = None

    # map soft labels to original space (in parallel because this takes a while)
    io_futures.append(executor().submit(
        map_softlabels_to_orig,
        outputs_soft=cc_fn_softlabels,
        orig_fsaverage_vox2vox=orig2fsavg_vox2vox,
        orig=orig,
        slices_to_analyze=slices_to_analyze,
        orig_space_segmentation_path=orig_space_segmentation_path,
        fsaverage_middle=FSAVERAGE_MIDDLE,
        subdivision_mask=subdivision_mask,
    ))
    io_futures.append(executor().submit(
        nib.save,
        nib.MGHImage(cc_fn_seg_labels, seg_affine, orig.header),
        sd.filename_by_attribute("cc_segmentation"),
    ))

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
    output_metrics_middle_slice = {metric: middle_slice_result[metric] for metric in METRICS}

    # Create enhanced output dictionary with all slice results
    per_slice_output_dict = {
        "slices": [
            convert_numpy_to_json_serializable({metric: result[metric] for metric in METRICS})
            for result in slice_results
        ],
    }

    ########## Save outputs ##########
    additional_metrics = {}
    if len(outer_contours) > 1:
        cc_volume_voxel = segmentation_postprocessing.get_cc_volume_voxel(
            desired_width_mm=5, 
            cc_mask=cc_fn_seg_labels == CC_LABEL,
            voxel_size=orig.header.get_zooms()
        )
        cc_volume_contour = segmentation_postprocessing.get_cc_volume_contour(
            cc_contours=outer_contours, 
            voxel_size=orig.header.get_zooms()
        )
        logger.info(f"CC volume voxel: {cc_volume_voxel}")
        logger.info(f"CC volume contour: {cc_volume_contour}")
        
        additional_metrics["cc_5mm_volume"] = cc_volume_voxel
        additional_metrics["cc_5mm_volume_pv_corrected"] = cc_volume_contour

    

    # get ac and pc in all spaces
    ac_coords_3d = np.hstack((FSAVERAGE_MIDDLE, ac_coords))
    pc_coords_3d = np.hstack((FSAVERAGE_MIDDLE, pc_coords))
    standardized_to_orig_vox2vox, ac_coords_standardized, pc_coords_standardized, ac_coords_orig, pc_coords_orig = (
        calc_mapping_to_standard_space(orig, ac_coords_3d, pc_coords_3d, orig2fsavg_vox2vox)
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

    logger.info(f"Saving CC markers to {sd.filename_by_attribute('cc_markers')}")
    with open(sd.filename_by_attribute("cc_markers"), "w") as f:
        json.dump(output_metrics_middle_slice, f, indent=4)

    per_slice_output_dict = convert_numpy_to_json_serializable(per_slice_output_dict | additional_metrics)

    # Save slice-wise postprocessing results to JSON
    with open(sd.filename_by_attribute("cc_postproc_results"), "w") as f:
        json.dump(per_slice_output_dict, f, indent=4)

    logger.info(f"Multiple slice post-processing results saved to {sd.filename_by_attribute('cc_postproc_results')}")

    # save lta to fsaverage space
    logger.info(f"Saving LTA to fsaverage space: {sd.filename_by_attribute('upright_lta')}")
    lta.writeLTA(
        sd.filename_by_attribute("upright_lta"),
        orig2fsavg_ras2ras,
        sd.filename_by_attribute("aseg_name"),
        aseg_nib.header,
        "fsaverage",
        fsavg_header,
    )

    # save lta to standardized space (fsaverage + nodding + ac to center)
    orig2standardized_ras2ras = orig.affine @ np.linalg.inv(standardized_to_orig_vox2vox) @ np.linalg.inv(orig.affine)
    logger.info(f"Saving LTA to standardized space: {sd.filename_by_attribute('cc_orient_volume_lta')}")
    lta.writeLTA(
        sd.filename_by_attribute("cc_orient_volume_lta"),
        orig2standardized_ras2ras,
        sd.conf_name,
        orig.header,
        sd.conf_name,
        orig.header,
    )

    for e in filter(lambda x: x and isinstance(x, Exception), (fut.exception() for fut in io_futures)):
        logger.exception(e)

    logger.info("CorpusCallosum analysis pipeline completed successfully")


if __name__ == "__main__":
    options = options_parse()

    # Set up logging if verbose mode is enabled
    logging.setup_logging(None, options.verbose)  # Log to stdout only

    main(
        conf_name=options.conf_name,
        aseg_name=options.aseg_name,
        subject_dir=options.subject_dir,
        slice_selection=options.slice_selection,
        qc_output_dir=options.qc_output_dir,
        num_thickness_points=options.num_thickness_points,
        subdivisions=options.subdivisions,
        subdivision_method=options.subdivision_method,
        contour_smoothing=options.contour_smoothing,
        save_template_dir=options.save_template_dir,
        device=options.device,
        upright_volume_path=options.upright_volume_path,
        segmentation_path=options.segmentation_path,
        postproc_results_path=options.postproc_results_path,
        cc_markers_path=options.cc_markers_path,
        upright_lta_path=options.upright_lta_path,
        orient_volume_lta_path=options.orient_volume_lta_path,
        surf_file_path=options.surf_file_path,
        overlay_file_path=options.overlay_file_path,
        cc_html_path=options.cc_html_path,
        vtk_file_path=options.vtk_file_path,
        orig_space_segmentation_path=options.orig_space_segmentation_path,
        qc_image_path=options.qc_image_path,
        thickness_image_path=options.thickness_image_path,
        softlabels_cc_path=options.softlabels_cc_path,
        softlabels_fn_path=options.softlabels_fn_path,
        softlabels_background_path=options.softlabels_background_path,
    )
