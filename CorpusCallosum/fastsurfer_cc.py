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
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter_ns
from typing import Literal, TypeVar, cast

import nibabel as nib
import numpy as np
import torch
from monai.networks.nets import DenseNet
from nibabel.freesurfer.mghformat import MGHHeader
from scipy.ndimage import affine_transform

from CorpusCallosum.data.constants import (
    CC_LABEL,
    DEFAULT_INPUT_PATHS,
    DEFAULT_OUTPUT_PATHS,
    FSAVERAGE_CENTROIDS_PATH,
    FSAVERAGE_DATA_PATH,
    FSAVERAGE_MIDDLE,
    THIRD_VENTRICLE_LABEL,
)
from CorpusCallosum.data.read_write import (
    MGHHeaderDict,
    calc_ras_centroids_from_seg,
    convert_numpy_to_json_serializable,
    load_fsaverage_centroids,
    load_fsaverage_data,
)
from CorpusCallosum.localization import inference as localization_inference
from CorpusCallosum.segmentation import inference as segmentation_inference
from CorpusCallosum.segmentation import segmentation_postprocessing
from CorpusCallosum.shape.postprocessing import (
    check_area_changes,
    make_subdivision_mask,
    offset_affine,
    recon_cc_surf_measures_multi,
)
from CorpusCallosum.utils.mapping_helpers import (
    apply_transform_to_pt,
    apply_transform_to_volume,
    calc_mapping_to_standard_space,
    map_softlabels_to_orig,
)
from CorpusCallosum.utils.types import CCMeasuresDict, SliceSelection, SubdivisionMethod
from FastSurferCNN.data_loader.conform import conform, is_conform
from FastSurferCNN.segstats import HelpFormatter
from FastSurferCNN.utils import (
    AffineMatrix4x4,
    Image3d,
    Image4d,
    Mask3d,
    Shape3d,
    Vector2d,
    logging,
    nibabelHeader,
    nibabelImage,
)
from FastSurferCNN.utils.arg_types import path_or_none
from FastSurferCNN.utils.common import SubjectDirectory, find_device
from FastSurferCNN.utils.lta import write_lta
from FastSurferCNN.utils.parallel import get_num_threads, serial_executor, shutdown_executors, thread_executor
from FastSurferCNN.utils.parser_defaults import modify_argument
from recon_surf.align_points import find_rigid

logger = logging.get_logger(__name__)

_TPathLike = TypeVar("_TPathLike", str, Path, Literal[None])


class ArgumentDefaultsHelpFormatter(HelpFormatter):
    """Help message formatter which adds default values to argument help."""

    def _get_help_string(self, action):
        """
        Add the default value to the option help message.
        """
        help = action.help
        if help is None:
            help = ''

        if "%(default)" not in help and not getattr(action, "required", False):
            if action.default is not argparse.SUPPRESS and not getattr(action.default, "DO_NOT_PRINT_DEFAULT", False):
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += " (not used by default)" if action.default is None else " (default: %(default)s)"
        return help


class _FixFloatFormattingList(list):
    def __init__(self, items: Iterable, item_format_spec: str):
        self._format_spec = item_format_spec
        super().__init__(items)

    def __str__(self):
        return "[" + ", ".join(map(lambda x: format(x, self._format_spec), self)) + "]"


def _do_not_print(value):
    class _DoNotPrintGeneric(type(value)):
        DO_NOT_PRINT_DEFAULT = True

    return _DoNotPrintGeneric(value)


def make_parser() -> argparse.ArgumentParser:
    """Create the argument parse object for the pipeline."""
    from FastSurferCNN.utils.parser_defaults import add_arguments

    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=_do_not_print(0),
        help="Enable verbose (pass twice for debug-output).",
    )
    # Specify subject directory + subject ID, OR specify individual MRI and segmentation files + output paths
    add_arguments(parser, ["sd", "sid", "conformed_name", "aseg_name", "device"])

    def _set_help_sid(action):
        action.help = "The subject id to use."
    modify_argument(parser, "--sid", _set_help_sid)

    parser.add_argument(
        "--num_thickness_points",
        type=int,
        default=100,
        help="Number of points for thickness estimation."
    )
    parser.add_argument(
        "--subdivisions",
        type=float,
        nargs='*',
        metavar="FRAC",
        default=_FixFloatFormattingList([1 / 6, 1 / 2, 2 / 3, 3 / 4], ".3f"),
        help="List of subdivision fractions for the corpus callosum subsegmentation."
          "The method allows for an arbitrary number of fractions."
          "By default it uses following Hofer-Frahms convention."
    )
    parser.add_argument(
        "--subdivision_method",
        default=_do_not_print("shape"),
        help="Method for contour subdivision. Options: <br>"
             "- shape (default): Intercallosal subdivision perpendicular to intercallosal line, <br>"
             "- vertical: orthogonal to the most anterior and posterior points in the AC/PC standardized CC contour, "
             "<br>"
             "- angular: subdivision based on equally spaced angles, as proposed by Hampel and colleagues, <br>"
             "- eigenvector: primary direction, same as FreeSurfers mri_cc.",
        choices=["shape", "vertical", "angular", "eigenvector"],
    )
    parser.add_argument(
        "--contour_smoothing",
        type=float,
        default=5,
        help="Gaussian sigma for smoothing during contour detection. Higher values mean a smoother CC outline, at the "
             "cost of precision.",
    )
    def _slice_selection(a: str) -> SliceSelection:
        if (b := a.lower()) in ("middle", "all"):
            return b
        return int(a)
    parser.add_argument(
        "--slice_selection",
        type=_slice_selection,
        default=_do_not_print("all"),
        help="Which slices to process. Options: 'middle', 'all' (default), or a specific slice number.",
    )

    ######## OUTPUT PATHS #########
    # 4. Options for advanced, technical parameters
    advanced = parser.add_argument_group(
        title="Advanced options",
        description="Custom output paths, useful if no standard case directory is used. Relative paths are always "
                    "relative to the subject_dir defined via --sd and --sid!",
    )
    add_arguments(advanced, ["threads"])
    advanced.add_argument(
        "--segmentation", "--seg",
        type=path_or_none,
        help="Output path for corpus callosum and fornix segmentation output.",
        default=Path(DEFAULT_OUTPUT_PATHS["segmentation"]),
    )
    advanced.add_argument(
        "--segmentation_in_orig",
        type=path_or_none,
        help="Output path for corpus callosum and fornix segmentation output in the input MRI space.",
        default=DEFAULT_OUTPUT_PATHS["segmentation_in_orig"],
    )
    advanced.add_argument(
        "--cc_measures",
        type=path_or_none,
        help="Output path for surface-based corpus callosum measures describing shape and volume for each image slice.",
        default=Path(DEFAULT_OUTPUT_PATHS["cc_measures"]),
    )
    advanced.add_argument(
        "--cc_mid_measures",
        type=path_or_none,
        help="Output path for surface-based corpus callosum measures of the midslice describing CC shape and volume.",
        default=DEFAULT_OUTPUT_PATHS["cc_markers"],
    )
    advanced.add_argument(
        "--upright_lta",
        type=path_or_none,
        help="Output path for upright LTA transform. This makes sure the midplane is at 128 in LR direction, "
             "but no nodding correction is applied.",
        default=DEFAULT_OUTPUT_PATHS["upright_lta"],
    )
    advanced.add_argument(
        "--upright_volume",
        type=path_or_none,
        help="Output path for upright volume (input image with cc_up.lta applied).",
        default=None,
    )
    advanced.add_argument(
        "--orient_volume_lta",
        type=path_or_none,
        help="Output path for orientation volume LTA transform. This makes sure the midplane is the volume center, "
             "the anterior and posterior commisures are on the coordinate line, and the posterior commissure is "
             "at the origin - standardizing the head position.",
        default=DEFAULT_OUTPUT_PATHS["orient_volume_lta"],
    )
    advanced.add_argument(
        "--qc_image",
        type=path_or_none,
        help="Output path for QC visualization image.",
        default=DEFAULT_OUTPUT_PATHS["qc_image"],
    )
    advanced.add_argument(
        "--save_template_dir",
        type=path_or_none,
        help="Directory path where to save contours.txt and thickness_values.txt files. These files can be used to "
             "visualize the CC shape and volume with the cc_visualization.py script.",
        default=None,
    )
    advanced.add_argument(
        "--thickness_image",
        type=path_or_none,
        help="Output path for thickness image.",
        default=DEFAULT_OUTPUT_PATHS["thickness_image"],
    )
    advanced.add_argument(
        "--surf",
        dest="cc_surf",
        type=path_or_none,
        help="Output path for surf file for visualization in freeview, use --save_template_dir and contours.txt to "
             "obtain source CC contours.",
        default=DEFAULT_OUTPUT_PATHS["cc_surf"],
    )
    advanced.add_argument(
        "--thickness_overlay",
        type=path_or_none,
        help="Output path for corpus callosum thickness overlay file for visualization in freeview, use "
             "--save_template_dir and thickness_values.txt to obtain source CC thickness values.",
        default=DEFAULT_OUTPUT_PATHS["cc_thickness_overlay"],
    )
    advanced.add_argument(
        "--cc_interactive_html", "--cc_html",
        dest="cc_html",
        type=path_or_none,
        help="Output path to the corpus callosum interactive 3D visualization HTML file.",
        default=DEFAULT_OUTPUT_PATHS["cc_html"],
    )
    advanced.add_argument(
        "--cc_surf_vtk",
        type=path_or_none,
        help=f"Output path for vtk file, showing the CC 3D mesh for visualization, use --save_template_dir and "
             f"contours.txt to obtain source CC contours. Example: {DEFAULT_OUTPUT_PATHS['cc_surf_vtk']}.",
        default=None,
    )
    advanced.add_argument(
        "--softlabels_cc",
        type=path_or_none,
        help=f"Output path for corpus callosum softlabels, which contains the soft labels of each voxel. "
             f"Example: {DEFAULT_OUTPUT_PATHS['softlabels_cc']}.",
        default=None,
    )
    advanced.add_argument(
        "--softlabels_fn",
        type=path_or_none,
        help=f"Output path for fornix softlabels, which contains the soft labels of each voxel. "
             f"Example: {DEFAULT_OUTPUT_PATHS['softlabels_fn']}.",
        default=None,
    )
    advanced.add_argument(
        "--softlabels_background",
        type=path_or_none,
        help=f"Output path for background softlabels, which contains the probability of each voxel. "
             f"Example: {DEFAULT_OUTPUT_PATHS['softlabels_background']}.",
        default=None,
    )
    ############ END OF OUTPUT PATHS ############
    return parser


def options_parse() -> argparse.Namespace:
    """Parse command line arguments for the pipeline."""
    parser = make_parser()
    args = parser.parse_args()

    # Reconstruct subject_dir from sd and sid (but sd might be stored as out_dir by parser_defaults)
    sd_value = getattr(args, 'out_dir', None)
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
        # Create standard FreeSurfer subdirectories
        if not args.conf_name:
            args.conf_name = args.subject_dir / DEFAULT_INPUT_PATHS["conf_name"]

        if not args.aseg_name:
            args.aseg_name = args.subject_dir / DEFAULT_INPUT_PATHS["aseg_name"]
    else:
        print("WARNING: Not providing subject_dir leads to discarding of files with relative paths!")
        args.subject_dir = None
        for arg, path in (("--aseg_name", args.aseg_name), ("--conformed_name", args.conf_name)):
            if path is None or not Path(path).is_absolute():
                parser.error(
                    f"When not passing --sd <path>, arguments of --aseg_name and --conformed_name must be "
                    f"absolute! But the argument passed to {arg} was {path}, i.e. not absolute."
                )

        all_paths = ("segmentation", "segmentation_in_orig", "cc_measures", "upright_lta", "orient_volume_lta",
                     "cc_surf", "softlabels_cc", "softlabels_fn", "softlabels_background", "cc_mid_measures",
                     "thickness_overlay", "qc_image", "thickness_image", "cc_html")

        warnings_paths = []
        # Create parent directories for all output paths
        for path_name in all_paths:
            path: Path | None = getattr(args, path_name, None)
            if isinstance(path, Path) and not args.subject_dir and not path.is_absolute():
                # set path to none in arguments
                warnings_paths.append(path_name)
                setattr(args, path_name, None)
        if warnings_paths:
            _warnings_paths = "' '".join(warnings_paths)
            print(f"WARNING: Not writing '{_warnings_paths}', because --sd and --sid are not specified and "
                  f"its paths are relative.")
    return args


def register_centroids_to_fsavg(aseg_nib: nibabelImage) \
        -> tuple[AffineMatrix4x4, AffineMatrix4x4, AffineMatrix4x4, MGHHeaderDict]:
    """Perform centroid-based registration between subject and fsaverage space.

    Computes a rigid transformation between the subject's segmentation and fsaverage space
    by aligning centroids of corresponding anatomical structures.

    Parameters
    ----------
    aseg_nib : nibabel.analyze.SpatialImage
        Subject's segmentation image.

    Returns
    -------
    aseg2fsaverage_vox2vox : AffineMatrix4x4
        Transformation matrix from original to fsaverage voxel space.
    aseg2fsaverage_ras2ras : AffineMatrix4x4
        Transformation matrix from original to fsaverage RAS space.
    fsaverage_hires_vox2ras : AffineMatrix4x4
        High-resolution fsaverage affine matrix.
    fsaverage_header : MGHHeaderDict
        FSAverage header fields for LTA writing.

    Notes
    -----
    The function uses pre-computed fsaverage centroids and data from static files
    to perform the registration. It matches corresponding anatomical structures
    between the subject's segmentation and fsaverage space.
    """
    logger.info("Starting centroid registration")

    # Load pre-computed fsaverage centroids and data from static files
    fsaverage_data_future = thread_executor().submit(load_fsaverage_data, FSAVERAGE_DATA_PATH)
    ras_centroids_dst = load_fsaverage_centroids(FSAVERAGE_CENTROIDS_PATH)

    ras_centroids_mov = calc_ras_centroids_from_seg(aseg_nib, label_ids=list(ras_centroids_dst.keys()))

    # get the set of joint labels
    joint_centroid_labels = [lbl for lbl, v in ras_centroids_mov.items() if v is not None]

    ras_centroids_mov = np.array([ras_centroids_mov[lbl] for lbl in joint_centroid_labels]).T
    ras_centroids_dst = np.array([ras_centroids_dst[lbl] for lbl in joint_centroid_labels]).T

    aseg2fsaverage_ras2ras: AffineMatrix4x4 = find_rigid(p_mov=ras_centroids_mov.T, p_dst=ras_centroids_dst.T)

    # make affine that increases resolution to orig resolution
    aseg_zooms_ras = np.asarray(nib.as_closest_canonical(aseg_nib).header.get_zooms()[:3])
    resolution_trans: AffineMatrix4x4 = np.diagflat(np.append(aseg_zooms_ras[[0, 2, 1]], [1])).astype(float)

    fsaverage_vox2ras, fsavg_header = fsaverage_data_future.result()
    fsavg_header["delta"] = aseg_zooms_ras[[0, 2, 1]] # vox sizes in lia
    # fsavg_hires_vox2ras translation should be 128 always (independent of resolution)
    fsavg_hires_vox2ras: AffineMatrix4x4 = np.concatenate(
        [(resolution_trans @ fsaverage_vox2ras)[:, :3], fsaverage_vox2ras[:, 3:4]],
        axis=1,
    )
    fsavg_header["dims"] = np.ceil(fsavg_header["dims"] @ np.linalg.inv(resolution_trans[:3, :3])).astype(int).tolist()

    # Correct fsavg_header["Pxyz_c"] by (vox_size - 1) / 2 in all three directions, because Pxyz_c is not actually in
    # the center of the image, but in the center of the voxel in increasing voxel index direction, i.e. index 128 for a
    # 256 image (where the center would be at 127.5).
    fsavg_header["Pxyz_c"] += (aseg_zooms_ras - 1) / 2 @ fsavg_header["Mdc"]

    aseg2fsavg_vox2vox: AffineMatrix4x4 = np.linalg.inv(fsavg_hires_vox2ras) @ aseg2fsaverage_ras2ras @ aseg_nib.affine
    logger.info("Centroid registration successful!")
    return aseg2fsavg_vox2vox, aseg2fsaverage_ras2ras, fsavg_hires_vox2ras, fsavg_header


def localize_ac_pc(
    orig_data: Image3d,
    aseg_nib: nibabelImage,
    orig2midslice_vox2vox: AffineMatrix4x4,
    model_localization: DenseNet,
    resample_shape: Shape3d,
) -> tuple[Vector2d, Vector2d]:
    """Localize anterior and posterior commissure points in the brain.

    Uses a trained model to detect AC and PC points in mid-sagittal slices,
    using the third ventricle as an anatomical reference.

    Parameters
    ----------
    orig_data : np.ndarray
        Array of intensity data.
    aseg_nib : nibabelImage
        Subject's segmentation image in native subject space.
    orig2midslice_vox2vox : np.ndarray
        Transformation matrix from subject/native space to fsaverage space (in lia).
    model_localization : DenseNet
        Trained model for AC-PC detection.
    resample_shape : 3-tuple of ints
        Number of slices to process.

    Returns
    -------
    ac_coords : np.ndarray
        AC voxel coordinates with shape (2,) containing its [y,x] positions.
    pc_coords : np.ndarray
        PC voxel coordinates with shape (2,) containing its [y,x] positions.
    """
    num_slices_to_analyze = resample_shape[0]
    resample_shape = (num_slices_to_analyze + 2,) + resample_shape[1:] # 2 for context slices
    _midslices_fut = thread_executor().submit(
        affine_transform,
        orig_data,
        np.linalg.inv(orig2midslice_vox2vox), # inverse is required for affine_transform
        output_shape=resample_shape,
        order=2, # unclear, why this is not order=3
        mode="constant",
        cval=0,
        prefilter=True, # unclear, why we are using a smoothing filter here
    )

    # get center of third ventricle from aseg and map to fsaverage space (voxel coordinates)
    third_ventricle_mask = np.asarray(aseg_nib.dataobj) == THIRD_VENTRICLE_LABEL
    third_ventricle_center = np.argwhere(third_ventricle_mask).mean(axis=0)
    third_ventricle_center_vox = apply_transform_to_pt(third_ventricle_center, orig2midslice_vox2vox, inv=False)

    # get 5 mm of slices with 3 slices per inference (cropping num_slices_to_analyze + 2 slices around the center)
    ac_coords, pc_coords = localization_inference.run_inference_on_slice(
        model_localization, _midslices_fut.result(), third_ventricle_center_vox[1:],
    )

    return ac_coords, pc_coords


def segment_cc(
    midslices: Image3d,
    ac_coords: Vector2d,
    pc_coords: Vector2d,
    aseg_nib: nibabelImage,
    model_segmentation: "torch.nn.Module",
) -> tuple[Mask3d, Image4d]:
    """Segment the corpus callosum using a trained model.

    Performs corpus callosum segmentation on mid-sagittal slices using a trained model, with AC-PC points as anatomical
    references. Includes post-processing to clean the cc_seg_labels.

    Parameters
    ----------
    midslices : np.ndarray
        Array of mid-sagittal slices in upright space and LIA-orientation.
    ac_coords : np.ndarray
        AC voxel coordinates with shape (2,) containing its [y,x] positions.
    pc_coords : np.ndarray
        PC voxel coordinates with shape (2,) containing its [y,x] positions.
    aseg_nib : nibabelImage
        Subject's cc_seg_labels image.
    model_segmentation : torch.nn.Module
        Trained model for CC cc_seg_labels.

    Returns
    -------
    cc_seg_labels : np.ndarray
        Binary cc_seg_labels of the corpus callosum in upright space and LIA-orientation.
    cc_softlabels : np.ndarray
        Soft cc_seg_labels probabilities of shape in upright space and LIA-orientation (H, W, D, C=3).
    """
    pre_clean_segmentation, inputs, cc_softlabels = segmentation_inference.run_inference_on_slice(
        model_segmentation,
        midslices,
        ac_center=ac_coords,
        pc_center=pc_coords,
        voxel_size=nib.as_closest_canonical(aseg_nib).header.get_zooms()[2:0:-1],  # convert from RAS to LIA
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
    num_thickness_points: int = 100,
    subdivisions: list[float] | None = None,
    subdivision_method: SubdivisionMethod = "shape",
    contour_smoothing: int = 5,
    save_template_dir: str | Path | None = None,
    device: str | torch.device = "auto",
    upright_volume: str | Path | None = None,
    segmentation: str | Path | None = None,
    cc_measures: str | Path | None = None,
    cc_mid_measures: str | Path | None = None,
    upright_lta: str | Path | None = None,
    orient_volume_lta: str | Path | None = None,
    cc_surf: str | Path | None = None,
    cc_thickness_overlay: str | Path | None = None,
    cc_html: str | Path | None = None,
    cc_surf_vtk: str | Path | None = None,
    segmentation_in_orig: str | Path | None = None,
    qc_image: str | Path | None = None,
    thickness_image: str | Path | None = None,
    softlabels_cc: str | Path | None = None,
    softlabels_fn: str | Path | None = None,
    softlabels_background: str | Path | None = None,
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
    num_thickness_points : int, default=100
        Number of points for thickness estimation.
    subdivisions : list[float], optional
        List of subdivision fractions for CC subsegmentation.
    subdivision_method : any of "shape", "vertical", "angular", "eigenvector", default="shape"
        Method for contour subdivision.
    contour_smoothing : int, default=5
        Gaussian sigma for smoothing during contour detection.
    save_template_dir : str or Path, optional
        Directory path where to save contours.txt and thickness_values.txt files. These files can be used to visualize
        the CC shape and volume in 3D. Files are only saved, if a valid directory path is passed.
    device : str, default="auto"
        Device to run inference on ('auto', 'cpu', 'cuda', or 'cuda:X').
    upright_volume : str or Path, optional
        Path to save upright volume.
    segmentation : str or Path, optional
        Path to save segmentation.
    cc_measures : str or Path, optional
        Path to save post-processing results.
    cc_mid_measures : str or Path, optional
        Path to save CC markers.
    upright_lta : str or Path, optional
        Path to save upright LTA transform.
    orient_volume_lta : str or Path, optional
        Path to save orientation transform.
    cc_surf : str or Path, optional
        Path to save surface file.
    cc_thickness_overlay : str or Path, optional
        Path to save overlay file.
    cc_html : str or Path, optional
        Path to save HTML visualization.
    cc_surf_vtk : str or Path, optional
        Path to save VTK file.
    segmentation_in_orig : str or Path, optional
        Path to save segmentation in original space.
    qc_image : str or Path, optional
        Path to save QC images.
    thickness_image : str or Path, optional
        Path to save thickness visualization.
    softlabels_cc : str or Path, optional
        Path to save CC soft labels.
    softlabels_fn : str or Path, optional
        Path to save fornix soft labels.
    softlabels_background : str or Path, optional
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
    start = perf_counter_ns()

    import sys

    if subdivisions is None:
        subdivisions = [1 / 6, 1 / 2, 2 / 3, 3 / 4]

    subject_dir = Path("/dev/null/no-subject-dir" if subject_dir is None else subject_dir)

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
        upright_volume=upright_volume,
        cc_segmentation=segmentation,
        cc_measures=cc_measures,
        cc_mid_measures=cc_mid_measures,
        upright_lta=upright_lta,
        cc_orient_volume_lta=orient_volume_lta,
        cc_surf=cc_surf,
        cc_thickness_overlay=cc_thickness_overlay,
        cc_html=cc_html,
        cc_mesh=cc_surf_vtk,
        cc_orig_segfile=segmentation_in_orig,
        cc_qc_image=qc_image,
        cc_thickness_image=thickness_image,
        cc_softlabels_cc=softlabels_cc,
        cc_softlabels_fn=softlabels_fn,
        cc_softlabels_background=softlabels_background,
    )

    # Validate subdivision fractions
    if any(i < 0 or i > 1 for i in subdivisions):
        logger.error(f"Subdivision fractions must be between 0 and 1, but got: {subdivisions}")
        sys.exit(1)

    #### setup variables
    io_futures = []

    # load models
    device = find_device(device)
    logger.info(f"Using device: {device}")

    logger.info("Loading models")
    _model_localization = thread_executor().submit(localization_inference.load_model, device=device)
    _model_segmentation = thread_executor().submit(segmentation_inference.load_model, device=device)

    _aseg_fut = thread_executor().submit(nib.load, sd.filename_by_attribute("aseg_name"))
    orig = cast(nibabelImage, nib.load(sd.conf_name))

    # check that the image is conformed, i.e. isotropic 1mm voxels, 256^3 size, LIA orientation
    if not is_conform(orig, vox_size=None, img_size=None, orientation=None):
        logger.info("Internally conforming orig to soft-LIA.")
        orig = conform(orig, vox_size=None, img_size=None, orientation=None)

    # 5 mm around the midplane (guaranteed to be aligned RAS by as_closest_canonical)
    vox_size_ras: tuple[float, float, float] = nib.as_closest_canonical(orig).header.get_zooms()
    vox_size = vox_size_ras[0], vox_size_ras[2], vox_size_ras[1]  # convert from RAS to LIA
    slices_to_analyze = int(np.ceil(5 / vox_size[0]))
    # slices_to_analyze must be odd
    if slices_to_analyze % 2 == 0:
        slices_to_analyze += 1

    logger.info(
        f"Segmenting {slices_to_analyze} slices (5 mm width at {vox_size[0]:.3f} mm resolution, "
        "center around the mid-sagittal plane)"
    )

    aseg_img = cast(nibabelImage, _aseg_fut.result())

    if not np.allclose(aseg_img.affine, orig.affine):
        logger.error("Input MRI and segmentation are not aligned! Please check your input files.")
        sys.exit(1)

    logger.info("Performing centroid registration to fsaverage space")
    orig2fsavg_vox2vox, orig2fsavg_ras2ras, fsavg_vox2ras, _fsavg_header_dict = register_centroids_to_fsavg(aseg_img)
    fsavg_header = init_mgh_header(orig.header, _fsavg_header_dict)

    # start saving upright volume, this is the image in fsaverage space but not yet oriented via AC-PC
    if sd.has_attribute("upright_volume"):
        # upright == fsaverage-aligned
        # FIXME: upright currently does not get saved correctly
        io_futures.append(
            thread_executor().submit(
                apply_transform_to_volume,
                orig,
                orig2fsavg_vox2vox,
                save_vox2ras=fsavg_vox2ras,
                output_path=sd.filename_by_attribute("upright_volume"),
                output_size=fsavg_header["dims"][:3],
            )
        )

    # calculate affine for segmentation volume
    fsavg2midslice_vox2vox: AffineMatrix4x4 = offset_affine([-FSAVERAGE_MIDDLE / vox_size[0], 0, 0])
    orig2midslice_vox2vox = fsavg2midslice_vox2vox @ orig2fsavg_vox2vox

    # calculate vox2vox for input resampling volumes
    def _orig2midslab_vox2vox(additional_context: int = 0) -> AffineMatrix4x4:
        fsavg2midslab = offset_affine([slices_to_analyze // 2 + additional_context // 2, 0, 0])
        # first, orig->fsaverage, then fsaverage->midslab (all in vox2vox)
        return fsavg2midslab @ orig2midslice_vox2vox

    # first, midslice->fsaverage in vox2vox, then vox2ras in fsaverage space
    fsavg2midslab_vox2vox = offset_affine([slices_to_analyze // 2, 0, 0]) @ fsavg2midslice_vox2vox
    fsaverage_midslab_vox2ras: AffineMatrix4x4 = fsavg_vox2ras @ np.linalg.inv(fsavg2midslab_vox2vox)


    #### do localization and segmentation inference
    logger.info("Starting AC/PC localization")
    target_shape: tuple[int, int, int] = (slices_to_analyze, fsavg_header["dims"][1], fsavg_header["dims"][2])
    # predict ac and pc coordinates in upright AS space
    ac_coords_vox, pc_coords_vox = localize_ac_pc(
        np.asarray(orig.dataobj),
        aseg_img,
        _orig2midslab_vox2vox(additional_context=2),
        _model_localization.result(),
        target_shape,
    )
    logger.info("Starting corpus callosum segmentation")
    num_context = 8  # 8 extra in x-direction for context slices
    target_shape: Shape3d = (slices_to_analyze + num_context, fsavg_header["dims"][1], fsavg_header["dims"][2])
    midslices: Image3d = affine_transform(
        np.asarray(orig.dataobj),
        np.linalg.inv(_orig2midslab_vox2vox(additional_context=num_context)), # inverse is required for affine_transform
        output_shape=target_shape,
        order=2, # @ClePol unclear, why this is not order=3
        mode="constant",
        cval=0,
        prefilter=True, # unclear, why we are using a smoothing filter here
    )
    cc_fn_seg_labels, cc_fn_softlabels = segment_cc(
        midslices,
        ac_coords_vox,
        pc_coords_vox,
        aseg_img,
        _model_segmentation.result(),
    )

    # save segmentation softlabels
    for i, (attr, name) in enumerate((("background",) * 2, ("cc", "Corpus Callosum"), ("fn", "Fornix"))):
        if sd.has_attribute(f"cc_softlabels_{attr}"):
            logger.info(f"Saving {name} softlabels to {sd.filename_by_attribute(f'cc_softlabels_{attr}')}")
            io_futures.append(thread_executor().submit(
                nib.save,
                nib.MGHImage(cc_fn_softlabels[..., i], fsaverage_midslab_vox2ras, orig.header),
                sd.filename_by_attribute(f"cc_softlabels_{attr}"),
            ))

    # Create a temporary segmentation image with proper affine for enhanced postprocessing
    # Process slices based on selection mode

    logger.info(f"Processing slices with selection mode: {slice_selection}")
    slice_results, slice_io_futures = recon_cc_surf_measures_multi(
        segmentation=cc_fn_seg_labels,
        slice_selection=slice_selection,
        upright_header=fsavg_header,
        fsavg2midslab_vox2vox=fsavg2midslab_vox2vox,
        fsavg_vox2ras=fsavg_vox2ras,
        orig2fsavg_vox2vox=orig2fsavg_vox2vox,
        midslices=midslices,
        ac_coords_vox=ac_coords_vox,
        pc_coords_vox=pc_coords_vox,
        num_thickness_points=num_thickness_points,
        subdivisions=subdivisions,
        subdivision_method=cast(SubdivisionMethod, subdivision_method),
        contour_smoothing=contour_smoothing,
        vox_size=vox_size,
        subject_dir=sd,
    )
    io_futures.extend(slice_io_futures)

    outer_contours = [slice_result["split_contours"][0] for slice_result in slice_results]

    if len(outer_contours) > 1 and not check_area_changes(outer_contours):
        logger.warning(
            "Large area changes detected between consecutive slices, this is likely due to a segmentation error."
        )

    # Get middle slice result
    middle_slice_result: CCMeasuresDict = slice_results[len(slice_results) // 2]
    if len(middle_slice_result["split_contours"]) <= 5:
        cc_subseg_midslice = make_subdivision_mask(
            (cc_fn_seg_labels.shape[1], cc_fn_seg_labels.shape[2]),
            middle_slice_result["split_contours"],
            vox_size[1:3],
        )
    else:
        logger.warning("Too many subsegments for lookup table, skipping sub-division of output segmentation.")
        cc_subseg_midslice = None

    # save segmentation labels, this
    if sd.has_attribute("cc_segmentation"):
        io_futures.append(thread_executor().submit(
            nib.save,
            nib.MGHImage(cc_fn_seg_labels, fsaverage_midslab_vox2ras, orig.header),
            sd.filename_by_attribute("cc_segmentation"),
        ))
    # map soft labels to original space (in parallel because this takes a while, and we only do it to save the labels)
    if sd.has_attribute("cc_orig_segfile"):
        # if num_threads is not large enough (>1), this might be blocking ; serial_executor runs the function in submit
        executor = thread_executor() if get_num_threads() > 2 else serial_executor()
        io_futures.append(executor.submit(
            map_softlabels_to_orig,
            cc_fn_softlabels=cc_fn_softlabels,
            orig=orig,
            orig_space_segmentation_path=sd.filename_by_attribute("cc_orig_segfile"),
            orig2slab_vox2vox=_orig2midslab_vox2vox(),
            cc_subseg_midslice=cc_subseg_midslice,
            orig2midslice_vox2vox=orig2midslice_vox2vox,
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
        "slices": [convert_numpy_to_json_serializable({metric: result[metric] for metric in METRICS})
                   for result in slice_results],
    }

    ########## Save outputs ##########
    additional_metrics = {}
    if len(outer_contours) > 1:
        cc_volume_voxel = segmentation_postprocessing.get_cc_volume_voxel(
            desired_width_mm=5,
            cc_mask=np.equal(cc_fn_seg_labels, CC_LABEL),
            voxel_size=vox_size, # in LIA order
        )
        logger.info(f"CC volume voxel: {cc_volume_voxel}")
        # FIXME: Create a proper mesh and use cc_mesh.volume for this volume --> not closed, but move function to
        #  CCContour?
        try:
            cc_volume_contour = segmentation_postprocessing.get_cc_volume_contour(
                cc_contours=outer_contours,
                voxel_size=vox_size, # in LIA order
            )
            logger.info(f"CC volume contour: {cc_volume_contour}")
        except AssertionError as e:
            logger.warning("Could not compute CC volume from contours, setting to NaN")
            logger.exception(e)
            cc_volume_contour = float('nan')

        additional_metrics["cc_5mm_volume"] = cc_volume_voxel
        additional_metrics["cc_5mm_volume_pv_corrected"] = cc_volume_contour

    # get ac and pc in all spaces
    ac_coords_3d = np.hstack((FSAVERAGE_MIDDLE, ac_coords_vox))
    pc_coords_3d = np.hstack((FSAVERAGE_MIDDLE, pc_coords_vox))
    standardized2orig_vox2vox, ac_coords_standardized, pc_coords_standardized, ac_coords_orig, pc_coords_orig = (
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
    additional_metrics["voxel_size"] = np.asarray(orig.header.get_zooms(), dtype=float).tolist()
    additional_metrics["num_thickness_points"] = num_thickness_points
    additional_metrics["subdivision_method"] = subdivision_method
    additional_metrics["subdivision_ratios"] = subdivisions
    additional_metrics["contour_smoothing"] = contour_smoothing
    additional_metrics["slice_selection"] = slice_selection


    if sd.has_attribute("cc_mid_measures"):
        io_futures.append(thread_executor().submit(
            save_cc_measures_json,
            sd.filename_by_attribute('cc_mid_measures'),
            output_metrics_middle_slice | additional_metrics,
            ))

    if sd.has_attribute("cc_measures"):
        io_futures.append(thread_executor().submit(
            save_cc_measures_json,
            sd.filename_by_attribute("cc_measures"),
            per_slice_output_dict | additional_metrics,
            ))

    # save lta to fsaverage space

    if sd.has_attribute("upright_lta"):
        sd.filename_by_attribute("cc_mid_measures").parent.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving LTA to fsaverage space: {sd.filename_by_attribute('upright_lta')}")
        io_futures.append(thread_executor().submit(
            write_lta,
            sd.filename_by_attribute("upright_lta"),
            orig2fsavg_ras2ras,
            sd.filename_by_attribute("aseg_name"),
            aseg_img.header,
            "fsaverage",
            fsavg_header,
        ))

    if sd.has_attribute("cc_orient_volume_lta"):
        sd.filename_by_attribute("cc_orient_volume_lta").parent.mkdir(exist_ok=True, parents=True)
        # save lta to standardized space (fsaverage + nodding + ac to center)
        orig2standardized_ras2ras = orig.affine @ np.linalg.inv(standardized2orig_vox2vox) @ np.linalg.inv(orig.affine)
        logger.info(f"Saving LTA to standardized space: {sd.filename_by_attribute('cc_orient_volume_lta')}")
        io_futures.append(thread_executor().submit(
            write_lta,
            sd.filename_by_attribute("cc_orient_volume_lta"),
            orig2standardized_ras2ras,
            sd.conf_name,
            orig.header,
            sd.conf_name,
            orig.header,
        ))

    # this waits for all io to finish
    for fut in io_futures:
        e = fut.exception()
        if e and isinstance(e, Exception):
            logger.exception(e)
    shutdown_executors()

    duration = (perf_counter_ns() - start) / 1e9
    logger.info(f"CorpusCallosum analysis pipeline completed successfully in {duration:.2f} seconds.")


def init_mgh_header(header: nibabelHeader, header_dict: MGHHeaderDict) -> MGHHeader:
    """
    Generates a MGHHeader object from a header and a header dictionary.

    Parameters
    ----------
    header : nibabelHeader
        The header object used to initialize the generated header.
    header_dict : MGHHeaderDict
        A dictionary of values to overwrite in the generated header.

    Returns
    -------
    MGHHeader
        The header updated with values in header_dict.
    """
    new_header: MGHHeader = MGHHeader.from_header(header)
    if "dims" in header_dict:
        new_header["dims"] = np.append(header_dict["dims"], [1])
    for key in ("delta", "Pxyz_c", "Mdc"):
        if key in header_dict:
            new_header[key] = header_dict[key]
    return new_header


def save_cc_measures_json(cc_mid_measure_file: Path, metrics: dict[str, object]):
    """Save JSON metrics file."""
    # Convert numpy arrays to lists for JSON serialization
    logger.info(f"Saving CC markers to {cc_mid_measure_file}")
    cc_mid_measure_file.parent.mkdir(exist_ok=True, parents=True)
    with open(cc_mid_measure_file, "w") as f:
        json.dump(convert_numpy_to_json_serializable(metrics), f, indent=4)


if __name__ == "__main__":
    options = options_parse()

    # Set up logging if verbose mode is enabled
    logging.setup_logging(None, options.verbose)  # Log to stdout only

    main(
        conf_name=options.conf_name,
        aseg_name=options.aseg_name,
        subject_dir=options.subject_dir,
        slice_selection=options.slice_selection,
        num_thickness_points=options.num_thickness_points,
        subdivisions=list(options.subdivisions),
        subdivision_method=str(options.subdivision_method),
        contour_smoothing=options.contour_smoothing,
        save_template_dir=options.save_template_dir,
        device=options.device,
        upright_volume=options.upright_volume,
        segmentation=options.segmentation,
        cc_measures=options.cc_measures,
        cc_mid_measures=options.cc_mid_measures,
        upright_lta=options.upright_lta,
        orient_volume_lta=options.orient_volume_lta,
        cc_surf=options.cc_surf,
        cc_thickness_overlay=options.thickness_overlay,
        cc_html=options.cc_html,
        cc_surf_vtk=options.cc_surf_vtk,
        segmentation_in_orig=options.segmentation_in_orig,
        qc_image=options.qc_image,
        thickness_image=options.thickness_image,
        softlabels_cc=options.softlabels_cc,
        softlabels_fn=options.softlabels_fn,
        softlabels_background=options.softlabels_background,
    )
