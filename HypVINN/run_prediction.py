# Copyright 2024 AI in Medical Imaging, German Center for Neurodegenerative Diseases(DZNE), Bonn
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
# IMPORTS
from typing import TYPE_CHECKING, Optional, cast, Literal
import argparse
from pathlib import Path
from time import time

import numpy as np
from numpy import typing as npt
import torch

if TYPE_CHECKING:
    import yacs.config
    from nibabel.filebasedimages import FileBasedHeader

from FastSurferCNN.utils import PLANES, Plane, logging, parser_defaults
from FastSurferCNN.utils.checkpoint import (
    get_checkpoints,
    load_checkpoint_config_defaults,
)
from FastSurferCNN.utils.common import assert_no_root, SerialExecutor

from HypVINN.config.hypvinn_files import HYPVINN_SEG_NAME, HYPVINN_MASK_NAME
from HypVINN.data_loader.data_utils import hypo_map_label2subseg, rescale_image
from HypVINN.inference import Inference
from HypVINN.utils import ModalityDict, ModalityMode, ViewOperations
from HypVINN.utils.checkpoint import YAML_DEFAULT as CHECKPOINT_PATHS_FILE
from HypVINN.utils.img_processing_utils import save_segmentation
from HypVINN.utils.load_config import load_config
from HypVINN.utils.misc import create_expand_output_directory
from HypVINN.utils.mode_config import get_hypinn_mode
from HypVINN.utils.preproc import hypvinn_preproc
from HypVINN.utils.stats_utils import compute_stats
from HypVINN.utils.visualization_utils import plot_qc_images

logger = logging.get_logger(__name__)

##
# Input array preparation
##


def optional_path(a: Path | str) -> Optional[Path]:
    """
    Convert a string to a Path object or None.

    Parameters
    ----------
    a : Path, str
        The input to convert.

    Returns
    -------
    Path, optional
        The converted Path object.
    f"""
    if isinstance(a, Path):
        return a
    if a.lower() in ("none", ""):
        return None
    return Path(a)


def option_parse() -> argparse.ArgumentParser:
    """
    A function to create an ArgumentParser object and parse the command line arguments.

    Returns
    -------
    argparse.ArgumentParser
        The parser object to parse arguments from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Script for Hypothalamus Segmentation.",
    )

    # 1. Directory information (where to read from, where to write from and to incl. search-tag)
    parser = parser_defaults.add_arguments(
        parser, ["sd", "sid"],
    )

    parser = parser_defaults.add_arguments(parser, ["seg_log"])

    # 2. Options for the MRI volumes
    parser = parser_defaults.add_arguments(
        parser, ["t1"]
    )
    parser.add_argument(
        '--t2',
        type=optional_path,
        default=None,
        required=False,
        help="Path to the T2 image to process.",
    )

    # 3. Image processing options
    parser.add_argument(
        "--qc_snap",
        default='store_true',
        dest="qc_snapshots",
        help="Create qc snapshots in <sd>/<sid>/qc_snapshots.",
    )
    parser.add_argument(
        "--reg_mode",
        type=str,
        default="coreg",
        choices=["none", "coreg", "robust"],
        help="Freesurfer Registration type to run. coreg: mri_coreg, "
             "robust : mri_robust_register, none: entirely deactivates "
             "registration of T2 to T1, if both images are passed, "
             "images need to be register properly externally.",
    )
    default_hypo_segfile = Path("mri") / HYPVINN_SEG_NAME
    parser.add_argument(
        "--hypo_segfile",
        type=Path,
        default=default_hypo_segfile,
        dest="hypo_segfile",
        help=f"File pattern on where to save the hypothalamus segmentation file "
             f"(default: {default_hypo_segfile})."
    )

    # 4. Options for advanced, technical parameters
    advanced = parser.add_argument_group(title="Advanced options")
    parser_defaults.add_arguments(
        advanced,
        ["device", "viewagg_device", "threads", "batch_size", "async_io", "allow_root"],
    )

    files: dict[Plane, str | Path] = {k: "default" for k in PLANES}
    # 5. Checkpoint to load
    parser_defaults.add_plane_flags(
        advanced,
        "checkpoint",
        files,
        CHECKPOINT_PATHS_FILE,
    )

    parser_defaults.add_plane_flags(
        advanced,
        "config",
        {
            "coronal": Path("HypVINN/config/HypVINN_coronal_v1.1.0.yaml"),
            "axial": Path("HypVINN/config/HypVINN_axial_v1.1.0.yaml"),
            "sagittal": Path("HypVINN/config/HypVINN_sagittal_v1.1.0.yaml"),
        },
        CHECKPOINT_PATHS_FILE,
    )
    return parser


def main(
        out_dir: Path,
        t2: Optional[Path],
        orig_name: Optional[Path],
        sid: str,
        ckpt_ax: Path,
        ckpt_cor: Path,
        ckpt_sag: Path,
        cfg_ax: Path,
        cfg_cor: Path,
        cfg_sag: Path,
        hypo_segfile: str = HYPVINN_SEG_NAME,
        hypo_maskfile: str = HYPVINN_MASK_NAME,
        allow_root: bool = False,
        qc_snapshots: bool = False,
        reg_mode: Literal["coreg", "robust", "none"] = "coreg",
        threads: int = -1,
        batch_size: int = 1,
        async_io: bool = False,
        device: str = "auto",
        viewagg_device: str = "auto",
) -> int | str:
    f"""
    Main function of the hypothalamus segmentation module.

    Parameters
    ----------
    out_dir : Path
        The output directory where the results will be stored.
    t2 : Path, optional
        The path to the T2 image to process.
    orig_name : Path, optional
        The path to the T1 image to process or FastSurfer orig image.
    sid : str
        The subject ID.
    ckpt_ax : Path
        The path to the axial checkpoint file.
    ckpt_cor : Path
        The path to the coronal checkpoint file.
    ckpt_sag : Path
        The path to the sagittal checkpoint file.
    cfg_ax : Path
        The path to the axial configuration file.
    cfg_cor : Path
        The path to the coronal configuration file.
    cfg_sag : Path
        The path to the sagittal configuration file.
    hypo_segfile : str, default="{HYPVINN_SEG_NAME}"
        The name of the hypothalamus segmentation file. Default is {HYPVINN_SEG_NAME}.
    hypo_maskfile : str, default="{HYPVINN_MASK_NAME}"
        The name of the hypothalamus mask file. Default is {HYPVINN_MASK_NAME}.
    allow_root : bool, default=False
        Whether to allow running as root user. Default is False.
    qc_snapshots : bool, optional
        Whether to create QC snapshots. Default is False.
    reg_mode : "coreg", "robust", "none", default="coreg"
        The registration mode to use. Default is "coreg".
    threads : int, default=-1
        The number of threads to use. Default is -1, which uses all available threads.
    batch_size : int, default=1
        The batch size to use. Default is 1.
    async_io : bool, default=False
        Whether to use asynchronous I/O. Default is False.
    device : str, default="auto"
        The device to use. Default is "auto", which automatically selects the device.
    viewagg_device : str, default="auto"
        The view aggregation device to use. Default is "auto", which automatically 
        selects the device.

    Returns
    -------
    int, str
        0, if successful, an error message describing the cause for the
        failure otherwise.
    """
    from concurrent.futures import ProcessPoolExecutor, Future
    if threads != 1:
        pool = ProcessPoolExecutor(threads)
    else:
        pool = SerialExecutor()
    prep_tasks: dict[str, Future] = {}

    # mapped freesurfer orig input name to the hypvinn t1 name
    t1_path = orig_name
    t2_path = t2
    subject_name = sid
    subject_dir = out_dir / sid
    # Warning if run as root user
    allow_root or assert_no_root()
    start = time()
    try:
        # Set up logging
        prep_tasks["cp"] = pool.submit(prepare_checkpoints, ckpt_ax, ckpt_cor, ckpt_sag)

        kwargs = {}
        if t1_path is not None:
            kwargs["t1_path"] = Path(t1_path)
        if t2_path:
            kwargs["t2_path"] = Path(t2_path)
        # Get configuration to run multi-modal or uni-modal
        mode = get_hypinn_mode(**kwargs)

        if not mode:
            return (
                f"Failed Evaluation on {subject_name} couldn't determine the "
                f"processing mode. Please check that T1 or T2 images are "
                f"available.\nT1 image path: {t1_path}\nT2 image path "
                f"{t2_path}.\nNo T1 or T2 image available."
            )

        # Create output directory if it does not already exist.
        create_expand_output_directory(subject_dir, qc_snapshots)
        logger.info(
            f"Running HypVINN segmentation pipeline on subject {sid}"
        )
        logger.info(f"Output will be stored in: {subject_dir}")
        logger.info(f"T1 image input {t1_path}")
        logger.info(f"T2 image input {t2_path}")

        # Pre-processing -- T1 and T2 registration
        if mode == "t1t2":
            # Note, that t1_path and t2_path are guaranteed to be not None
            # via get_hypvinn_mode, which only returns t1t2, if t1 and t2
            # exist.
            # hypvinn_preproc returns the path to the t2 that is registered
            # to the t1
            prep_tasks["reg"] = pool.submit(
                hypvinn_preproc,
                mode,
                reg_mode,
                subject_dir=Path(subject_dir),
                threads=threads,
                **kwargs,
            )

        # Segmentation pipeline
        seg = time()
        view_ops: ViewOperations = {a: None for a in PLANES}
        logger.info("Setting up HypVINN run")

        cfgs = (cfg_ax, cfg_cor, cfg_sag)
        ckpts = (ckpt_ax, ckpt_cor, ckpt_sag)
        for plane, _cfg_file, _ckpt_file in zip(PLANES, cfgs, ckpts):
            logger.info(f"{plane} model configuration from {_cfg_file}")
            view_ops[plane] = {
                "cfg": set_up_cfgs(_cfg_file, subject_dir, batch_size),
                "ckpt": _ckpt_file,
            }

            model = view_ops[plane]["cfg"].MODEL
            if mode != model.MODE and "HypVinn" not in model.MODEL_NAME:
                raise AssertionError(
                    f"Modality mode different between input arg: "
                    f"{mode} and axial train cfg: {model.MODE}"
                )

        cfg_fin, ckpt_fin = view_ops["coronal"].values()

        if "reg" in prep_tasks:
            t2_path = prep_tasks["reg"].result()
            kwargs["t2_path"] = t2_path
        prep_tasks["load"] = pool.submit(load_volumes, mode=mode, **kwargs)

        # Set up model
        model = Inference(
            cfg=cfg_fin,
            async_io=async_io,
            threads=threads,
            viewagg_device=viewagg_device,
            device=device,
        )

        logger.info('----' * 30)
        logger.info(f"Evaluating hypothalamus model on {subject_name}")

        # wait for all prep tasks to finish
        for ptask in prep_tasks.values():
            if e := ptask.exception():
                raise e

        # Load  Images
        image_data, affine, header, orig_zoom, orig_size = prep_tasks["load"].result()
        logger.info(f"Scale factor: {orig_zoom}")

        pred = time()
        pred_classes = get_prediction(
            subject_name,
            image_data,
            orig_zoom,
            model,
            target_shape=orig_size,
            view_opts=view_ops,
            out_scale=None,
            mode=mode,
        )
        logger.info(f"Model prediction finished in {time() - pred:0.4f} seconds")
        logger.info(f"Saving results in {subject_dir}")

        if mode == 't1t2' or mode == 't1':
            orig_path = t1_path
        else:
            orig_path = t2_path

        time_needed = save_segmentation(
            pred_classes,
            orig_path=orig_path,
            ras_affine=affine,
            ras_header=header,
            subject_dir=subject_dir,
            seg_file=hypo_segfile,
            mask_file=hypo_maskfile,
            save_mask=True,
        )
        logger.info(f"Prediction successfully saved in {time_needed} seconds.")
        if qc_snapshots:
            qc_future: Optional[Future] = pool.submit(
                plot_qc_images,
                subject_qc_dir=subject_dir / "qc_snapshots",
                orig_path=orig_path,
                prediction_path=Path(hypo_segfile),
            )
            qc_future.add_done_callback(
                lambda x: logger.info(f"QC snapshots saved in {x.result()} seconds."),
            )
        else:
            qc_future = None

        logger.info("Computing stats")
        return_value = compute_stats(
            orig_path=orig_path,
            prediction_path=Path(hypo_segfile),
            stats_dir=subject_dir / "stats",
            threads=threads,
        )
        if return_value != 0:
            logger.error(return_value)

        logger.info(
            f"Processing segmentation finished in {time() - seg:0.4f} seconds."
        )
    except (FileNotFoundError, RuntimeError) as e:
        logger.info(f"Failed Evaluation on {subject_name}:")
        logger.exception(e)
    else:
        if qc_future:
            # finish qc
            qc_future.result()

        logger.info(
            f"Processing whole pipeline finished in {time() - start:.4f} seconds."
        )


def prepare_checkpoints(ckpt_ax, ckpt_cor, ckpt_sag):
    """
    Prepare the checkpoints for the Hypothalamus Segmentation model.

    This function checks if the checkpoint files for the axial, coronal, and sagittal planes exist.
    If they do not exist, it downloads them from the default URLs specified in the configuration file.

    Parameters
    ----------
    ckpt_ax : str
        The path to the axial checkpoint file.
    ckpt_cor : str
        The path to the coronal checkpoint file.
    ckpt_sag : str
        The path to the sagittal checkpoint file.
    """
    logger.info("Checking or downloading default checkpoints ...")
    urls = load_checkpoint_config_defaults(
        "url",
        filename=CHECKPOINT_PATHS_FILE,
    )
    get_checkpoints(ckpt_ax, ckpt_cor, ckpt_sag, urls=urls)


def load_volumes(
        mode: ModalityMode,
        t1_path: Optional[Path] = None,
        t2_path: Optional[Path] = None,
) -> tuple[
    ModalityDict,
    npt.NDArray[float],
    "FileBasedHeader",
    tuple[float, float, float],
    tuple[int, int, int],
]:
    """
    Load the volumes of T1 and T2 images.

    This function loads the T1 and T2 images, checks their compatibility based on the mode, and returns the loaded
    volumes along with their affine transformations, headers, zoom levels, and sizes.

    Parameters
    ----------
    mode : ModalityMode
        The mode of operation. Can be 't1', 't2', or 't1t2'.
    t1_path : Path, optional
        The path to the T1 image. Default is None.
    t2_path : Path, optional
        The path to the T2 image. Default is None.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - modalities: A dictionary with keys 't1' and/or 't2' and values being the corresponding loaded and rescaled images.
        - affine: The affine transformation of the loaded image(s).
        - header: The header of the loaded image(s).
        - zoom: The zoom level of the loaded image(s).
        - size: The size of the loaded image(s).

    Raises
    ------
    RuntimeError
        If the mode is inconsistent with the provided image paths, or if the number of dimensions of the data is invalid.
    ValueError
        If the mode is invalid, or if a header is missing.
    AssertionError
        If the mode is 't1t2' but the T1 and T2 images have different resolutions or sizes.
    """
    import nibabel as nib
    modalities: ModalityDict = {}

    t1_size = ()
    t2_size = ()
    t1_zoom = ()
    t2_zoom = ()
    affine: npt.NDArray[float] = np.ndarray([0])
    header: Optional["FileBasedHeader"] = None
    zoom: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[int, ...] = (0, 0, 0)

    if t1_path:
        logger.info(f'Loading T1 image from : {t1_path}')
        t1 = nib.load(t1_path)
        t1 = nib.as_closest_canonical(t1)
        if mode in ('t1t2', 't1'):
            affine = t1.affine
            header = t1.header
        else:
            raise RuntimeError(f"Invalid mode {mode}, or inconsistent with t1_path!")
        t1_zoom = t1.header.get_zooms()
        zoom = cast(tuple[float, float, float], tuple(np.round(t1_zoom, 3)))
        # Conform Intensities
        modalities["t1"] = rescale_image(np.asarray(t1.dataobj))
        t1_size: tuple[int, ...] = modalities["t1"].shape
        size = t1_size
    if t2_path:
        logger.info(f"Loading T2 image from {t2_path}")
        t2 = nib.load(t2_path)
        t2 = nib.as_closest_canonical(t2)
        t2_zoom = t2.header.get_zooms()
        if mode == "t2":
            affine = t2.affine
            header = t2.header
            zoom = cast(tuple[float, float, float], tuple(np.round(t2_zoom, 3)))
        elif mode == "t1t2":
            pass
        else:
            raise RuntimeError(f"Invalid mode {mode}, or inconsistent with t2_path!")
        # Conform Intensities
        modalities["t2"] = np.asarray(rescale_image(t2.get_fdata()), dtype=np.uint8)
        t2_size = modalities["t2"].shape
        size = t2_size

    if mode == "t1t2":
        if not np.allclose(np.array(t1_zoom), np.array(t2_zoom), rtol=0.05):
            raise AssertionError(
                f"T1 {t1_zoom} and T2 {t2_zoom} images have different resolutions!"
            )
        if not np.allclose(np.array(t1_size), np.array(t2_size), rtol=0.05):
            raise AssertionError(
                f"T1 {t1_size} and T2 {t2_size} images have different size!"
            )
    elif mode not in ("t1", "t2"):
        raise ValueError(f"Invalid mode {mode}, vs. 't1', 't2', 't1t2'")

    if header is None:
        raise ValueError("Missing a header!")
    if len(size) != 3:
        raise RuntimeError("Invalid ndims of data!")
    _size = cast(tuple[int, int, int], size)

    return modalities, affine, header, zoom, _size


def get_prediction(
        subject_name: str,
        modalities: ModalityDict,
        orig_zoom,
        model: Inference,
        target_shape: tuple[int, int, int],
        view_opts: ViewOperations,
        out_scale=None,
        mode: ModalityMode = "t1t2",
) -> npt.NDArray[int]:
    """
    Run the prediction for the Hypothalamus Segmentation model.

    This function sets up the prediction process for the Hypothalamus Segmentation model. It runs the model for each
    plane (axial, coronal, sagittal), accumulates the prediction probabilities, and then generates the final prediction.

    Parameters
    ----------
    subject_name : str
        The name of the subject.
    modalities : ModalityDict
        A dictionary containing the modalities (T1 and/or T2) and their corresponding images.
    orig_zoom : npt.NDArray[float]
        The original zoom of the subject.
    model : Inference
        The Inference object of the model.
    target_shape : tuple[int, int, int]
        The target shape of the output prediction.
    view_opts : ViewOperations
        A dictionary containing the configurations for each plane.
    out_scale : optional
        The output scale. Default is None.
    mode : ModalityMode, default="t1t2"
        The mode of operation. Can be 't1', 't2', or 't1t2'. Default is 't1t2'.

    Returns
    -------
    pred_classes: npt.NDArray[int]
        The final prediction of the model.
    """
    # TODO There are probably several possibilities to accelerate this script.
    #  FastSurferVINN takes 7-8s vs. HypVINN 10+s per slicing direction.
    #  Solution: make this script/function more similar to the optimized FastSurferVINN
    device, viewagg_device = model.get_device()
    dim = model.get_max_size()

    pred_shape = (dim, dim, dim, model.get_num_classes())
    # Set up tensor to hold probabilities and run inference
    pred_prob = torch.zeros(pred_shape, dtype=torch.float, device=viewagg_device)
    for plane, opts in view_opts.items():
        logger.info(f"Evaluating {plane} model, cpkt :{opts['ckpt']}")
        model.set_model(opts["cfg"])
        model.load_checkpoint(opts["ckpt"])
        pred_prob += model.run(subject_name, modalities, orig_zoom, pred_prob, out_scale, mode=mode)

    # Post processing
    h, w, d = target_shape  # final prediction shape equivalent to input ground truth shape

    if np.any(target_shape < pred_prob.shape[:3]):
        # if orig was padded before running through model (difference in
        # aseg_size and pred_shape), select slices of interest only.
        # This currently works only for "top_left" padding (see augmentation)
        pred_prob = pred_prob[0:h, 0:w, 0:d, :]

    # Get hard predictions and map to freesurfer label space
    _, pred_classes = torch.max(pred_prob, 3)
    del pred_prob
    pred_classes = pred_classes.cpu().numpy()
    pred_classes = hypo_map_label2subseg(pred_classes)

    return pred_classes


##
# Processing
##
def set_up_cfgs(
        cfg: "yacs.config.CfgNode",
        out_dir: Path,
        batch_size: int = 1,
) -> "yacs.config.CfgNode":
    """
    Set up the configuration for the Hypothalamus Segmentation model.

    This function loads the configuration, sets the output directory and batch size, and adjusts the output tensor
    dimensions based on the padded size specified in the configuration.

    Parameters
    ----------
    cfg : yacs.config.CfgNode
        The configuration node to load.
    out_dir : Path
        The output directory where the results will be stored.
    batch_size : int, default=1
        The batch size to use. Default is 1.

    Returns
    -------
    yacs.config.CfgNode
        The loaded and adjusted configuration node.

    """
    cfg = load_config(cfg)
    cfg.OUT_LOG_DIR = str(out_dir or cfg.LOG_DIR)
    cfg.TEST.BATCH_SIZE = batch_size

    out_dims = cfg.DATA.PADDED_SIZE
    if out_dims > cfg.DATA.PADDED_SIZE:
        cfg.MODEL.OUT_TENSOR_WIDTH = out_dims
        cfg.MODEL.OUT_TENSOR_HEIGHT = out_dims
    else:
        cfg.MODEL.OUT_TENSOR_WIDTH = cfg.DATA.PADDED_SIZE
        cfg.MODEL.OUT_TENSOR_HEIGHT = cfg.DATA.PADDED_SIZE
    return cfg


if __name__ == "__main__":
    # arguments
    parser = option_parse()
    args = vars(parser.parse_args())
    log_name = (args["log_name"] or
                args["out_dir"] / args["sid"] / "scripts/hypvinn_seg.log")
    del args["log_name"]

    from FastSurferCNN.utils.logging import setup_logging
    setup_logging(log_name)

    import sys
    sys.exit(main(**args))
