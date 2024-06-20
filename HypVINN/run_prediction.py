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
import argparse
from pathlib import Path
from time import time

import numpy as np
import torch
import nibabel as nib

import FastSurferCNN.utils.logging as logging

from HypVINN.config.hypvinn_files import HYPVINN_SEG_NAME
from HypVINN.config.hypvinn_global_var import Plane, planes
from HypVINN.data_loader.data_utils import hypo_map_label2subseg
from HypVINN.inference import Inference
from HypVINN.models.networks import HypVINN
from HypVINN.utils import ModalityDict, ModalityMode, ViewOperations
from HypVINN.utils.load_config import load_config
from HypVINN.data_loader.data_utils import rescale_image
from HypVINN.utils.stats_utils import compute_stats
from HypVINN.utils.img_processing_utils import save_segmentation
from HypVINN.utils.visualization_utils import plot_qc_images

logger = logging.get_logger(__name__)

##
# Input array preparation
##


def load_volumes(
        mode: ModalityMode,
        t1_path: Path,
        t2_path: Path,
) -> tuple[
    ModalityDict,
    np.ndarray,
    nib.FilebasedHeader,
    np.ndarray,
    tuple[int, ...],
]:
    modalities: ModalityDict = {}

    t1_size = ()
    t2_size = ()
    t1_zoom = ()
    t2_zoom = ()
    affine = np.ndarray([0])
    header = None
    zoom = ()
    size = ()

    if "t1" in mode:
        logger.info(f'Loading T1 image from : {t1_path}')
        t1 = nib.load(t1_path)
        t1 = nib.as_closest_canonical(t1)
        if mode in ('t1t2', 't1'):
            affine = t1.affine
            header = t1.header
        t1_zoom = t1.header.get_zooms()
        zoom = np.round(t1_zoom, 3)
        # Conform Intensities
        modalities['t1'] = rescale_image(np.asarray(t1.dataobj))
        t1_size = modalities['t1'].shape
        size = t1_size
    if "t2" in mode:
        logger.info(f'Loading T2 image from : {t2_path}')
        t2 = nib.load(t2_path)
        t2 = nib.as_closest_canonical(t2)
        if mode == 't2':
            affine = t2.affine
            header = t2.header
        t2_zoom = t2.header.get_zooms()
        zoom = np.round(t2_zoom, 3)
        # Conform Intensities
        modalities['t2'] = np.asarray(rescale_image(t2.get_fdata()), dtype=np.uint8)
        t2_size = modalities['t2'].shape
        size = t2_size

    if mode == "t1t2":
        if not np.allclose(np.array(t1_zoom), np.array(t2_zoom), rtol=0.05):
            raise AssertionError(
                f"T1 : {t1_zoom} and T2 : {t2_zoom} images have different "
                f"resolutions"
            )
        if not np.allclose(np.array(t1_size), np.array(t2_size), rtol=0.05):
            raise AssertionError(
                f"T1 : {t1_size} and T2 : {t2_size} images have different size"
            )
    elif mode not in ("t1", "t2"):
        raise ValueError(f"Invalid Mode in for modality {mode}")

    return modalities, affine, header, zoom, size


def get_prediction(
        subject_name: str,
        modalities: ModalityDict,
        orig_zoom,
        model: HypVINN,
        target_shape: tuple[int, int, int],
        view_opts: ViewOperations,
        out_scale=None,
        mode: ModalityMode = "t1t2",
) -> torch.Tensor:
    device, viewagg_device = model.get_device()
    dim = model.get_max_size()

    model.set_model(view_opts["coronal"]["cfg"])
    model.load_checkpoint(view_opts["coronal"]["ckpt"])

    pred_shape = (dim, dim, dim, model.get_num_classes())
    # Set up tensor to hold probabilities and run inference
    pred_prob = torch.zeros(pred_shape, dtype=torch.float, device=viewagg_device)
    for plane, opts in view_opts.items():
        logger.info(f"Evaluating {plane} model, cpkt :{opts['ckpt']}")
        model.set_cfg(opts["cfg"])
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
def set_up_cfgs(cfg, args):
    cfg = load_config(cfg)
    cfg.OUT_LOG_DIR = args.out_dir if args.out_dir is not None else cfg.LOG_DIR
    cfg.TEST.BATCH_SIZE = args.batch_size

    out_dims = cfg.DATA.PADDED_SIZE
    cfg.MODEL.OUT_TENSOR_WIDTH = out_dims if out_dims > cfg.DATA.PADDED_SIZE else cfg.DATA.PADDED_SIZE
    cfg.MODEL.OUT_TENSOR_HEIGHT = out_dims if out_dims > cfg.DATA.PADDED_SIZE else cfg.DATA.PADDED_SIZE
    return cfg


def run_hypo_seg(
        args: argparse.Namespace,
        subject_name: str,
        mode: ModalityMode,
        t1_path: Path,
        t2_path: Path,
        out_dir: Path,
        threads: int,
        seg_file: Path = Path("mri") / HYPVINN_SEG_NAME,
):
    start = time()

    view_ops: ViewOperations = {a: None for a in planes}
    logger.info('Setting up HypVINN run')
    cfg_ax = set_up_cfgs(args.cfg_ax, args)
    logger.info(f'Axial model configuration from : {args.cfg_ax}')
    view_ops["axial"] = {"cfg": cfg_ax, "ckpt": args.ckpt_ax}

    cfg_sag = set_up_cfgs(args.cfg_sag, args)
    logger.info(f'Sagittal model configuration from : {args.cfg_sag}')
    view_ops["sagittal"] = {"cfg": cfg_sag, "ckpt": args.ckpt_sag}

    cfg_cor = set_up_cfgs(args.cfg_cor, args)
    logger.info(f'Coronal model configuration from : {args.cfg_cor}')
    view_ops["coronal"] = {"cfg": cfg_cor, "ckpt": args.ckpt_cor}

    for plane, pcfg in zip(planes, (cfg_ax, cfg_cor, cfg_sag)):
        model = pcfg.MODEL
        if mode != model.MODE and 'HypVinn' not in model.MODEL_NAME:
            raise AssertionError(
                f"Modality mode different between input arg: "
                f"{mode} and axial train cfg:  {model.MODE}"
            )

    cfg_fin, ckpt_fin = cfg_cor, args.ckpt_cor

    # Set up model
    model = Inference(cfg=cfg_fin, args=args)

    logger.info('----' * 30)
    logger.info(f"Evaluating hypothalamus model on {subject_name}")
    load = time()

    # Load  Images
    modalities, ras_affine, ras_header, orig_zoom, orig_size = load_volumes(
        mode=mode,
        t1_path=t1_path,
        t2_path=t2_path,
    )
    logger.info(f"Scale factor: {orig_zoom}")
    logger.info(f"images loaded in {time() - load:0.4f} seconds")

    load = time()
    pred_classes = get_prediction(
        subject_name,
        modalities,
        orig_zoom,
        model,
        target_shape=orig_size,
        view_opts=view_ops,
        out_scale=None,
        mode=mode,
        logger=logger,
    )
    logger.info(f"Model prediction finished in {time() - load:0.4f} seconds")
    logger.info(f"Saving prediction at {out_dir}")

    save = time()
    if mode == 't1t2' or mode == 't1':
        orig_path = t1_path
    else:
        orig_path = t2_path

    pred_path = save_segmentation(
        pred_classes,
        orig_path=orig_path,
        ras_affine=ras_affine,
        ras_header=ras_header,
        save_dir=out_dir,
        seg_file=seg_file,
        save_mask=True,
    )
    logger.info(
        f"Prediction successfully saved as {pred_path} in "
        f"{time() - save:0.4f} seconds"
    )
    if getattr(args, "qc_snapshots", False):
        plot_qc_images(
            save_dir=out_dir / "qc_snapshots",
            orig_path=orig_path,
            prediction_path=pred_path,
        )

    logger.info("Computing stats")
    return_value = compute_stats(
        orig_path=orig_path,
        prediction_path=pred_path,
        save_dir=out_dir / "stats",
        threads=threads,
    )
    if return_value != 0:
        logger.error(return_value)

    logger.info(
        f"Processing segmentation finished in {time() - start:0.4f} seconds"
    )
