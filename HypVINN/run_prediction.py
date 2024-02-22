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
import numpy as np
import torch
import os
import nibabel as nib
import time
from collections import defaultdict
from HypVINN.data_loader.data_utils import hypo_map_label2subseg, hypo_map_subseg_2_fsseg
from HypVINN.inference import Inference
from HypVINN.utils.load_config import load_config
from HypVINN.data_loader.data_utils import rescale_image
from HypVINN.utils.stats_utils import compute_stats
from HypVINN.utils.img_processing_utils import save_segmentation
from HypVINN.utils.visualization_utils import plot_qc_images
import FastSurferCNN.utils.logging as logging

logger = logging.get_logger(__name__)

##
# Input array preparation
##

def load_volumes(mode,t1_path,t2_path):
    modalities = defaultdict(lambda: defaultdict(list))

    if mode == 't1':
        t1_mode = True
        t2_mode = False
    elif mode == 't2':
        t1_mode = False
        t2_mode = True
    else:
        t1_mode = True
        t2_mode = True

    if t1_mode:
        logger.info('Loading T1 image from : {}'.format(t1_path))
        t1 = nib.load(t1_path)
        t1 = nib.as_closest_canonical(t1)
        affine = t1.affine
        header = t1.header
        t1_zoom = t1.header.get_zooms()
        zoom = np.round(t1_zoom, 3)
        # Conform Intensities
        modalities['t1'] = np.asarray(rescale_image(t1.get_fdata()), dtype=np.uint8)
        t1_size = modalities['t1'].shape
        size = t1_size
    if t2_mode:
        logger.info('Loading T2 image from : {}'.format(t2_path))
        t2 = nib.load(t2_path)
        t2 = nib.as_closest_canonical(t2)
        affine = t2.affine
        header = t2.header
        t2_zoom = t2.header.get_zooms()
        zoom = np.round(t2_zoom, 3)
        # Conform Intensities
        modalities['t2'] = np.asarray(rescale_image(t2.get_fdata()), dtype=np.uint8)
        t2_size = modalities['t2'].shape
        size = t2_size

    if t1_mode and t2_mode:
        assert np.allclose(np.array(t1_zoom), np.array(t2_zoom),
                           rtol=0.05), "T1 : {} and T2 : {} images have different resolutions".format(t1_zoom,
                                                                                                      t2_zoom)
        assert np.allclose(np.array(t1_size), np.array(t2_size),
                           rtol=0.05), "T1 : {} and T2 : {} images have different size".format(t1_size, t2_size)

    return modalities,affine,header,zoom,size


def run_model(model, subject_name, modalities, orig_zoom, pred_prob, out_scale,mode='multi'):
    # get prediction
    pred_prob = model.run(subject_name, modalities, orig_zoom, pred_prob, out_res=out_scale,mode=mode)

    return pred_prob

def get_prediction(subject_name, modalities, orig_zoom, model, gt_shape, view_opts, logger, out_scale=None,mode='multi'):
    from scipy.special import softmax
    device,viewagg_device = model.get_device()
    dim = model.get_max_size()

    # Coronal model
    logger.info(f'Evaluating Coronal model, cpkt :{view_opts["coronal"]["ckpt"]}')
    model.set_model(view_opts["coronal"]["cfg"])
    model.load_checkpoint(view_opts["coronal"]["ckpt"])

    pred_prob = torch.zeros((dim, dim, dim, model.get_num_classes()), dtype=torch.float).to(viewagg_device)

    # Set up tensor to hold probabilities and run inference (coronal model by default)
    pred_prob = run_model(model, subject_name, modalities, orig_zoom, pred_prob, out_scale,mode=mode)

    # Axial model
    logger.info(f'Evaluating Axial model, cpkt :{view_opts["axial"]["ckpt"]}')
    model.set_cfg(view_opts["axial"]["cfg"])
    model.load_checkpoint(view_opts["axial"]["ckpt"])
    pred_prob += run_model(model, subject_name, modalities, orig_zoom, pred_prob, out_scale,mode=mode)

    # Sagittal model
    logger.info(f'Evaluating Sagittal model, cpkt :{view_opts["sagittal"]["ckpt"]}')
    model.set_model(view_opts["sagittal"]["cfg"])
    model.load_checkpoint(view_opts["sagittal"]["ckpt"])
    pred_prob += run_model(model, subject_name, modalities, orig_zoom, pred_prob, out_scale,mode=mode)

    # Post processing
    h, w, d = gt_shape  # final prediction shape equivalent to input ground truth shape

    if np.any(gt_shape < pred_prob.shape[:3]):
        # if orig was padded before running through model (difference in aseg_size and pred_shape), select
        # slices of interest only. This currently works only for "top_left" padding (see augmentation)
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

def run_hypo_seg(args):

    start = time.time()

    view_ops = {a: None for a in ["coronal", "axial", "sagittal"]}
    logger.info('Setting up HypVINN run')
    cfg_ax = set_up_cfgs(args.cfg_ax, args)
    logger.info(f'Axial model configuration from : {args.cfg_ax}')
    view_ops["axial"] = {"cfg": cfg_ax, "ckpt": args.ckpt_ax}
    assert args.mode == cfg_ax.MODEL.MODE or 'HypVinn' in cfg_ax.MODEL.MODEL_NAME  , 'Modalitie mode different between input arg : {} and axial train cfg:  {}'.format(args.mode,cfg_ax.MODEL.MODE)

    cfg_sag = set_up_cfgs(args.cfg_sag, args)
    logger.info(f'Sagittal model configuration from : {args.cfg_sag}')
    view_ops["sagittal"] = {"cfg": cfg_sag, "ckpt": args.ckpt_sag}
    assert args.mode == cfg_sag.MODEL.MODE or 'HypVinn' in cfg_sag.MODEL.MODEL_NAME, 'Modalitie mode different between input arg : {} and sagittal train cfg:  {}'.format(args.mode,cfg_sag.MODEL.MODE)

    cfg_cor = set_up_cfgs(args.cfg_cor, args)
    logger.info(f'Coronal model configuration from : {args.cfg_cor}')
    view_ops["coronal"] = {"cfg": cfg_cor, "ckpt": args.ckpt_cor}
    assert args.mode == cfg_cor.MODEL.MODE or 'HypVinn' in cfg_cor.MODEL.MODEL_NAME, 'Modalitie mode different between input arg : {} and coronal train cfg:  {}'.format(args.mode,cfg_cor.MODEL.MODE)

    cfg_fin, ckpt_fin = cfg_cor, args.ckpt_cor

    # Set up model
    model = Inference(cfg=cfg_fin,args=args)

    try:
        logger.info('----'*30)
        logger.info(f"Evaluating hypothalamus model on {args.sid}")
        load = time.time()

        # Load  Images
        modalities, ras_affine, ras_header, orig_zoom, orig_size = load_volumes(mode=args.mode, t1_path=args.t1, t2_path=args.t2)
        logger.info("Scale factor in: {}".format(orig_zoom))

        logger.info("images loaded in {:0.4f} seconds".format(time.time() - load))


        load = time.time()
        pred_classes = get_prediction(args.sid,modalities, orig_zoom, model, gt_shape=orig_size,
                              view_opts=view_ops, out_scale=None,mode=args.mode,logger=logger)
        logger.info("Model prediction finished in {:0.4f} seconds".format(time.time()-load))


        logger.info(f"Saving prediction at {args.out_dir}")

        save = time.time()
        if args.mode == 'multi' or args.mode == 't1':
            orig_path = args.t1
        else:
            orig_path = args.t2

        pred_path = save_segmentation(pred_classes, orig_path= orig_path, affine=ras_affine, header=ras_header, save_dir=os.path.join(args.out_dir,'mri'))
        logger.info("Prediction successfully saved as {} in {:0.4f} seconds".format(pred_path, time.time()-save))
        if args.qc_snapshots:
            plot_qc_images(save_dir=os.path.join(args.out_dir,'qc_snapshots'),orig_path=orig_path,prediction_path=pred_path)

        logger.info('Computing stats')
        flag = compute_stats(orig_path=orig_path, prediction_path=pred_path, save_dir=os.path.join(args.out_dir,'stats'),threads=args.threads)
        if flag != 0:
            logger.info(flag)

    except FileNotFoundError as e:
        logger.info("Failed Evaluation on {} with exception:\n{} )".format(args.sid, e))

    logger.info("Processing segmentation finished in {:0.4f} seconds".format(time.time() - start))


