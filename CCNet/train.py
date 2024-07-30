
# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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
import pprint
import subprocess
import time
import os
from networkx import center
#import concurrent.futures
#import multiprocessing # TODO: replace this with concurrent.futures when python 3.8 is phased out
#from collections import defaultdict
#import sys

import torch
import ignite # needed for ignite.metrics
from ignite.metrics import Metric
import ignite.metrics
from torch.utils.tensorboard import SummaryWriter
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


import numpy as np
from tqdm import tqdm
import random
from tqdm.contrib.logging import logging_redirect_tqdm

from CCNet.data_loader import loader
from CCNet.models.networks import build_model
from FastSurferCNN.models.optimizer import get_optimizer
from CCNet.models.losses import get_loss_func, SSIMLoss, GradientLoss, MSELoss
from FastSurferCNN.utils import logging, checkpoint as cp
from FastSurferCNN.utils.lr_scheduler import get_lr_scheduler
from CCNet.utils.meters import Meter, DiceScore, LocDistance
#from CCNet.utils.metrics import iou_score, precision_recall
from CCNet.utils.misc import update_num_steps, plot_predictions, plot_predictions_inpainting, calculate_centers_of_comissures
from FastSurferCNN.config.global_var import get_class_names

logger = logging.getLogger(__name__)


class CCNetTrainer:
    def __init__(self, cfg):
        # Set random seed from configs.
        #np.random.seed(cfg.RNG_SEED)
        #torch.manual_seed(cfg.RNG_SEED)
        
        # self.set_determinism(cfg.RNG_SEED)
        self.cfg = cfg

        # Create the checkpoint dir.
        self.checkpoint_dir = cp.create_checkpoint_dir(cfg.LOG_DIR, cfg.EXPR_NUM)
        logging.setup_logging(os.path.join(cfg.LOG_DIR, "logs", cfg.EXPR_NUM + ".log"))
        logger.setLevel(cfg.LOG_LEVEL)
        logger.info("New training run with config:")
        logger.info(pprint.pformat(cfg))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(cfg)
        self.loss_func = get_loss_func(cfg)

        
        #assert(16 % cfg.TRAIN.BATCH_SIZE == 0), "Batch size must be a divisor of 16"

        # set up class names
        self.class_names = get_class_names(cfg.DATA.PLANE, cfg.DATA.CLASS_OPTIONS)

        # Set up logger format
        self.format_placeholder_classes = "{}\t" * (cfg.MODEL.NUM_CLASSES - 2) + "{}"
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.plot_dir = os.path.join(cfg.LOG_DIR, "pred", str(cfg.EXPR_NUM))
        os.makedirs(self.plot_dir, exist_ok=True)

        #self.subepoch = False if self.cfg.TRAIN.BATCH_SIZE == 16 else True
        #self.subepoch = False

        self.has_cutout = 'Cutout' in self.cfg.DATA.AUG or any('cutout' in i.lower() for i in self.cfg.DATA.AUG)

        if self.has_cutout:
            self.ssim_loss = SSIMLoss()
            #self.gradient_loss = GradientLoss()
            #self.mse = MSELoss(reduction='mean')
            self.mae = torch.nn.L1Loss(reduction='mean')

            self.inpainting_loss = lambda pred, orig: self.ssim_loss(pred, orig) #+ self.gradient_loss(pred, orig) + self.mse(pred)
            self.inpainting_loss_mask = lambda pred, orig, mask: self.mae(pred[mask], orig[mask]) + self.mae(pred[~mask], torch.zeros_like(pred[~mask]))  #+ self.ssim_loss(pred, orig, mask) #+ self.gradient_loss(pred, orig, mask) + self.mse(pred, orig, mask)

    def run_epoch(self, train: bool, data_loader, meter: Meter, epoch: int, log_name: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        meter.reset()

        if train:
            self.model.train()
        else:
            self.model.eval() # TODO: add nograd

        logger.info(f"{log_name} epoch started ")
        epoch_start = time.time()
        loss_batch = torch.zeros(1, device=self.device)

        with torch.set_grad_enabled(train), logging_redirect_tqdm(): #, multiprocessing.Pool() as process_executor:
            for curr_iter, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        
                images, labels, weights, scale_factors, cutout_mask = batch['image'].to(self.device), \
                                                        batch['label'].to(self.device), \
                                                        batch['weight'].float().to(self.device), \
                                                        batch['scale_factor'], \
                                                        batch['cutout_mask']
                
                loc_loss = None
                mse_loss = None
                loss_seg = None

                cutout_mask_midslice = cutout_mask[:, cutout_mask.shape[1]//2] # get middle slice

                if train: # and (not self.subepoch or (curr_iter)%(16/self.cfg.TRAIN.BATCH_SIZE) == 0):
                    optimizer.zero_grad() # every second epoch to get batchsize of 16 if using 8                

                if self.cfg.MODEL.MODEL_NAME == 'FastSurferPaint':

                    cutout_mask_empty = (cutout_mask_midslice == 0).all()

                    network_input = torch.concat([images, cutout_mask.to(self.device)], dim=1)
                    pred = self.model(network_input, scale_factors)

                    orig_slice = batch['unmodified_center_slice'].to(self.device)
                    pred, estimated_slice = pred
                    estimated_slice = estimated_slice.squeeze(1)
                    loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)

                    #inpainting_loss = self.ssim_loss(estimated_slice, orig_slice, cutout_mask if not cutout_mask_empty else None)

                    if not cutout_mask_empty:
                        inpainting_loss = self.inpainting_loss_mask(estimated_slice, orig_slice, cutout_mask_midslice)
                        self.update_metrics(meter.ignite_metrics, pred, labels, pred_slice=estimated_slice, orig_slice=orig_slice, cutout_mask=cutout_mask_midslice)
                    else:
                        inpainting_loss = 0
                        #inpainting_loss = self.inpainting_loss(estimated_slice, orig_slice)
                        self.update_metrics(meter.ignite_metrics, pred, labels)
                        
                    #assert(cutout_mask_empty == (torch.sum(cutout_mask_midslice) == 0)), 'cutout mask is not empty but sum is 0'

                    loss_total = loss_total * (1 - self.cfg.INPAINT_LOSS_WEIGHT) + (self.cfg.INPAINT_LOSS_WEIGHT * inpainting_loss if not cutout_mask_empty else 0)
                elif self.cfg.MODEL.MODEL_NAME == 'FastSurferLocalisation':
                    
                    pred = self.model(images, scale_factors)         
                    loss_total, loss_seg, loss_dice, loss_ce, loc_loss, mse_loss = self.loss_func(pred, labels, weights)
                    # metrics only on classification
                    self.update_metrics(meter.ignite_metrics, pred, labels)
                    #logger.debug(f'Losses: loss_total: {loss_total}, loss_dice: {loss_dice}, loss_ce: {loss_ce}, seg_loss: {seg_loss}, loc_loss: {loc_loss}')
                else:
                    pred = self.model(images, scale_factors)         
                    loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)
                    self.update_metrics(meter.ignite_metrics, pred, labels)

                if torch.isnan(loss_total):
                    logger.info('loss is nan - stopping training and starting debugging')
                    #import pdb; pdb.set_trace()

                meter.update_stats(pred, labels, loss_total)
                meter.log_iter(curr_iter, epoch)
                
                meter.write_summary(loss_total, 
                                    lr        = scheduler.get_last_lr() if scheduler is not None else [self.cfg.OPTIMIZER.BASE_LR],
                                    loss_ce   = loss_ce,
                                    loss_dice = loss_dice, 
                                    loss_seg  = loss_seg,
                                    loc_loss  = loc_loss,
                                    dist_loss = mse_loss)
               



                if train:
                    loss_total.backward() # TODO: should this be loss_batch?
                    #if not self.subepoch or (curr_iter+1)%(16/self.cfg.TRAIN.BATCH_SIZE) == 0:
                    optimizer.step() # every second epoch to get batchsize of 16 if using 8
                    if scheduler is not None:
                        scheduler.step(epoch + curr_iter / len(data_loader))

                    loss_batch += loss_total

                # Plot sample predictions
                if curr_iter == 1: # try to log cutout images if possible
                    plt_title = 'Training Results Epoch ' + str(epoch)
                    file_save_name = os.path.join(self.plot_dir,
                                                'Epoch_' + str(epoch) + f'_{log_name}_Predictions.pdf')

                    logger.debug(f'Plotting {file_save_name}')

                    if self.cfg.MODEL.MODEL_NAME == 'FastSurferLocalisation':
                        logger.debug(f'pred shape: {pred.shape}')
                        logger.debug(f'labels shape: {labels.shape}')
                        seg_pred = pred[:,:-1,:,:].detach()
                        seg_labels = labels[:,:labels.shape[1]//2, :]
                        logger.debug(f'seg_pred shape: {seg_pred.shape}')
                        logger.debug(f'seg_labels shape: {seg_labels.shape}')

                    else:
                        seg_pred = pred.detach()
                        seg_labels = labels
                    
                    _, batch_output = torch.max(seg_pred, dim=1)

                    logger.debug(f'Unique  labels: {np.unique(seg_labels.detach().cpu())}')
                    logger.debug(f'Unique  batch output: {np.unique(batch_output.detach().cpu())}')

                    plt_images = images.detach()

                    assert (batch_output.cpu().numpy().astype(np.int64) == batch_output.cpu().numpy()).all(), 'batch output is not int'
                    
                    assert (seg_labels.cpu().numpy().astype(np.int64) == seg_labels.cpu().numpy()).all(), 'seg labels is not int'

                        

                    plot_predictions(plt_images, seg_labels.detach().int(), batch_output.int(), plt_title, file_save_name, process_executor=None, lut=self.cfg.DATA.LUT) #process_executor)


                    if self.cfg.MODEL.MODEL_NAME == 'FastSurferPaint':
                        file_save_name = os.path.join(self.plot_dir,
                                                    'Epoch_' + str(epoch) + f'_{log_name}_Predictions_EstimatedSlice.pdf')
                        plot_predictions_inpainting(plt_images, orig_slice.detach(), estimated_slice.detach(), plt_title, file_save_name, process_executor=None) #process_executor)

            #process_executor.shutdown(wait=True)

            # temporarily removed
            # process_executor.close()
            # process_executor.join()


        meter.log_epoch(epoch, runtime=time.time() - epoch_start)
        logger.info("{} epoch {} finished in {:.04f} seconds".format(log_name, epoch, time.time() - epoch_start))

        if 'IoU' in meter.ignite_metrics.keys():
            mIOU = meter.ignite_metrics['IoU'].compute().mean()
        elif 'FastSurfer_dice' in meter.ignite_metrics.keys():
            mIOU = meter.ignite_metrics['FastSurfer_dice'].compute().mean()
        else:
            mIOU = None
        
        
        return mIOU
    
    def create_ignite_metrics(self, train=True):

        #device = torch.device("cpu") # confusion matrix is faster on cpu
        device = self.device
        
        ignite_metrics = {}

        if not train:
            # ignite_metrics = {'confusion_matrix': ignite.metrics.confusion_matrix.ConfusionMatrix(self.num_classes, device=torch.device('cpu'))}
            # ignite_metrics['DICE_ignite'] = ignite.metrics.DiceCoefficient(ignite_metrics['confusion_matrix'], ignore_index=0) # ignore background
            # ignite_metrics['IoU'] = ignite.metrics.IoU(ignite_metrics['confusion_matrix'], ignore_index=0) # ignore background
            # ignite_metrics['MeanRecall'] = ignite.metrics.Recall(average=True, device=device)
            # ignite_metrics['MeanPrecision'] = ignite.metrics.Precision(average=True, device=device)


            ignite_metrics['FastSurfer_dice'] = DiceScore(self.num_classes, device=device)
        else:
            ignite_metrics['FastSurfer_dice'] = DiceScore(self.num_classes, device=device)

        if self.has_cutout:
            ignite_metrics['PSNR'] = ignite.metrics.PSNR(data_range=255, device=device) # TODO: should be 1?
            ignite_metrics['PSNR_inpaint'] = ignite.metrics.PSNR(data_range=255, device=device)
            ignite_metrics['SSIM'] = ignite.metrics.SSIM(data_range=255, kernel_size=(11, 11), sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True, device=torch.device('cpu')) # always cpu because of determinism TODO: make flag
            ignite_metrics['SSIM_inpaint'] = ignite.metrics.SSIM(data_range=255, kernel_size=(11, 11), sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True, device=torch.device('cpu'))

        if self.cfg.MODEL.MODEL_NAME == 'FastSurferLocalisation':
            # Add localisational distance error
            ignite_metrics['locational_distance'] = LocDistance(device=device, axis = 2)

        return ignite_metrics
    
    
    def update_metrics(self, ignite_metrics, pred, labels, pred_slice=None, orig_slice=None, cutout_mask=None):
        if self.cfg.MODEL.MODEL_NAME == 'FastSurferLocalisation':
            if 'locational_distance' in ignite_metrics.keys():
                ignite_metrics['locational_distance'].update(pred, labels)
            pred = pred[:,:-1,:,:]
            labels = labels[:,:labels.shape[1]//2, :]

        # dice and iou are calculated from confusion matrix

        if 'FastSurfer_dice' in ignite_metrics.keys():
            ignite_metrics['FastSurfer_dice'].update((pred, labels.long()))

        if 'confusion_matrix' in ignite_metrics.keys():
            # NOTE: this will give gibberish if given uint8 tensors
            ignite_metrics['confusion_matrix'].update((torch.nn.functional.softmax(pred.detach().to(torch.float32), dim=1).cpu(), labels.detach().to(torch.int64)).cpu()) # this is slow on GPU - fixed in https://github.com/pytorch/pytorch/pull/97090, but we dont have that version yet

        if 'MeanRecall' in ignite_metrics.keys():
            ignite_metrics['MeanRecall'].update((pred, labels.long()))
        if 'MeanPrecision' in ignite_metrics.keys():
            ignite_metrics['MeanPrecision'].update((pred, labels.long()))

        if self.has_cutout and pred_slice is not None and orig_slice is not None and cutout_mask.sum() > 0:
            if 'PSNR' in ignite_metrics.keys():
                ignite_metrics['PSNR'].update((pred_slice.unsqueeze(1), orig_slice.unsqueeze(1)))
            if 'SSIM' in ignite_metrics.keys():
                ignite_metrics['SSIM'].update((pred_slice.unsqueeze(1), orig_slice.unsqueeze(1)))


            if ('PSNR_inpaint' in ignite_metrics.keys() or 'SSIM_inpaint' in ignite_metrics.keys()) and cutout_mask is not None:
                for s in range(pred_slice.shape[0]): # iterate over samples in batch
                    i, j = np.where(cutout_mask[s])
                    if len(i) == 0 or len(j) == 0:
                        continue

                    maxi = np.max(i)
                    mini = np.min(i)
                    maxj = np.max(j)
                    minj = np.min(j)

                    pred_slice_masked = pred_slice[s, mini:maxi+1, minj:maxj+1]
                    orig_slice_masked = orig_slice[s, mini:maxi+1, minj:maxj+1]

                    if 'PSNR_inpaint' in ignite_metrics.keys():
                        ignite_metrics['PSNR_inpaint'].update((pred_slice_masked[None, None, ...], orig_slice_masked[None, None, ...]))
                        # if np.isnan(ignite_metrics['PSNR_inpaint'].compute()):
                        #     print('PSNR is nan - stopping training and starting debugging')
                    if 'SSIM_inpaint' in ignite_metrics.keys():
                        try:
                            ignite_metrics['SSIM_inpaint'].update((pred_slice_masked[None, None, ...], orig_slice_masked[None, None, ...]))
                        except RuntimeError:
                            assert(pred_slice_masked.shape == orig_slice_masked.shape)
                            assert(pred_slice_masked.shape[0] <= 5 or pred_slice_masked.shape[1] <= 5), \
                                    'SSIM on cutout region failed - but cutout area was of sufficient size - unexpected error!'
                            
        return True


    
    def create_logging(self, start_epoch, train_loader, val_loader, val_loader2=None, val_loader_inpaint=None, has_cutout=False):
        # Create tensorboard summary writer
        writer = SummaryWriter(self.cfg.SUMMARY_PATH, flush_secs=15)

        create_meter = lambda name, is_train, data_loader: Meter(self.cfg,
                                                    mode=name,
                                                    global_step=start_epoch,
                                                    total_iter=len(data_loader),
                                                    total_epoch=self.cfg.TRAIN.NUM_EPOCHS,
                                                    writer=writer,
                                                    ignite_metrics=self.create_ignite_metrics(train=is_train))
        

        train_meter = create_meter('train', True, train_loader)

        val_meter = create_meter('val', False, val_loader)
        
        # if cutout is used, create a separate validation meter for inpainting
        if self.has_cutout:
            val_meter_inpaint = create_meter('val_inpaint', False, val_loader_inpaint)
        else:    
            val_meter_inpaint = None
            
        if val_loader2:
            val_meter_tumor = create_meter('val_tumor', False, val_loader2)
        else:
            val_meter_tumor = None
            
        return train_meter, val_meter, val_meter_inpaint, val_meter_tumor
    
    def make_data_loaders(self):
        train_loader = loader.get_dataloader(self.cfg, "train")
        val_loader = loader.get_dataloader(self.cfg, "val")
        if self.has_cutout:
            val_loader_inpaint = loader.get_dataloader(self.cfg, "val_inpainting", val_dataset=val_loader.dataset)
        else:
            val_loader_inpaint = None
        if self.cfg.DATA.PATH_HDF5_VAL2:
            val_loader2 = loader.get_dataloader(self.cfg, "val_tumor", data_path=self.cfg.DATA.PATH_HDF5_VAL2)
        else:
            val_loader2 = None

        return train_loader, val_loader, val_loader_inpaint, val_loader2
    
    def save_code(self, dir_path):
        dir_path = os.path.abspath(dir_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        # run git ls-files | tar Tczf - mycode.zip
        cmd = f'git ls-files | tar Tczf - {os.path.join(dir_path, "repository_code.zip")}'
        #subprocess.Popen([cmd], shell=True) # TODO: Fix tar "Cannot stat: No such file or directory"




    def run(self):
        if self.cfg.NUM_GPUS > 1:
            assert self.cfg.NUM_GPUS <= torch.cuda.device_count(), f"Trying to use {self.cfg.NUM_GPUS} GPUs, but only {torch.cuda.device_count()} are available"
            logger.info(f"Using  {self.cfg.NUM_GPUS} GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        train_loader, val_loader, val_loader_inpaint, val_loader2 = self.make_data_loaders()
        
        # check loaders
        
        sample_train = next(iter(train_loader))
        sample_val = next(iter(val_loader))
        assert(sample_train['image'].shape[1:] == sample_val['image'].shape[1:]), 'Train and val loader have different input shapes'
        
        # TODO: add additional logging for loaders
        update_num_steps(train_loader, self.cfg)

        # Transfer the model to device(s)
        self.model = self.model.to(self.device)

        optimizer = get_optimizer(self.model, self.cfg)
        scheduler = get_lr_scheduler(optimizer, self.cfg)

        checkpoint_paths = cp.get_checkpoint_path(self.cfg.LOG_DIR,
                                                 self.cfg.TRAIN.RESUME_EXPR_NUM)
                                                 
        if self.cfg.TRAIN.RESUME and checkpoint_paths:
            try:
                checkpoint_path = checkpoint_paths.pop()
                checkpoint_epoch, best_metric = cp.load_from_checkpoint(
                    checkpoint_path,
                    self.model,
                    optimizer,
                    scheduler,
                    self.cfg.TRAIN.FINE_TUNE
                )
                start_epoch = checkpoint_epoch
                best_miou = best_metric
                logger.info(f"Resume training from epoch {start_epoch}")
            except Exception as e:
                logger.warning("No model to restore. Resuming training from Epoch 0. {}".format(e))
        else:
            logger.info("Training from scratch")
            start_epoch = 0
            best_miou = 0

            logger.info("Saving code to {}".format(self.cfg.LOG_DIR))
            self.save_code(self.cfg.LOG_DIR)

        logger.info("{} parameters in total".format(sum(x.numel() for x in self.model.parameters())))

        
        train_meter, val_meter, val_meter_inpaint, val_meter_tumor = self.create_logging(
            start_epoch, train_loader, val_loader, val_loader2, val_loader_inpaint, has_cutout=self.has_cutout)
        

        logger.info("Summary path {}".format(self.cfg.SUMMARY_PATH))
        # Perform the training loop.
        logger.info("Start epoch: {}".format(start_epoch + 1))

        # with torch.profiler.profile(
        #     schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=0),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(self.cfg.SUMMARY_PATH),
        #     record_shapes=False,
        #     profile_memory=False,
        #     with_stack=True) as prof: #, torch.amp.autocast(enabled=True, device_type=self.device.__str__(), cache_enabled=None): TODO: not supported in prelu?



        for epoch in range(start_epoch, self.cfg.TRAIN.NUM_EPOCHS):
            #self.train(train_loader, optimizer, scheduler, train_meter, epoch=epoch)
            _ = self.run_epoch(train=True, data_loader=train_loader, optimizer=optimizer, scheduler=scheduler, meter=train_meter, epoch=epoch, log_name='Train')

            #miou = self.eval(val_loader, val_meter, epoch=epoch)
            miou = self.run_epoch(train=False, data_loader=val_loader, optimizer=None, scheduler=None, meter=val_meter, epoch=epoch, log_name='Validation')

            if val_loader2:
            #_ = self.eval(val_loader2, val_meter_tumor, epoch=epoch, log_name='Validation_Tumor')
                _ = self.run_epoch(train=False, data_loader=val_loader2, optimizer=None, scheduler=None, meter=val_meter_tumor, epoch=epoch, log_name='Validation_Tumor')

            if self.has_cutout:
                #_ = self.eval(val_loader_inpaint, val_meter_inpaint, epoch=epoch, log_name='Validation_Inpainting')
                _ = self.run_epoch(train=False, data_loader=val_loader_inpaint, optimizer=None, scheduler=None, meter=val_meter_inpaint, epoch=epoch, log_name='Validation_Inpainting')


            if (epoch+1) % self.cfg.TRAIN.CHECKPOINT_PERIOD == 0:
                logger.info(f"Saving checkpoint at epoch {epoch+1}")
                cp.save_checkpoint(self.checkpoint_dir,
                                epoch+1,
                                best_miou,
                                self.cfg.NUM_GPUS,
                                self.cfg,
                                self.model,
                                optimizer,
                                scheduler
                                )

            if miou > best_miou:
                best_miou = miou
                logger.info(f"New best checkpoint reached at epoch {epoch+1} with miou of {best_miou}\nSaving new best model.")
                cp.save_checkpoint(self.checkpoint_dir,
                                epoch+1,
                                best_miou,
                                self.cfg.NUM_GPUS,
                                self.cfg,
                                self.model,
                                optimizer,
                                scheduler,
                                best=True
                                )
                                
    @staticmethod
    def set_determinism( # from monai.utils https://docs.monai.io/en/stable/_modules/monai/utils/misc.html#set_determinism
        seed: int = 0,
        use_deterministic_algorithms: bool = True) -> None:
        """
        Set random seed for modules to enable or disable deterministic training.

        Args:
            seed: the random seed to use, default is np.iinfo(np.int32).max.
                It is recommended to set a large seed, i.e. a number that has a good balance
                of 0 and 1 bits. Avoid having many 0 bits in the seed.
                if set to None, will disable deterministic training.
            use_deterministic_algorithms: Set whether PyTorch operations must use "deterministic" algorithms.
        """
        seed = int(seed)
        torch.manual_seed(seed)

        global _seed
        _seed = seed
        random.seed(seed)
        np.random.seed(seed)

        if torch.backends.flags_frozen():
            logger.warn("PyTorch global flag support of backends is disabled, enable it to set global `cudnn` flags.")
            torch.backends.__allow_nonbracketed_mutation_flag = True

        if seed is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:  # restore the original flags
            torch.backends.cudnn.deterministic = torch.backends.cudnn.deterministic
            torch.backends.cudnn.benchmark = torch.backends.cudnn.benchmark
        if use_deterministic_algorithms:
            if hasattr(torch, "use_deterministic_algorithms"):  # `use_deterministic_algorithms` is new in torch 1.8.0
                torch.use_deterministic_algorithms(use_deterministic_algorithms, warn_only=True)
            elif hasattr(torch, "set_deterministic"):  # `set_deterministic` is new in torch 1.7.0
                torch.set_deterministic(use_deterministic_algorithms)
            else:
                logger.warn("use_deterministic_algorithms=True, but PyTorch version is too old to set the mode.")