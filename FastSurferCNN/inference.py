
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
import torch
import numpy as np
import utils.logging as logging
import time

from torch.utils.data import DataLoader
from torchvision import transforms
from models.networks import build_model
from data_loader.augmentation import ToTensorTest, ZeroPad2DTest
from data_loader.data_utils import map_prediction_sagittal2full
from data_loader.dataset import MultiScaleOrigDataThickSlices


logger = logging.getLogger(__name__)


class Inference:
    def __init__(self, cfg, ckpt="", device=""):
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg
        
        # Switch on denormal flushing for faster CPU processing
        # seems to have less of an effect on VINN than old CNN
        torch.set_flush_denormal(True)
        
        # if specific device is requested, check and stop if not available:
        if device.split(':')[0] == "cuda" and not torch.cuda.is_available():
            logger.info("cuda not available, try switching to cpu: --device cpu")
            raise ValueError("--device cuda not available, try --device cpu !")
        if device == "mps" and not torch.backends.mps.is_available():
            logger.info("mps not available, try switching to cpu: --device cpu")
            raise ValueError("--device mps not available, try --device cpu !")
           
        # If auto detect:
        if device == "auto" or not device:
            # 1st check cuda
            if torch.cuda.is_available(): 
                device="cuda"
            elif torch.backends.mps.is_available():
                device="mps"
            else:
                device="cpu"
        # Define device and transfer model
        logger.info("Using device: {}".format(device))
        self.device = torch.device(device)

        # Options for parallel run
        if torch.cuda.device_count() > 0 and self.device == "cuda":
            self.model_parallel = True
        else:
            self.model_parallel = False

        # Initial model setup
        self.model = self.setup_model(cfg)
        self.model_name = self.cfg.MODEL.MODEL_NAME

        # Initial checkpoint loading
        if ckpt:
            self.load_checkpoint(ckpt)

    def setup_model(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg

        # Set up model
        model = build_model(self.cfg)  # ~ model = FastSurferCNN(params_network)
        model.to(self.device)

        if self.model_parallel:
            model = torch.nn.DataParallel(model)
        return model

    def set_cfg(self, cfg):
        self.cfg = cfg

    def set_model(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg

        # Set up model
        model = build_model(self.cfg)
        model.to(self.device)

        if self.model_parallel:
            model = torch.nn.DataParallel(model)
        self.model = model

    def load_checkpoint(self, ckpt):
        logger.info("Loading checkpoint {}".format(ckpt))
        # workaround for mps (directly loading to map_location=mps results in zeros)
        if (self.device.type == 'mps'):
            model_state=torch.load(ckpt, map_location='cpu')
            self.model.to('cpu')
            self.model.load_state_dict(model_state['model_state'])
            self.model.to(self.device)
        else:
            model_state = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(model_state['model_state'])

    def get_modelname(self):
        return self.model_name

    def get_cfg(self):
        return self.cfg

    def get_num_classes(self):
        return self.cfg.MODEL.NUM_CLASSES

    def get_plane(self):
        return self.cfg.DATA.PLANE

    def get_model_height(self):
        return self.cfg.MODEL.HEIGHT

    def get_model_width(self):
        return self.cfg.MODEL.WIDTH

    def get_max_size(self):
        if self.cfg.MODEL.OUT_TENSOR_WIDTH == self.cfg.MODEL.OUT_TENSOR_HEIGHT:
            return self.cfg.MODEL.OUT_TENSOR_WIDTH
        else:
            return self.cfg.MODEL.OUT_TENSOR_WIDTH, self.cfg.MODEL.OUT_TENSOR_HEIGHT

    def get_device(self):
        return self.device

    @torch.no_grad()
    def eval(self, val_loader, out: torch.Tensor, out_scale=None):
        """Perform prediction and inplace-aggregate views into pred_prob. Return pred_prob."""
        self.model.eval()

        start_index = 0
        from tqdm.contrib.logging import logging_redirect_tqdm
        from tqdm import tqdm
        with logging_redirect_tqdm():
            try:
                for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), unit="batch"):

                    images, scale_factors = batch['image'].to(self.device), batch['scale_factor'].to(self.device)
                    pred = self.model(images, scale_factors, out_scale)

                    # cut prediction to the image size
                    # TODO: a bit "hacky" to get the orig image size of the current plane...; maybe this can just be in batch
                    shape = val_loader.dataset.images.shape
                    pred = pred[:, :, :shape[1], :shape[2]]

                    if self.cfg.DATA.PLANE == "axial":
                        pred = pred.permute(3, 0, 2, 1).to(out.device)  # the to-operation is implicit
                        out[:, start_index:start_index + pred.shape[1], :, :].add_(pred, alpha=0.4)
                        start_index += pred.shape[1]

                    elif self.cfg.DATA.PLANE == "coronal":
                        pred = pred.permute(2, 3, 0, 1).to(out.device)
                        out[:, :, start_index:start_index + pred.shape[2], :].add_(pred, alpha=0.4)
                        start_index += pred.shape[2]

                    else:
                        pred = map_prediction_sagittal2full(pred).permute(0, 3, 2, 1).to(out.device)
                        out[start_index:start_index + pred.shape[0], :, :, :].add_(pred, alpha=0.2)
                        start_index += pred.shape[0]
            except:
                logger.exception("Exception in batch {} of {} inference.".format(batch_idx, self.cfg.DATA.PLANE))
                raise
            else:
                logger.info("Inference on {} batches for {} successful".format(batch_idx, self.cfg.DATA.PLANE))

        return out

    @torch.no_grad()
    def run(self, img_filename, orig_data, orig_zoom, out, noise=0, out_res=None):
        # Set up DataLoader
        test_dataset = MultiScaleOrigDataThickSlices(img_filename, orig_data, orig_zoom, self.cfg, gn_noise=noise,
                                                     transforms=transforms.Compose([ToTensorTest()]))

        test_data_loader = DataLoader(dataset=test_dataset, shuffle=False,
                                      batch_size=self.cfg.TEST.BATCH_SIZE)

        # Run evaluation
        start = time.time()
        out = self.eval(test_data_loader, out, out_scale=out_res)
        logger.info("{}-Inference on {} finished in {:0.4f} seconds".format(self.cfg.DATA.PLANE, img_filename, time.time()-start))

        return out

