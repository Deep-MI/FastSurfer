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

import torch
import numpy as np
import time
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms
from HypVINN.models.networks import build_model
from FastSurferCNN.data_loader.augmentation import ToTensorTest, ZeroPad2DTest
from HypVINN.data_loader.data_utils import hypo_map_prediction_sagittal2full
from HypVINN.data_loader.dataset import HypoVINN_dataset
import FastSurferCNN.utils.logging as logging
from FastSurferCNN.utils.common import find_device

logger = logging.get_logger(__name__)

class Inference:
    def __init__(self, cfg,args):

        self._threads = getattr(args, "threads", 1)
        torch.set_num_threads(self._threads)
        self._async_io = getattr(args, "async_io", False)

        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg

        # Switch on denormal flushing for faster CPU processing
        # seems to have less of an effect on VINN than old CNN
        torch.set_flush_denormal(True)

        # Define device and transfer model
        self.device = find_device(args.device)

        if self.device.type == "cpu" and args.viewagg_device == "auto":
            self.viewagg_device = self.device
        else:
            # check, if GPU is big enough to run view agg on it
            # (this currently takes the memory of the passed device)
            self.viewagg_device = torch.device(
                find_device(
                    args.viewagg_device,
                    flag_name="viewagg_device",
                    min_memory=4 * (2 ** 30),
                )
            )

        logger.info(f"Running view aggregation on {self.viewagg_device}")

        # Options for parallel run
        self.model_parallel = (
                torch.cuda.device_count() > 1
                and self.device.type == "cuda"
                and self.device.index is None
        )

        # Initial model setup
        self.model = self.setup_model(cfg)
        self.model_name = self.cfg.MODEL.MODEL_NAME

        # Initial checkpoint loading
        #self.load_checkpoint(ckpt)

    def setup_model(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg

        # Set up model
        model = build_model(self.cfg)  #
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
        model_state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(model_state["model_state"])

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
        return self.device,self.viewagg_device

    @torch.no_grad()
    def eval(self, val_loader, pred_prob, out_scale=None):
        self.model.eval()

        start_index = 0
        for batch_idx, batch in tqdm(enumerate(val_loader),total=len(val_loader)):

            images, scale_factors,weight_factors = (batch['image'].to(self.device),
                                                    batch['scale_factor'].to(self.device),
                                                    batch['weight_factor'].to(self.device))


            pred = self.model(images, scale_factors, weight_factors, out_scale)

            if self.cfg.DATA.PLANE == "axial":
                pred = pred.permute((2, 3, 0, 1)).to(self.viewagg_device)
                pred_prob[:, :, start_index:start_index + pred.shape[2], :] += torch.mul(pred, 0.4)
                start_index += pred.shape[2]

            elif self.cfg.DATA.PLANE == "coronal":
                pred = pred.permute(2, 0, 3, 1).to(self.viewagg_device)
                pred_prob[:, start_index:start_index + pred.shape[1], :, :] += torch.mul(pred, 0.4)
                start_index += pred.shape[1]

            else:
                pred = hypo_map_prediction_sagittal2full(pred).permute(0, 2, 3, 1).to(self.viewagg_device)
                pred_prob[start_index:start_index + pred.shape[0],:, :, :] += torch.mul(pred, 0.2)
                start_index += pred.shape[0]

        logger.info("--->  {} Model Testing Done.".format(self.cfg.DATA.PLANE))

        return pred_prob

    def run(self, subject_name, modalities, orig_zoom, pred_prob, out_res=None,mode='multi'):
        # Set up DataLoader
        test_dataset = HypoVINN_dataset(subject_name, modalities, orig_zoom, self.cfg, mode = mode,
                                                     transforms=transforms.Compose([ZeroPad2DTest((self.cfg.DATA.PADDED_SIZE, self.cfg.DATA.PADDED_SIZE)), ToTensorTest()]))

        test_data_loader = DataLoader(dataset=test_dataset, shuffle=False,
                                      batch_size=self.cfg.TEST.BATCH_SIZE)

        # Run evaluation
        start = time.time()
        pred_prob = self.eval(test_data_loader, pred_prob, out_scale=out_res)
        logger.info("{} Inference on {} finished in {:0.4f} seconds".format(self.cfg.DATA.PLANE,subject_name, time.time()-start))

        return pred_prob

