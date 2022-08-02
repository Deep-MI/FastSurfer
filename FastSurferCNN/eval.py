
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
import utils.logging_script as logging
import time

from torch.utils.data import DataLoader
from torchvision import transforms
from models.networks import build_model
from data_loader.augmentation import ToTensorTest, ZeroPad2DTest
from data_loader.data_utils import map_prediction_sagittal2full
from data_loader.dataset import MultiScaleOrigDataThickSlices


logger = logging.get_logger(__name__)


class Inference:
    def __init__(self, cfg, ckpt, no_cuda, small_gpu):
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg
        self.no_cuda = no_cuda
        self.small_gpu = small_gpu

        # Set up logging
        logging.setup_logging(cfg.OUT_LOG_DIR, cfg.OUT_LOG_NAME)
        logger.info("Run Inference with config:")
        logger.info(pprint.pformat(cfg))

        # Define device and transfer model
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")

        # Options for parallel run
        if torch.cuda.device_count() > 0 and self.device == "cuda":
            self.model_parallel = True
        else:
            self.model_parallel = False

        # Initial model setup
        self.model = self.setup_model(cfg)
        self.model_name = self.cfg.MODEL.MODEL_NAME

        # Initial checkpoint loading
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
        return self.device

    @torch.no_grad()
    def eval(self, val_loader, pred_prob, out_scale=None):
        self.model.eval()

        start_index = 0
        for batch_idx, batch in enumerate(val_loader):

            images, scale_factors = batch['image'].to(self.device), batch['scale_factor'].to(self.device)
            pred = self.model(images, scale_factors, out_scale)

            if self.cfg.DATA.PLANE == "axial":
                pred = pred.permute(3, 0, 2, 1)
                if self.small_gpu:
                    pred = pred.cpu()
                pred_prob[:, start_index:start_index + pred.shape[1], :, :] += torch.mul(pred, 0.4)
                start_index += pred.shape[1]

            elif self.cfg.DATA.PLANE == "coronal":
                pred = pred.permute(2, 3, 0, 1)
                if self.small_gpu:
                    pred = pred.cpu()
                pred_prob[:, :, start_index:start_index + pred.shape[2], :] += torch.mul(pred, 0.4)
                start_index += pred.shape[2]

            else:
                pred = map_prediction_sagittal2full(pred).permute(0, 3, 2, 1)
                if self.small_gpu:
                    pred = pred.cpu()
                pred_prob[start_index:start_index + pred.shape[0], :, :, :] += torch.mul(pred, 0.2)
                start_index += pred.shape[0]

            logger.info("---> Batch {} {} Testing Done.".format(batch_idx, self.cfg.DATA.PLANE))

        return pred_prob

    def run(self, img_filename, orig_data, orig_zoom, pred_prob, noise=0, out_res=None):
        # Set up DataLoader
        test_dataset = MultiScaleOrigDataThickSlices(img_filename, orig_data, orig_zoom, self.cfg, gn_noise=noise,
                                                     transforms=transforms.Compose([ZeroPad2DTest((self.cfg.DATA.PADDED_SIZE, self.cfg.DATA.PADDED_SIZE)),
                                                                                    ToTensorTest()]))

        test_data_loader = DataLoader(dataset=test_dataset, shuffle=False,
                                      batch_size=self.cfg.TEST.BATCH_SIZE)

        # Run evaluation
        start = time.time()
        pred_prob = self.eval(test_data_loader, pred_prob, out_scale=out_res)
        logger.info("Inference on {} finished in {:0.4f} seconds".format(img_filename, time.time()-start))

        return pred_prob

