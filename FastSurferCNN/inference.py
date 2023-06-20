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
import time
from typing import Optional, Dict, Tuple, Union
import os

import numpy as np
from numpy import typing as npt
import torch
import yacs.config
from pandas import DataFrame
from torch.utils.data import DataLoader
from torchvision import transforms

from FastSurferCNN.utils import logging
from FastSurferCNN.models.networks import build_model
from FastSurferCNN.data_loader.augmentation import ToTensorTest
from FastSurferCNN.data_loader.data_utils import map_prediction_sagittal2full
from FastSurferCNN.data_loader.dataset import MultiScaleOrigDataThickSlices


logger = logging.getLogger(__name__)


class Inference:
    """
    Model evaluation class to run inference using FastSurferCNN

    Functions:
        __init__(cfg, device, ckpt): Constructor
        setup_model(cfg, device): Set up the initial model
        set_cfg(cfg): Set configuration node
        to(device): Moves and/or casts the parameters and buffers.
        load_checkpoint(ckpt): function to load the checkpoint
        eval(init_pred, val_loader, *, out_scale, out): evaluate predictions
        run(init_pred, img_filename, orig_data, orig_zoom, out, out_res, batch_size): run the loaded model

    Attributes:
        permute_order (Dict[str, Tuple[int, int, int, int]]): permutation order for axial, coronal, and sagittal
        device (Optional[torch.device]): device specification for distributed computation usage.
        default_device (torch.device): default device specification for distributed computation usage.
        cfg (yacs.config.CfgNode): configuration Node
        model_parallel (bool): option for parallel run
        model (torch.nn.Module): neural network model
        model_name (str): name of the model
        alpha (dict[str, float]): [help]
        post_prediction_mapping_hook (): [help]

    """

    permute_order: Dict[str, Tuple[int, int, int, int]]
    device: Optional[torch.device]
    default_device: torch.device

    def __init__(
            self,
            cfg: yacs.config.CfgNode,
            device: torch.device,
            ckpt: str = "",
            lut: Union[None, str, np.ndarray, DataFrame] = None
    ):
        """ Constructor

        Args:
            cfg: configuration Node
            device: device specification for distributed computation usage.
            ckpt: string or os.PathLike object containing the name to the checkpoint file
        """

        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg

        # Switch on denormal flushing for faster CPU processing
        # seems to have less of an effect on VINN than old CNN
        torch.set_flush_denormal(True)

        self.default_device = device

        # Options for parallel run
        self.model_parallel = (
            torch.cuda.device_count() > 1
            and self.default_device.type == "cuda"
            and self.default_device.index is None
        )

        # Initial model setup
        self.model = None
        self._model_not_init = None
        self.setup_model(cfg, device=self.default_device)
        self.model_name = self.cfg.MODEL.MODEL_NAME

        self.alpha = {"sagittal": 0.2}
        self.permute_order = {
            "axial": (3, 0, 2, 1),
            "coronal": (2, 3, 0, 1),
            "sagittal": (0, 3, 2, 1),
        }
        self.lut = lut

        # Initial checkpoint loading
        if ckpt:
            # this also moves the model to the para
            self.load_checkpoint(ckpt)

    def setup_model(self, cfg=None, device: torch.device = None):
        """
        function to set up the model

        Args:
            cfg (yacs.config.CfgNode): configuration Node
            device (torch.device): device specification for distributed computation usage.
        """

        if cfg is not None:
            self.cfg = cfg
        if device is None:
            device = self.default_device

        # Set up model
        self._model_not_init = build_model(
            self.cfg
        )  # ~ model = FastSurferCNN(params_network)
        self._model_not_init.to(device)
        self.device = None

    def set_cfg(self, cfg: yacs.config.CfgNode):
        self.cfg = cfg

    def to(self, device: Optional[torch.device] = None):
        """
        Moves and/or casts the parameters and buffers.

        Args:
            device: the desired device of the parameters and buffers in this module
        """

        if self.model_parallel:
            raise RuntimeError(
                "Moving the model to other devices is not supported for multi-device models."
            )
        _device = self.default_device if device is None else device
        self.device = _device
        self.model.to(device=_device)

    def load_checkpoint(self, ckpt: Union[str, os.PathLike]):
        """
        function to load the checkpoint and set device and model

        Args:
            ckpt: string or os.PathLike object containing the name to the checkpoint file
        """

        logger.info("Loading checkpoint {}".format(ckpt))

        self.model = self._model_not_init
        # If device is None, the model has never been loaded (still in random initial configuration)
        if self.device is None:
            self.device = self.default_device

        # workaround for mps (directly loading to map_location=mps results in zeros)
        device = self.device
        if self.device.type == "mps":
            self.model.to("cpu")
            device = "cpu"
        else:
            # make sure the model is, where it is supposed to be
            self.model.to(self.device)

        model_state = torch.load(ckpt, map_location=device)
        self.model.load_state_dict(model_state["model_state"])

        # workaround for mps (move the model back to mps)
        if self.device.type == "mps":
            self.model.to(self.device)

        if self.model_parallel:
            self.model = torch.nn.DataParallel(self.model)

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
    def eval(
            self,
            init_pred: torch.Tensor,
            val_loader: DataLoader,
            *,
            out_scale: Optional = None,
            out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform prediction and inplace-aggregate views into pred_prob.

        Args:
            init_pred: initial prediction
            val_loader : value loader
            out_scale: [help]
            out: previous prediction tensor

        Returns:
            prediction probability tensor
        """

        self.model.eval()
        # we should check here, whether the DataLoader is a Random or a SequentialSampler, but we cannot easily.
        if not isinstance(val_loader.sampler, torch.utils.data.SequentialSampler):
            logger.warning(
                "The Validation loader seems to not use the SequentialSampler. This might interfere with "
                "the assumed sorting of batches."
            )

        start_index = 0
        plane = self.cfg.DATA.PLANE
        index_of_current_plane = self.permute_order[plane].index(0)
        target_shape = init_pred.shape
        ii = [slice(None) for _ in range(4)]
        pred_ii = tuple(slice(i) for i in target_shape[:3])

        from tqdm.contrib.logging import logging_redirect_tqdm
        from tqdm import tqdm

        if out is None:
            out = init_pred.detach().clone()
        with logging_redirect_tqdm():
            try:
                for batch_idx, batch in tqdm(
                    enumerate(val_loader), total=len(val_loader), unit="batch"
                ):

                    # move data to the model device
                    images, scale_factors = batch["image"].to(self.device), batch[
                        "scale_factor"
                    ].to(self.device)

                    # predict the current batch, outputs logits
                    pred = self.model(images, scale_factors, out_scale)
                    batch_size = pred.shape[0]
                    end_index = start_index + batch_size

                    # check if we need a special mapping (e.g. as for sagittal)
                    if self.get_plane() == "sagittal":
                        pred = map_prediction_sagittal2full(
                            pred, num_classes=self.get_num_classes(), lut=self.lut
                        )

                    # permute the prediction into the out slice order
                    pred = pred.permute(*self.permute_order[plane]).to(
                        out.device
                    )  # the to-operation is implicit

                    # cut prediction to the image size
                    pred = pred[pred_ii]

                    # add prediction logits into the output (same as multiplying probabilities)
                    ii[index_of_current_plane] = slice(start_index, end_index)
                    out[tuple(ii)].add_(pred, alpha=self.alpha.get(plane, 0.4))
                    start_index = end_index

            except:
                logger.exception(
                    "Exception in batch {} of {} inference.".format(batch_idx, plane)
                )
                raise
            else:
                logger.info(
                    "Inference on {} batches for {} successful".format(
                        batch_idx + 1, plane
                    )
                )

        return out

    @torch.no_grad()
    def run(
            self,
            init_pred: torch.Tensor,
            img_filename: str,
            orig_data: npt.NDArray,
            orig_zoom: npt.NDArray,
            out: Optional[torch.Tensor] = None,
            out_res: Optional[int] = None,
            batch_size: int = None
    ) -> torch.Tensor:
        """ [help]
        Run the loaded model on the data (T1) from orig_data and filename img_filename with scale factors orig_zoom.

        Args:
            init_pred: initial prediction
            img_filename: original image filename
            orig_data: original image data
            orig_zoom: original zoom
            out: updated output tensor (Default = None)
            out_res: output resolution
            batch_size: batch size (Default = None)

        Returns:
            prediction probability tensor
        """

        # Set up DataLoader
        test_dataset = MultiScaleOrigDataThickSlices(
            orig_data,
            orig_zoom,
            self.cfg,
            transforms=transforms.Compose([ToTensorTest()]),
        )

        test_data_loader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=self.cfg.TEST.BATCH_SIZE if batch_size is None else batch_size,
        )

        # Run evaluation
        start = time.time()
        out = self.eval(init_pred, test_data_loader, out=out, out_scale=out_res)
        time_delta = time.time() - start
        logger.info(
            f"{self.cfg.DATA.PLANE.capitalize()} inference on {img_filename} finished in "
            f"{time_delta:0.4f} seconds"
        )

        return out
