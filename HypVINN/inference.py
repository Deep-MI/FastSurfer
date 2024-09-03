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

from time import time
from typing import Optional

import torch
import numpy as np
import yacs.config
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

import FastSurferCNN.utils.logging as logging
from FastSurferCNN.utils.common import find_device
from FastSurferCNN.data_loader.augmentation import ToTensorTest, ZeroPad2DTest
from HypVINN.models.networks import build_model
from HypVINN.data_loader.data_utils import hypo_map_prediction_sagittal2full
from HypVINN.data_loader.dataset import HypVINNDataset
from HypVINN.utils import ModalityMode

logger = logging.get_logger(__name__)


class Inference:
    """
    Class for running inference on a single subject.

    Attributes
    ----------
    model : torch.nn.Module
        The model to use for inference.
    model_name : str
        The name of the model.

    Methods
    -------
    setup_model(cfg)
        Set up the model.
    """
    def __init__(
            self,
            cfg,
            threads: int = -1,
            async_io: bool = False,
            device: str = "auto",
            viewagg_device: str = "auto",
    ):
        """
        Initialize the Inference class.

        This method initializes the Inference class with the provided configuration, number of threads, async IO flag,
        device, and view aggregation device. It sets the random seed, switches on denormal flushing, defines the device,
        and sets up the initial model.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            The configuration node containing the parameters for the model.
        threads : int, optional
            The number of threads to use. Default is -1, which uses all available threads.
        async_io : bool, optional
            Whether to use asynchronous IO. Default is False.
        device : str, optional
            The device to use for computations. Can be 'auto', 'cpu', or 'cuda'. Default is 'auto'.
        viewagg_device : str, optional
            The device to use for view aggregation. Can be 'auto', 'cpu', or 'cuda'. Default is 'auto'.
        """
        self._threads = threads
        torch.set_num_threads(self._threads)
        self._async_io = async_io

        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg

        # Switch on denormal flushing for faster CPU processing
        # seems to have less of an effect on VINN than old CNN
        torch.set_flush_denormal(True)

        # Define device and transfer model
        self.device = find_device(device)

        if self.device.type == "cpu" and viewagg_device == "auto":
            self.viewagg_device = self.device
        else:
            # check, if GPU is big enough to run view agg on it
            # (this currently takes the memory of the passed device)
            self.viewagg_device = torch.device(
                find_device(
                    viewagg_device,
                    flag_name="viewagg_device",
                    min_memory=4 * (2 ** 30),
                )
            )

        logger.info(f"Running view aggregation on {self.viewagg_device}")

        # Initial model setup
        self.model = self.setup_model(cfg)
        self.model_name = self.cfg.MODEL.MODEL_NAME

    def setup_model(
            self,
            cfg: Optional["yacs.config.CfgNode"] = None,
    ) -> torch.nn.Module:
        """
        Set up the model.

        This method sets up the model for inference.

        Parameters
        ----------
        cfg : yacs.config.CfgNode, optional
            The configuration node containing the parameters for the model.

        Returns
        -------
        model : torch.nn.Module
            The model set up for inference.
        """
        if cfg is not None:
            self.cfg = cfg

        # Set up model
        model = build_model(self.cfg)  #
        model.to(self.device)

        return model

    def set_cfg(self, cfg):
        """
        Set the configuration node.

        Parameters
        ----------
        cfg : yacs.config.CfgNode
            The configuration node containing the parameters for the model.
        """
        self.cfg = cfg

    def set_model(self, cfg: yacs.config.CfgNode = None):
        """
        Set the model for the Inference instance.

        Parameters
        ----------
        cfg : yacs.config.CfgNode, optional
            The configuration node containing the parameters for the model. (Default = None).
        """
        if cfg is not None:
            self.cfg = cfg

        # Set up model
        model = build_model(self.cfg)
        model.to(self.device)
        self.model = model

    def load_checkpoint(self, ckpt: str):
        """
        Load a model checkpoint.

        This method loads a model checkpoint from a .pth file containing a state dictionary of a model.

        Parameters
        ----------
        ckpt : str
            The path to the checkpoint file. The checkpoint file should be a .pth file containing a state dictionary
            of a model.
        """
        logger.info("Loading checkpoint {}".format(ckpt))
        # WARNING: weights_only=False can cause unsafe code execution, but here the
        # checkpoint can be considered to be from a safe source
        model_state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(model_state["model_state"])

    def get_modelname(self):
        """
        Get the name of the model.

        This method returns the name of the model used in the Inference instance.

        Returns
        -------
        str
            The name of the model.
        """
        return self.model_name

    def get_cfg(self):
        """
        Get the configuration node.

        This method returns the configuration node used in the Inference instance.

        Returns
        -------
        yacs.config.CfgNode
            The configuration node containing the parameters for the model.
        """
        return self.cfg

    def get_num_classes(self):
        """
        Get the number of classes.

        This method returns the number of classes defined in the model configuration.

        Returns
        -------
        int
            The number of classes.
        """
        return self.cfg.MODEL.NUM_CLASSES

    def get_plane(self):
        """
        Get the plane.

        This method returns the plane defined in the data configuration.

        Returns
        -------
        str
            The plane.
        """
        return self.cfg.DATA.PLANE

    def get_model_height(self):
        """
        Get the model height.

        This method returns the height of the model defined in the model configuration.

        Returns
        -------
        int
            The height of the model.
        """
        return self.cfg.MODEL.HEIGHT

    def get_model_width(self):
        """
        Get the model width.

        This method returns the width of the model defined in the model configuration.

        Returns
        -------
        int
            The width of the model.
        """
        return self.cfg.MODEL.WIDTH

    def get_max_size(self):
        """
        Get the maximum size of the output tensor.

        Returns
        -------
        int or tuple
            The maximum size. If the width and height of the output tensor are equal, it returns the width. Otherwise, it
            returns both the width and height.
        """
        if self.cfg.MODEL.OUT_TENSOR_WIDTH == self.cfg.MODEL.OUT_TENSOR_HEIGHT:
            return self.cfg.MODEL.OUT_TENSOR_WIDTH
        else:
            return self.cfg.MODEL.OUT_TENSOR_WIDTH, self.cfg.MODEL.OUT_TENSOR_HEIGHT

    def get_device(self):
        """
        Get the device.

        This method returns the device and view aggregation device used in the Inference instance.

        Returns
        -------
        tuple
            The device and view aggregation device.
        """
        return self.device,self.viewagg_device

    #TODO check is possible to modify to CerebNet inference mode from RAS directly to LIA (CerebNet.Inference._predict_single_subject)
    @torch.no_grad()
    def eval(self, val_loader: DataLoader, pred_prob: torch.Tensor, out_scale: float = None) -> torch.Tensor:
        """
        Evaluate the model on a HypVINN dataset.

        This method runs the model in evaluation mode on a HypVINN Dataset. It iterates over the given dataset and
        computes the model's predictions.

        Parameters
        ----------
        val_loader : DataLoader
            The DataLoader for the validation set.
        pred_prob : torch.Tensor
            The tensor to update with the prediction probabilities.
        out_scale : float, optional
            The scale factor for the output. Default is None.

        Returns
        -------
        pred_prob: torch.Tensor
            The updated prediction probabilities.
        """
        self.model.eval()

        start_index = 0
        for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):

            images = batch["image"].to(self.device)
            scale_factors = batch["scale_factor"].to(self.device)
            weight_factors = batch["weight_factor"].to(self.device, dtype=torch.float32)

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

    def run(
            self,
            subject_name: str,
            modalities,
            orig_zoom,
            pred_prob,
            out_res=None,
            mode: ModalityMode = "t1t2",
    ):
        """
        Run the inference process on a single subject.

        This method sets up the HypVINN DataLoader for the subject, runs the model in evaluation mode on the subject's
        data,
        and returns the updated prediction probabilities.

        Parameters
        ----------
        subject_name : str
            The name of the subject.
        modalities : ModalityDict
            The modalities of the subject.
        orig_zoom : npt.NDArray[float]
            The original zoom of the subject.
        pred_prob : torch.Tensor
            The tensor to update with the prediction probabilities.
        out_res : float, optional
            The resolution of the output. Default is None.
        mode : ModalityMode, default="t1t2"
            The mode of the modalities. Default is 't1t2'.

        Returns
        -------
        pred_prob: torch.Tensor
            The updated prediction probabilities.
        """
        # Set up DataLoader
        test_dataset = HypVINNDataset(
            subject_name,
            modalities,
            orig_zoom,
            self.cfg,
            mode=mode,
            transforms=transforms.Compose(
                [
                    ZeroPad2DTest(
                        (self.cfg.DATA.PADDED_SIZE, self.cfg.DATA.PADDED_SIZE),
                    ),
                    ToTensorTest(),
                ],
            ),
        )

        test_data_loader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=self.cfg.TEST.BATCH_SIZE,
        )

        # Run evaluation
        start = time()
        pred_prob = self.eval(test_data_loader, pred_prob, out_scale=out_res)
        logger.info(
            f"{self.cfg.DATA.PLANE} Inference on {subject_name} finished in "
            f"{time()-start:0.4f} seconds"
        )

        return pred_prob
