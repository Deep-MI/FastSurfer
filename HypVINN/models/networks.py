
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
from typing import Dict
from torch import Tensor, nn
import torch
import FastSurferCNN.models.sub_module as sm
import FastSurferCNN.models.interpolation_layer as il
from FastSurferCNN.models.networks import FastSurferCNNBase
import numpy as np




class HypVINN(FastSurferCNNBase):
    """
          Construct HypVINN object.

          Parameters
          ----------
          params : Dict
              Dictionary of configurations.
          padded_size : int
              Size of image when padded (Default value = 256).
          """
    def __init__(self, params, padded_size=256):
        num_c = params["num_channels"]

        params["num_channels"] = params["num_filters_interpol"]

        super(HypVINN, self).__init__(params)

        # Flex options
        self.height = params["height"]
        self.width = params["width"]

        self.out_tensor_shape = tuple(
            params.get("out_tensor_" + k, padded_size) for k in ["width", "height"]
        )

        self.interpolation_mode = (
            params["interpolation_mode"]
            if "interpolation_mode" in params
            else "bilinear"
        )
        if self.interpolation_mode not in ["nearest", "bilinear", "bicubic", "area"]:
            raise ValueError("Invalid interpolation mode")

        self.crop_position = (
            params["crop_position"] if "crop_position" in params else "top_left"
        )
        if self.crop_position not in [
            "center",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ]:
            raise ValueError("Invalid crop position")

        # Reset input channels to two modalities head number (overwritten in super call)
        params["num_channels"] = num_c // 2

        self.m1_inp_block = sm.InputDenseBlock(params)
        self.m2_inp_block = sm.InputDenseBlock(params)

        # Initialize learneble modality weights
        self.mod_weights = nn.Parameter(torch.ones(2) * 0.5)
        self.normalize_weights = nn.Softmax(dim=0)

        params["num_channels"] = params["num_filters"] + params["num_filters_interpol"]

        self.outp_block = sm.OutputDenseBlock(params)

        self.interpol1 = il.Zoom2d((self.width, self.height),
                                   interpolation_mode=self.interpolation_mode,
                                   crop_position=self.crop_position)

        self.interpol2 = il.Zoom2d(self.out_tensor_shape,
                                   interpolation_mode=self.interpolation_mode,
                                   crop_position=self.crop_position)

        # Classifier logits options
        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, scale_factor, weight_factor, scale_factor_out=None):
        # Weight factor [wT1,wT2] has 3 stages [1,0],[0.5,0.5],[0,1],
        #if the weight factor is [0.5,0.5] the automatically weights (s_weights) are passed
        #If there is a 1 in the comparison the automatically weights will be replace by the first weight_factors pass
        comparison = weight_factor[0]

        x = torch.tensor_split(x, 2, dim=1)
        # Input block + Flex to 1 mm
        skip_encoder_01 = self.m1_inp_block(x[0])
        skip_encoder_02 = self.m2_inp_block(x[1])

        s_weights = self.normalize_weights(self.mod_weights)

        # If one weight 1 it means modality is not available
        if 1 in comparison:
            s_weights = comparison

        mw1 = s_weights[0].float()
        mw2 = s_weights[1].float()

        # Shared latent space
        skip_encoder_0 = mw1 * skip_encoder_01 + mw2 * skip_encoder_02

        encoder_output0, rescale_factor = self.interpol1(skip_encoder_0, scale_factor) # instead of maxpool = encoder_output_0

        # FastSurferCNN Base
        decoder_output1 = super().forward(encoder_output0, scale_factor=scale_factor)

        # Flex to original res
        if scale_factor_out is None:
            scale_factor_out = rescale_factor
        else:
            scale_factor_out = np.asarray(scale_factor_out) * np.asarray(rescale_factor) / np.asarray(scale_factor)

        prior_target_shape = self.interpol2.target_shape
        self.interpol2.target_shape = skip_encoder_0.shape[2:]
        try:
            decoder_output0, sf = self.interpol2(
                decoder_output1, scale_factor_out, rescale=True
            )
        finally:
            self.interpol2.target_shape = prior_target_shape

        outblock = self.outp_block(decoder_output0, skip_encoder_0)
        # Final logits layer
        logits = self.classifier.forward(outblock) # 1x1 convolution

        return logits


_MODELS = {
    "HypVinn": HypVINN,
}


def build_model(cfg) -> HypVINN:
    """
    Build requested model.

    Parameters
    ----------
    cfg : yacs.config.CfgNode
        Node of configs to be used.

    Returns
    -------
    model
        Object of the initialized model.
    """
    if cfg.MODEL.MODEL_NAME not in _MODELS:
        raise AssertionError(f"Model {cfg.MODEL.MODEL_NAME} not supported")
    params = {k.lower(): v for k, v in dict(cfg.MODEL).items()}
    model_type = _MODELS[cfg.MODEL.MODEL_NAME]
    return model_type(params, padded_size=cfg.DATA.PADDED_SIZE)
